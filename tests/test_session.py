"""The persistent-shell runner, against a fake remote shell.

A real fleet is not in a unit test; what is testable without one is the protocol the
runner speaks and the rules it keeps — one request at a time, a retry never lands on a
stream that may still be busy, a dead shell is reopened, the pool stays in budget. Each
of those is a decision that would be very expensive to get wrong against a bastion
(duplicate command, spliced reply, login storm).
"""

from __future__ import annotations

import asyncio

import pytest

from jms_top.config import Bastion
from jms_top.inventory import Host
from jms_top.session import END, SessionRunner
from jms_top.ssh import ExecResult, SSHRunner


class FakeStdin:
    def __init__(self, proc):
        self._proc = proc

    def write(self, data: bytes) -> None:
        self._proc.requests.append(data)
        # The far side answers each request with its configured reply.
        self._proc.replies.append(self._proc.answer(data))

    async def drain(self) -> None:
        await asyncio.sleep(0)


class FakeStdout:
    def __init__(self, proc):
        self._proc = proc

    async def read(self, _n: int) -> bytes:
        proc = self._proc
        await proc.gate.wait()
        proc.gate.clear()
        if proc.closed:
            return b""
        if proc.delay and proc.replies:
            await asyncio.sleep(proc.delay)
        if not proc.replies:
            return b""
        return proc.replies.pop(0)


class FakeProc:
    """A shell that answers ``<sentinel>…<END><n>`` for every request written to it."""

    def __init__(self, runner: SessionRunner, *, answer=None, delay: float = 0.0):
        self.runner = runner
        self.answer_fn = answer
        self.delay = delay
        self.requests: list[bytes] = []
        self.replies: list[bytes] = []
        self.gate = asyncio.Event()
        self.closed = False
        self.returncode = None
        self.killed = False
        self.pid = 4242
        self.stdin = FakeStdin(self)
        self.stdout = FakeStdout(self)
        self.stderr = None

    def answer(self, data: bytes) -> bytes:
        runner = self.runner
        text = data.decode()
        marker = END + str(runner._nonce)
        body = self.answer_fn(text) if self.answer_fn else f"jms-top-ok\nout for {marker}"
        self.gate.set()
        return f"{body}\n{marker}\n".encode()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.closed = True
        self.gate.set()

    def _kill_group(self) -> None:  # what the runner calls on the real process group
        self.kill()

    async def wait(self) -> None:
        await asyncio.sleep(0)


def make_runner(**kw) -> SessionRunner:
    runner = SessionRunner(Bastion(url="https://bastion.invalid", login="u", max_parallel=4), **kw)
    runner.spawned = []  # type: ignore[attr-defined]
    runner.behaviour = {}  # type: ignore[attr-defined]
    return runner


def install_spawn(monkeypatch, runner: SessionRunner) -> None:
    """Replace process creation with a fake whose behaviour the test sets per host."""

    import jms_top.session as session_module

    # The runner kills the process *group* (so a password helper goes too); against a
    # fake there is no group, so that call is routed to the fake's kill().
    monkeypatch.setattr(session_module, "_kill_group", lambda proc: proc._kill_group())

    async def fake_exec(*argv, **kw):
        target = argv[-2]
        proc = FakeProc(runner, **(runner.behaviour.get(target, {})))
        runner.spawned.append((target, proc))  # type: ignore[attr-defined]
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


HOSTS = [Host(name=f"h{i}", asset_id=f"id-{i}", account="root") for i in range(6)]


def test_one_session_serves_many_requests(monkeypatch):
    """The point of the whole module: a second poll does not log in again."""
    runner = make_runner()
    install_spawn(monkeypatch, runner)

    async def run():
        host = HOSTS[0]
        first = await runner.exec_on(host, "echo hi")
        second = await runner.exec_on(host, "echo hi")
        assert first.exit_code == 0 and second.exit_code == 0
        assert len(runner.spawned) == 1, "one shell, two requests"
        await runner.close()

    asyncio.run(run())


def test_requests_to_one_host_never_overlap(monkeypatch):
    """A shell is a single stream: two scripts interleaving would splice their replies."""
    runner = make_runner()
    runner.behaviour = {}
    install_spawn(monkeypatch, runner)

    async def run():
        host = HOSTS[0]
        results = await asyncio.gather(*(runner.exec_on(host, f"echo {i}") for i in range(5)))
        assert all(r.exit_code == 0 for r in results)
        # Each reply names its own request; a spliced stream would not.
        bodies = [r.stdout.strip().splitlines()[-1] for r in results]
        assert len(set(bodies)) == 5
        await runner.close()

    asyncio.run(run())


def test_a_timeout_discards_the_shell_and_is_not_replayed(monkeypatch):
    """The far side may still be running the old script; resending would splice it."""
    runner = make_runner()
    install_spawn(monkeypatch, runner)

    async def run():
        host = HOSTS[0]
        await runner.exec_on(host, "echo ok")
        assert len(runner.spawned) == 1
        # Make the next answer arrive too late.
        proc = runner.spawned[0][1]
        proc.delay = 10.0
        result = await runner.exec_on(host, "echo slow", timeout=0.05)
        assert result.timed_out and result.exit_code == 124
        assert proc.killed, "a timed-out request must not leave the old script running"
        # The next poll reopens instead of reusing the half-used stream.
        proc.delay = 0.0
        again = await runner.exec_on(host, "echo again", timeout=2)
        assert again.exit_code == 0
        assert len(runner.spawned) == 2

    asyncio.run(run())


def test_a_closed_shell_is_reopened_on_the_next_request(monkeypatch):
    runner = make_runner()
    install_spawn(monkeypatch, runner)

    async def run():
        host = HOSTS[0]
        assert (await runner.exec_on(host, "echo one")).exit_code == 0
        runner.spawned[0][1].closed = True
        runner.spawned[0][1].returncode = 1
        result = await runner.exec_on(host, "echo two")
        assert result.exit_code == 0
        assert len(runner.spawned) == 2

    asyncio.run(run())


def test_the_pool_closes_the_least_recently_used_session(monkeypatch):
    runner = make_runner(pool_size=2)
    install_spawn(monkeypatch, runner)

    async def run():
        for host in HOSTS[:2]:
            await runner.exec_on(host, "echo a")
        # Touching h0 makes h1 the least recently used.
        await runner.exec_on(HOSTS[0], "echo a")
        await runner.exec_on(HOSTS[2], "echo a")
        open_names = list(runner.sessions())
        assert len(open_names) == 2
        assert "h1" not in open_names
        killed = [p for _t, p in runner.spawned if p.killed]
        assert killed, "the evicted session must be closed, not leaked"
        await runner.close()

    asyncio.run(run())


def test_concurrent_first_requests_to_one_host_share_one_login(monkeypatch):
    """A burst of polls for a host that is not connected yet must not log in N times."""
    runner = make_runner()
    install_spawn(monkeypatch, runner)

    async def run():
        host = HOSTS[0]
        results = await asyncio.gather(*(runner.exec_on(host, "echo hi") for _ in range(4)))
        assert all(r.exit_code == 0 for r in results)
        assert len(runner.spawned) == 1
        await runner.close()

    asyncio.run(run())


def test_an_idle_session_is_closed_and_reopened(monkeypatch):
    now = [0.0]
    runner = make_runner(idle_timeout=60.0, clock=lambda: now[0])
    install_spawn(monkeypatch, runner)

    async def run():
        host = HOSTS[0]
        await runner.exec_on(host, "echo a")
        now[0] += 61.0
        await runner.exec_on(host, "echo b")
        assert len(runner.spawned) == 2
        assert runner.spawned[0][1].killed
        await runner.close()

    asyncio.run(run())


def test_a_failed_login_is_a_result_not_an_exception(monkeypatch):
    runner = make_runner()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)

    async def run():
        result = await runner.exec_on(HOSTS[0], "echo hi")
        assert isinstance(result, ExecResult)
        assert result.exit_code != 0
        assert "Boom" in result.stderr

    asyncio.run(run())


async def _boom(*_a, **_k):
    raise OSError("Boom")


def test_the_request_redirects_stderr_on_the_far_side():
    """Only ssh's own stderr stays on the local pipe; the script's is dropped remotely."""
    runner = make_runner()
    request = runner._request("echo hi")
    assert request.startswith("{ echo hi")
    assert "2>/dev/null" in request
    assert f"echo {END}" in request


def test_the_runner_is_still_an_ssh_runner():
    """Addressing, argv and the password prefix are inherited, not reimplemented."""
    runner = make_runner()
    assert isinstance(runner, SSHRunner)
    assert runner.argv(HOSTS[0])[-1] == "bash -s"


def test_unknown_backend_is_a_clear_error():
    from jms_top.monitor import MonitorError, make_runner

    with pytest.raises(MonitorError, match="unknown ssh backend"):
        make_runner(Bastion(url="https://bastion.invalid"), backend="carrier-pigeon")
