"""A persistent remote shell per host, so a poll stops paying for a login.

A probe is cheap to *run* and expensive to *start*. Measured on the deployment this was
built for: a warm ControlMaster still took 2-30 s per command (median 4-9 s), because
every poll forked ``ssh``, opened a channel, started a remote shell and tore all three
down again. The same script on a shell that is already up answers in **0.4 s** (eight
hosts in parallel: 0.57 s for the whole round).

So this runner keeps one ``ssh … bash`` per host and speaks a tiny protocol over it:

* a request is the script followed by a line printing :data:`END` on stdout, with
  stderr redirected to /dev/null on the far side — a monitor's probe output is
  deliberately bounded and nothing in it needs stderr, while a chatty host's banner or
  a vendor tool's warnings would otherwise have to be drained by someone;
* a reply is everything up to that marker line.

The rules that make it safe to keep a shell alive:

* **One request at a time per host.** A session is a single stream; two scripts sharing
  it would interleave their output. The per-host ``asyncio.Lock`` is the guarantee, and
  a probe's timeout is measured from when *this* request actually starts writing.
* **A retry only before the script is sent.** A timeout or a dropped connection leaves
  the far side possibly still running the old script; resending onto that stream would
  splice two replies together. The session is torn down and the next *poll* reconnects —
  the frame that failed is reported as failed, never replayed.
* **An idle session is closed.** :data:`IDLE_TIMEOUT` keeps abandoned hosts from holding
  a bastion connection (and its audit session) open for hours; the capacity is also
  capped by a least-recently-used pool, because 77 resident shells on one account is
  exactly the sort of thing a gateway limits.
* **A password never goes on a command line.** The ``sshpass -e`` prefix SSHRunner
  already uses is reused as-is: the password reaches the child through ``SSHPASS``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import Bastion
from .inventory import Host
from .probe import PING_SCRIPT, SENTINEL, with_sentinel
from .ssh import ExecResult, SSHRunner, _kill_group, require_ssh

#: The line a reply ends with. Chosen the way :data:`jms_top.probe.SENTINEL` was: a
#: process command line on the far side could print anything, so the marker has to be
#: unlikely and is also checked for a nonce.
END = "__JMS_TOP_END__"

#: How long a session survives without being asked anything.
IDLE_TIMEOUT = 600.0

#: How many sessions may be open at once. The rest are closed least-recently-used
#: first and reopened by the next poll that needs them; a few of them starting cold at
#: once is far cheaper than holding a shell on every asset in a 77-host inventory.
POOL_SIZE = 24

#: How long a new session's shell gets to answer a "are you there" round trip before it
#: is considered unusable — that is login, handshake and remote-shell start.
CONNECT_TIMEOUT = 45.0


@dataclass
class Session:
    """One remote shell and the little state needed to trust it."""

    host: Host
    proc: asyncio.subprocess.Process
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)
    #: Bumped whenever the process is replaced; a reply that arrives for an older
    #: generation belongs to a script whose result nobody wants any more.
    generation: int = 0
    alive: bool = True

    @property
    def pid(self) -> int:
        return self.proc.pid


class SessionRunner(SSHRunner):
    """Runs scripts on hosts through a shell that is already logged in.

    Subclasses :class:`SSHRunner` for the addressing (login string, control path, argv,
    password handling) — this class only changes *how* a script is sent.
    """

    def __init__(
        self,
        bastion: Bastion,
        *,
        max_parallel: int | None = None,
        pool_size: int = POOL_SIZE,
        idle_timeout: float = IDLE_TIMEOUT,
        connect_timeout: float = CONNECT_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(bastion, max_parallel=max_parallel)
        self.pool_size = max(1, int(pool_size))
        self.idle_timeout = max(0.0, float(idle_timeout))
        self.connect_timeout = max(1.0, float(connect_timeout))
        self._clock = clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        #: Set while a host's session is being (re)established, so a burst of polls for
        #: the same host waits rather than logging in three times.
        self._opening: dict[str, asyncio.Future] = {}
        self._nonce = 0

    # --- the pool ---------------------------------------------------------------

    async def session_for(self, host: Host) -> Session:
        """The open session for ``host``, opening one if needed.

        A burst of polls for a host that is not connected yet must cost one login, not
        one per poll: the first caller opens, the rest await the same future. That
        future carries the *opened session*, never the channel-protocol exchange — a
        waiter must not need the session's own lock to receive the session.
        """
        key = host.name
        while True:
            session = self._sessions.get(key)
            if session is not None and self._usable(session):
                self._sessions.move_to_end(key)
                return session
            pending = self._opening.get(key)
            if pending is not None:
                # A plain await: the opener resolves this with a Session, and nobody
                # else may cancel it out from under the other waiters.
                return await asyncio.shield(pending)
            if session is not None:
                await self._retire(key)
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._opening[key] = future
            try:
                await self._make_room()
                session = await self._open(host)
            except BaseException as exc:
                future.set_exception(exc)
                future.exception()  # retrieved here: waiters may all have gone
                raise
            else:
                future.set_result(session)
                return session
            finally:
                self._opening.pop(key, None)

    def _usable(self, session: Session) -> bool:
        if not session.alive or session.proc.returncode is not None:
            return False
        return self._clock() - session.last_used <= self.idle_timeout

    async def _make_room(self) -> None:
        """Close idle sessions first, then the least recently used, to stay in budget."""
        for key in list(self._sessions):
            session = self._sessions[key]
            if not self._usable(session):
                await self._retire(key)
        while len(self._sessions) >= self.pool_size:
            key = next(iter(self._sessions))
            await self._retire(key)

    async def _open(self, host: Host) -> Session:
        """Start the shell and prove it answers before anybody sends a real script."""
        require_ssh()
        # `bash`, not `bash -s`: the shell has to stay up between scripts, so its stdin
        # is the channel this runner writes to for the life of the session.
        argv = self.argv(host)[:-1] + ["bash"]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # stderr is redirected to /dev/null on the far side by `_request`, so this
            # pipe only ever carries ssh's own complaints — a login failure, say.
            stderr=asyncio.subprocess.PIPE,
            env=self._env(host),
            start_new_session=True,  # own group: the kill has to reach sshpass too
        )
        session = Session(host=host, proc=proc)
        self._sessions[host.name] = session
        try:
            # Prove the shell answers before anyone sends a real script, and do it under
            # the session lock: the first caller holds it and a concurrent poll waits.
            async with session.lock:
                await self._send(session, self._request(PING_SCRIPT), self.connect_timeout)
            session.last_used = self._clock()
        except BaseException:
            await self._retire(host.name)
            raise
        return session

    async def _retire(self, key: str) -> None:
        session = self._sessions.pop(key, None)
        if session is None:
            return
        session.alive = False
        session.generation += 1
        with contextlib.suppress(Exception):
            _kill_group(session.proc)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(session.proc.wait(), 5)

    def sessions(self) -> dict[str, Session]:
        return dict(self._sessions)

    # --- speaking to the shell --------------------------------------------------

    def _request(self, script: str, *, requires_sentinel: bool = False) -> str:
        """The text for one request: the script, then a line that proves it finished.

        stderr is dropped on the far side rather than read here: draining two streams
        per request needs either a permanent reader task per session or a select, and
        neither is worth it for a monitor whose probe output is bounded by design. SSH's
        own stderr stays local, on the pipe this process holds.
        """
        self._nonce += 1
        body = with_sentinel(script) if requires_sentinel else script
        return f"{{ {body}\n}} 2>/dev/null; echo {END}{self._nonce}\n"

    async def _send(self, session: Session, request: str, timeout: float) -> str:
        """Write one request and read to its end marker. Caller holds ``session.lock``."""
        marker = f"\n{END}{self._nonce}".encode()
        proc = session.proc
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(request.encode())
        try:
            await asyncio.wait_for(proc.stdin.drain(), timeout)
        except (TimeoutError, asyncio.TimeoutError):
            raise TimeoutError("timed out sending to the remote shell") from None
        buffer = b""
        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(65536), timeout)
            except (TimeoutError, asyncio.TimeoutError):
                raise TimeoutError(f"timed out after {timeout:g}s") from None
            if not chunk:
                raise ConnectionError(self._death_reason(session))
            buffer += chunk
            position = buffer.find(marker)
            if position >= 0:
                return buffer[:position].decode(errors="replace")
            if len(buffer) > 32 * 1024 * 1024:
                raise ConnectionError("remote shell produced more than 32 MiB for one request")

    def _death_reason(self, session: Session) -> str:
        """Why the shell closed mid-request, as far as ssh's stderr says."""
        proc = session.proc
        text = ""
        if proc.stderr is not None:
            with contextlib.suppress(Exception):
                text = proc.stderr.read_nowait().decode(errors="replace").strip()
        detail = text.splitlines()[-1] if text else "the remote shell closed"
        return f"session closed: {detail[:300]}"

    # --- execution --------------------------------------------------------------

    async def exec_on(self, host: Host, script: str, *, timeout: float = 60.0) -> ExecResult:
        """Run ``script`` on ``host`` over its shell; never raises for a far-side failure."""
        try:
            session = await asyncio.wait_for(self.session_for(host), self.connect_timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return ExecResult(
                host.name, 124, "", f"no session within {self.connect_timeout:g}s", True
            )
        except Exception as exc:  # noqa: BLE001 - one host's failure is that host's
            return ExecResult(host.name, 255, "", f"{type(exc).__name__}: {exc}", False)
        async with session.lock:
            if not self._usable(session):
                # A long queue behind one host may have outlived its shell.
                await self._retire(host.name)
                return ExecResult(host.name, 255, "", "session expired before this request", False)
            session.last_used = self._clock()
            generation = session.generation
            try:
                out = await self._send(session, self._request(script), timeout)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                # The far side may still be running the old script on this stream, so
                # the shell is discarded rather than reused: the next poll reconnects.
                await self._retire(host.name)
                return ExecResult(host.name, 124, "", str(exc), True)
            except Exception as exc:  # noqa: BLE001 - a dead shell is this host's failure
                await self._retire(host.name)
                return ExecResult(host.name, 255, "", f"{type(exc).__name__}: {exc}", False)
            if session.generation != generation:  # pragma: no cover - defensive
                return ExecResult(host.name, 255, "", "session was replaced mid-request", False)
            session.last_used = self._clock()
        return ExecResult(host.name, 0, out, "")

    async def ping(self, host: Host, *, timeout: float = 20.0) -> ExecResult:
        return await self.exec_on(host, PING_SCRIPT, timeout=timeout)

    # --- teardown ---------------------------------------------------------------

    async def close(self) -> None:
        for key in list(self._sessions):
            await self._retire(key)
        await super().close()


def sentinel_in(body: str) -> bool:
    """Whether a reply carries the reachability sentinel (used by the `--ping` path)."""
    return SENTINEL in body[:2048]
