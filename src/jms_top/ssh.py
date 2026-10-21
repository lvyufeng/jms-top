"""The SSH data plane: run one script per host through the JumpServer KoKo gateway.

This is the piece that actually touches machines, and the two things it gets right are
the two things jms-bridge learned the hard way:

**Control multiplexing.** A KoKo handshake costs seconds, and the deployment this was
built against has been measured at ~200 s time-to-first-byte for large buffered
responses. So one OpenSSH ControlMaster is opened per host and reused: N polls cost
one handshake, not N. The socket lives in a short temp path, because ``ControlPath``
must fit in a Unix socket's 104-byte limit and a long host name overflows it.

**A dead asset does not look alive.** KoKo answers a rejection — no permission, asset
offline, bad address — with **exit code 0** and the explanation on *stdout*. A caller
that trusts the exit status therefore sees a healthy host. Every reply is prefixed with
:data:`SENTINEL` by the script and checked by :func:`reached`, which is the only thing
that makes a liveness check trustworthy.

Scripts are piped to ``bash -s`` over stdin rather than passed as arguments: multi-line
scripts need no quoting, and nothing from the local shell is ever interpreted.

Unlike jms-bridge's runner this one never sources an ``env_setup`` or ``cd``s into a
workdir — a *monitor* must work on a host whose job environment is broken, because
"the env script fails" is exactly when you want the screen to still say the machine is
up and idle. That is why :meth:`SSHRunner.exec_on` here has no such parameters.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Bastion, ConfigError
from .inventory import Host
from .probe import SENTINEL

#: ``SENTINEL`` is defined in :mod:`jms_top.probe` — the module whose scripts print it —
#: and re-exported here, so the data plane and the parser can never disagree about the
#: string that proves a reply came from the far side.

#: Options every ssh invocation shares. ``accept-new`` trusts a host key on first use
#: and then refuses a changed one — the middle ground that makes an unattended monitor
#: work without disabling verification entirely.
_BASE_OPTS = [
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "ConnectTimeout=30",
]

#: Suppresses interactive prompts. Applied only when a password is *not* the thing
#: being answered — ``BatchMode=yes`` turns the password prompt off, which would leave
#: ``sshpass`` with nothing to feed.
_BATCH_MODE = ["-o", "BatchMode=yes"]


@dataclass
class ExecResult:
    """What one host's command returned."""

    host: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def reached(result: ExecResult) -> bool:
    """Whether the command provably ran on the far side (see :data:`SENTINEL`)."""
    return result.ok and SENTINEL in result.stdout[:2048]


def failure_reason(result: ExecResult) -> str:
    """One line saying why a host did not answer — KoKo's rejection text if that is it.

    The explanation is on stdout for a KoKo rejection (with exit code 0), on stderr for
    a genuine failure, and absent entirely for a timeout. Checked in that order of
    usefulness, so the screen shows "没有资产或资产未激活" rather than "exit 0".
    """
    if result.timed_out:
        return result.stderr.strip()[:300] or f"timed out (exit {result.exit_code})"
    if not result.ok:
        return (result.stderr or result.stdout).strip()[:300] or f"exit {result.exit_code}"
    text = result.stdout.strip()
    for line in text.splitlines():
        line = line.strip()
        if line and line != SENTINEL:
            return line[:300]
    return (result.stderr.strip() or "no reply")[:300]


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """Kill a spawned ``ssh``/``sshpass`` **and** its children, synchronously.

    ``sshpass`` spawns ``ssh``, so killing the child alone leaves the real ssh running —
    and a surviving ssh keeps the ControlMaster socket alive, which is how a later call
    ends up reusing a half-dead channel (a short timeout turning into a long call that
    wrongly reports success).

    Never signals our own process group: a child spawned without ``start_new_session``
    shares it, and ``killpg(getpgid(0))`` would take down the whole monitor.
    """
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        if pgid == os.getpgid(0):
            proc.kill()
        else:
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


class SSHRunner:
    """Runs scripts on inventory hosts through KoKo, multiplexed per host."""

    def __init__(self, bastion: Bastion, *, max_parallel: int | None = None):
        self.bastion = bastion
        # A cap below 1 would build a Semaphore nothing can acquire — a hang, which is
        # the worst way for a fan-out to fail.
        self.max_parallel = max(1, int(max_parallel or bastion.max_parallel))
        self._sem = asyncio.Semaphore(self.max_parallel)
        self._sockets: Path | None = None

    # --- addressing -------------------------------------------------------------

    @property
    def socket_dir(self) -> Path:
        """Short, private directory for the control sockets (created on first use)."""
        if self._sockets is None:
            # Short because ControlPath must fit in a Unix socket path (104 bytes on
            # macOS); a long TMPDIR plus a long host name overflows it.
            self._sockets = Path(tempfile.mkdtemp(prefix="jms-top-", dir=None))
        self._sockets.mkdir(parents=True, exist_ok=True)
        return self._sockets

    def control_path(self, host: Host) -> Path:
        digest = hashlib.sha256(host.name.encode()).hexdigest()[:12]
        return self.socket_dir / f"{digest}.sock"

    def target(self, host: Host) -> str:
        """The ssh destination for ``host``: KoKo's ``login#account#asset@host``."""
        if host.is_direct:
            return host.ssh_host or host.name
        return f"{host.login_string(self.bastion.login)}@{self.bastion.host}"

    def port(self, host: Host) -> int:
        if host.is_direct:
            return host.ssh_port or 22
        return self.bastion.ssh_port

    @property
    def password_auth(self) -> bool:
        return bool(self.bastion.ssh_password)

    def _prefix(self, host: Host) -> list[str]:
        """``sshpass -e`` when the bastion needs a password, otherwise nothing."""
        if host.is_direct or not self.password_auth:
            return []
        if shutil.which("sshpass") is None:
            raise ConfigError(
                "an ssh password is configured but `sshpass` is not installed "
                "(brew/apt install sshpass) — or register an SSH key with JumpServer instead"
            )
        return ["sshpass", "-e"]  # password arrives via the SSHPASS environment variable

    def argv(self, host: Host) -> list[str]:
        """The full ssh command line for a script piped to ``bash -s``."""
        argv = self._prefix(host) + ["ssh", *_BASE_OPTS]
        if not (self.password_auth and not host.is_direct):
            argv += _BATCH_MODE
        if host.is_direct and host.ssh_user:
            argv += ["-l", host.ssh_user]
        key = self.bastion.ssh_key
        if key:
            argv += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
        if self.password_auth and not host.is_direct:
            argv += ["-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password"]
        argv += [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self.control_path(host)}",
            "-o",
            "ControlPersist=300",
        ]
        port = self.port(host)
        if port != 22:
            argv += ["-p", str(port)]
        return argv + [self.target(host), "bash -s"]

    def _env(self, host: Host | None) -> dict[str, str]:
        env = {**os.environ, "TERM": os.environ.get("TERM", "dumb")}
        # The password goes only to the child that will use it: a direct host is
        # key-based, and cleanup commands talk to an existing socket.
        if host is not None and not host.is_direct and self.password_auth:
            env["SSHPASS"] = self.bastion.ssh_password or ""
        return env

    def probe(self) -> str:
        """A one-line script used to establish the connection, authenticated first.

        Deliberately the cheapest possible thing: the point is the handshake, and it
        keeps a first-run fleet visible while the expensive probes are still running.
        """
        return f"echo {SENTINEL}\n"

    # --- execution --------------------------------------------------------------

    async def exec_on(self, host: Host, script: str, *, timeout: float = 60.0) -> ExecResult:
        """Run ``script`` on ``host``; never raises for a far-side failure."""
        if shutil.which("ssh") is None:
            raise ConfigError("`ssh` is not installed — install an OpenSSH client")
        argv = self.argv(host)
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(host),
                start_new_session=True,  # own group, so the kill reaches sshpass too
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(script.encode()), timeout)
            except (TimeoutError, asyncio.TimeoutError):
                _kill_group(proc)
                # The ssh that *owned* the master has been killed, but the master is a
                # forked background process that outlives it and keeps the socket open.
                # Reusing that channel is how a timeout turns into a wrong success.
                await self._close_master(host)
                return ExecResult(host.name, 124, "", f"timed out after {timeout:g}s", True)
            except BaseException:
                _kill_group(proc)
                await self._close_master(host)
                raise
            return ExecResult(
                host=host.name,
                exit_code=proc.returncode or 0,
                stdout=out.decode(errors="replace"),
                stderr=err.decode(errors="replace"),
            )

    async def ping(self, host: Host, *, timeout: float = 20.0) -> ExecResult:
        return await self.exec_on(host, self.probe(), timeout=timeout)

    # --- teardown ---------------------------------------------------------------

    async def _close_master(self, host: Host) -> None:
        """Ask the ControlMaster for ``host`` to exit, then drop its socket.

        Best effort: it runs on the timeout/cancel paths, where a cleanup failure must
        not replace the real result. ``ControlMaster=no`` is essential — without it this
        invocation would try to *become* the master for the same ControlPath.
        """
        sock = self.control_path(host)
        if not sock.exists():
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o",
                f"ControlPath={sock}",
                "-o",
                "ControlMaster=no",
                "-o",
                "BatchMode=yes",
                "-O",
                "exit",
                self.target(host),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._env(None),
                start_new_session=True,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
        sock.unlink(missing_ok=True)

    async def close(self) -> None:
        """Tear down every multiplexed connection and the socket directory."""
        directory = self._sockets
        if directory is None:
            return
        for sock in list(directory.glob("*.sock")):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o",
                    f"ControlPath={sock}",
                    "-o",
                    "ControlMaster=no",
                    "-O",
                    "exit",
                    "localhost",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=self._env(None),
                )
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            sock.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        self._sockets = None


def require_ssh() -> None:
    """Fail early, with a fixable message, when there is no ssh client at all."""
    if shutil.which("ssh") is None:
        raise ConfigError("`ssh` is not installed — install an OpenSSH client")
