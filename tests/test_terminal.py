"""The terminal always comes back usable (see :mod:`jms_top.terminal`).

Every test here drives a real child process on a real pty, because the failure being
fixed is a kernel/tty one: signals on a process group, ``EIO`` on a dead master, modes
that outlive the process that set them. Textual's headless driver cannot show any of
that. The child runs the demo fleet, so nothing touches the network.
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no pty / termios on Windows")

SRC = Path(__file__).resolve().parents[1] / "src"

#: The child: the real app on the real driver, rendering the demo fleet.
_CHILD = """
import sys
sys.path.insert(0, {src!r})
from jms_top.demo import demo_monitor
from jms_top.tui import JmsTopApp
JmsTopApp(demo_monitor(interval=5), boot=True).run()
"""


class Pty:
    """A child process attached to a pty, with helpers to watch it without blocking."""

    def __init__(self, script: str, *, size: tuple[int, int] = (40, 120)):
        rows, cols = size
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.pid = os.fork()
        if self.pid == 0:  # pragma: no cover - this is the child half
            try:
                os.close(self.master)
                os.setsid()
                fcntl.ioctl(self.slave, termios.TIOCSCTTY, 0)
                os.dup2(self.slave, 0)
                os.dup2(self.slave, 1)
                os.dup2(self.slave, 2)
                os.close(self.slave)
                os.execv(sys.executable, [sys.executable, "-u", "-c", script])
            finally:
                os._exit(1)
        os.close(self.slave)
        os.set_blocking(self.master, False)
        self.output = b""

    # --- io --------------------------------------------------------------------

    def pump(self, seconds: float) -> bytes:
        """Drain the child's output for ``seconds`` and return what arrived."""
        end = time.monotonic() + seconds
        seen = b""
        while time.monotonic() < end:
            ready, _, _ = select.select([self.master], [], [], 0.05)
            if not ready:
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:  # the child closed the pty: normal on exit
                break
            if not data:
                break
            seen += data
        self.output += seen
        return seen

    def send(self, data: bytes) -> None:
        os.write(self.master, data)

    def wait_exit(self, seconds: float) -> int | None:
        """Poll for the child's exit; ``None`` means still running after ``seconds``."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            wpid, status = os.waitpid(self.pid, os.WNOHANG)
            if wpid == self.pid:
                self._reaped = True
                return status
            self.pump(0.1)
        return None

    def tty_lflag(self) -> int:
        return termios.tcgetattr(self.master)[3]

    def close(self) -> None:
        if not getattr(self, "_reaped", False):
            try:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
        try:
            os.close(self.master)
        except OSError:
            pass


def _start() -> Pty:
    child = Pty(textwrap.dedent(_CHILD).format(src=str(SRC)))
    child.pump(2.5)  # let the app take the terminal over
    return child


def _raw(lflag: int) -> bool:
    """The modes Textual sets: no echo, no line discipline, Ctrl-C is just a byte."""
    return not (lflag & termios.ECHO) and not (lflag & termios.ICANON)


class TestFatalSignalsHandTheTerminalBack:
    """A closed laptop sends SIGHUP; so does closing the terminal window."""

    @pytest.mark.parametrize("signame", ["SIGHUP", "SIGTERM"])
    def test_terminal_is_restored_and_the_signal_is_honoured(self, signame):
        signum = getattr(signal, signame)
        child = _start()
        try:
            assert _raw(child.tty_lflag()), "the app should own the terminal in raw mode"

            os.kill(child.pid, signum)
            status = child.wait_exit(5.0)

            assert status is not None, "the process must die on the signal"
            assert os.WIFSIGNALED(status), "exit status must report the signal"
            assert os.WTERMSIG(status) == signum
            # The screen: alternate screen left, cursor shown.
            assert b"\x1b[?1049l" in child.output
            assert b"\x1b[?25h" in child.output
            # The tty: a shell inheriting it can echo and read lines again.
            lflag = child.tty_lflag()
            assert lflag & termios.ECHO
            assert lflag & termios.ICANON
            assert lflag & termios.ISIG
        finally:
            child.close()


class TestTerminalTakenAway:
    """The window/tmux session goes away without signalling the app."""

    def test_the_process_ends_when_its_pty_is_closed(self):
        child = _start()
        try:
            os.close(child.master)
            child._reaped = False
            waiting = True
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                wpid, _status = os.waitpid(child.pid, os.WNOHANG)
                if wpid == child.pid:
                    waiting = False
                    child._reaped = True
                    break
                time.sleep(0.1)
            assert not waiting, "a monitor with no terminal must not stay behind"
        finally:
            child.close()


class TestNormalQuit:
    def test_q_still_quits_and_prints_nothing_extra(self):
        child = _start()
        try:
            child.send(b"q")
            status = child.wait_exit(5.0)
            assert status is not None and os.WIFEXITED(status)
            assert os.WEXITSTATUS(status) == 0
            assert b"terminal went away" not in child.output
        finally:
            child.close()
