"""A terminal that is always handed back usable, however the process goes down.

A full-screen app owns the terminal while it runs: raw mode (no echo, no line
buffering, Ctrl-C is only a keystroke), the alternate screen, a hidden cursor, mouse
reporting, the kitty keyboard protocol. Textual restores all of that on its way out —
*when it gets to run on its way out*. The cases that matter for a monitor left open all
day are the ones where it does not:

* **A laptop is closed, the Wi-Fi drops, the terminal window is closed.** The kernel
  sends ``SIGHUP``. Python's default for it is terminate, and Textual installs no
  handler, so the process dies with the terminal still in raw mode and on the
  alternate screen. The shell is then typing into a terminal nobody is reading: keys
  come back as escape-code debris ("乱码"), and ``q`` or Ctrl-C do nothing — there is
  no line discipline left to turn Ctrl-C into a signal. Only a new terminal gets it
  back.

* **A terminal emulator loses its session** (the tmux server exits, a tmux *client* is
  killed, the window's session is reset). The app is not signalled. Its stdin is still
  the old pty, which no longer has a master: the next read returns EOF, the next write
  fails with ``EIO``. Textual's input thread ends quietly on that EOF and its writer
  thread dies on the failed write; the app runs on with no way in and no way out —
  ``q`` is read by nobody, and quitting it is what needs a new terminal.

Both end the same way on screen: a live-looking process that owns a terminal nobody can
read, next to a shell that needs a restart to become usable again.

So this module does three things, none of which depend on Textual's own teardown:

:class:`TerminalGuard`
    Handles ``SIGHUP`` and ``SIGTERM`` by handing the terminal back and *then* dying
    the way the signal asked, and restores it again at process exit — a fresh,
    non-blocking ``dup`` of the tty, so restoring never waits on the app's own (possibly
    dead) writer.

:class:`Supervisor`
    A one-per-second status check, outside the event loop. When the app's writer thread
    has died, or a probe write to the terminal raises: the terminal is gone, so the app
    is told to exit (closing its SSH connections) — and the process is ended a few
    seconds later regardless, because one stuck thread must not keep the process alive
    with no screen to show for it.

:func:`harden_input_selector`
    Replaces the selector Textual's input thread waits on, so a raised ``EIO``/``EBADF``
    from a vanished tty is reported once rather than every iteration, instead of
    turning into a panic on screen.
"""

from __future__ import annotations

import atexit
import os
import select
import selectors
import signal
import sys
import termios
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.app import App

#: Undo every mode a full-screen app sets, and mean it. Ordered so the cursor is made
#: visible *before* the alternate screen (and its cursor state) goes away:
#:
#: * ``<u`` and ``SGR 0m`` — drop the kitty keyboard protocol and any colour left on.
#: * ``?1000``/``?1002``/``?1003``/``?1006`` off — mouse reporting.
#: * ``?1004`` off — focus reporting.
#: * ``?2004`` off — bracketed paste.
#: * ``?7h``, ``ESC[4l``, ``ESC(B`` — line wrap, insert mode, US character set.
#: * ``?25h`` — cursor visible on the alternate screen, so it survives the switch.
#: * ``?1049l`` — back to the main screen.
#: * ``?2026l`` — end any synchronised update that was cut off mid-frame.
RESTORE_SEQUENCE = (
    "\x1b[<u"
    "\x1b[0m"
    "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l"
    "\x1b[?1004l"
    "\x1b[?2004l"
    "\x1b[?7h\x1b[4l\x1b(B"
    "\x1b[?25h"
    "\x1b[?1049l"
    "\x1b[?2026l"
)

#: A pty whose reader is gone rejects writes only once its buffer fills, so the restore
#: must stay small; the timeout is for a stalled tty, not for a healthy one.
_WRITE_WAIT = 0.2

#: The terminal settings a shell needs back: what a TUI turns off and a prompt cannot
#: live without (``\n`` would no longer mean "new line and carriage return", Ctrl-C
#: would no longer be a signal, keys would not echo).
_SHELL_IFLAG = termios.ICRNL | termios.IXON
_SHELL_LFLAG = termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN
_SHELL_OFLAG = termios.OPOST

_guard: TerminalGuard | None = None
_guard_lock = threading.Lock()


class TerminalGuard:
    """Owns the terminal's exit path: signals, process exit, and the restore itself."""

    def __init__(self, *, watching: bool = True):
        #: Where the terminal is. Textual renders to ``sys.__stderr__``; the others are
        #: fallbacks for an app invoked with a redirected stream.
        self._streams = [s for s in (sys.__stderr__, sys.__stdout__, sys.__stdin__) if s]
        self._previous: dict[int, Any] = {}
        #: Set when the first restore attempt finishes, successfully or not.
        self.restored_once = False
        if watching:
            self._watch()

    # --- signals ----------------------------------------------------------------

    def _watch(self) -> None:
        for signum in (signal.SIGHUP, signal.SIGTERM):
            try:
                self._previous[signum] = signal.signal(signum, self._fatal)
            except (OSError, ValueError):  # not the main thread: leave it be
                pass

    def _fatal(self, signum: int, frame: Any) -> None:
        """``SIGHUP``/``SIGTERM``: give the terminal back, then die as the signal asked.

        Deliberately not chaining to the previous handler: Python's default for these is
        terminate, and terminate is what the shell around us expects (``$? == 129`` for
        ``SIGHUP``). Restoring *first* is the point — a raw terminal outlives the
        process that left it that way.
        """
        self.restore()
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except (OSError, ValueError):  # pragma: no cover - only if we are not the main thread
            os._exit(128 + signum)  # noqa: SLF001

    # --- the terminal itself ----------------------------------------------------

    def _fd(self) -> int | None:
        for stream in self._streams:
            try:
                fd = stream.fileno()
            except (AttributeError, OSError, ValueError):
                continue
            try:
                if os.isatty(fd):
                    return fd
            except OSError:
                continue
        return None

    def restore(self) -> bool:
        """Put the terminal back. Idempotent; safe to call from a signal handler.

        Returns ``False`` when nothing could be written to a terminal — the way this
        learns that the terminal is gone for good. Works on a fresh non-blocking
        ``dup`` rather than through the app: the app's writer is exactly what may be
        stuck, and a restore that waits on it is a restore that never happens.
        """
        data = RESTORE_SEQUENCE.encode()
        reached = False
        for stream in self._streams:
            fd = None
            try:
                fd = os.dup(stream.fileno())
                os.set_blocking(fd, False)
            except (AttributeError, OSError, ValueError):
                if fd is not None:
                    os.close(fd)
                continue
            try:
                _r, writable, _x = select.select([], [fd], [], _WRITE_WAIT)
                if not writable:
                    continue
                reached = os.write(fd, data[:4096]) > 0
                if reached:
                    break
            except OSError:
                continue
            finally:
                os.close(fd)
        self.restored_once = True
        if reached:
            # The bytes are on their way, so the modes the app set can go: a shell on
            # this terminal afterwards would otherwise inherit "no echo, no Ctrl-C".
            self._shell_modes_on()
        return reached

    def _shell_modes_on(self) -> None:
        fd = self._fd()
        if fd is None:
            return
        try:
            attrs = termios.tcgetattr(fd)
        except (termios.error, OSError):
            return
        attrs[0] |= _SHELL_IFLAG
        attrs[1] |= _SHELL_OFLAG
        attrs[3] |= _SHELL_LFLAG
        try:
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except (termios.error, OSError):
            pass


def guard_terminal(*, watching: bool = True) -> TerminalGuard:
    """Install (once) this process's terminal guard."""
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = TerminalGuard(watching=watching)
            atexit.register(_guard.restore)
        return _guard


def current_guard() -> TerminalGuard | None:
    return _guard


def restore() -> bool:
    """Restore now — for code that runs after the app has exited."""
    return _guard is not None and _guard.restore()


# --- supervising a running app ---------------------------------------------------


class Supervisor:
    """Watches a running app's terminal and ends the process when the app cannot.

    Started by :meth:`JmsTopApp.run`; one daemon thread, two list reads and one
    ``isalive`` a second. It sees the app's own machinery (its message queue, its output
    pipe) from outside the event loop — which matters, because the event loop being
    wedged is one of the ways this goes wrong.
    """

    #: After asking the app to quit, how long it gets to do so by itself.
    GRACE_S = 4.0
    #: Seconds between checks.
    PERIOD_S = 1.0

    def __init__(self, app: App):
        self.app = app
        #: Why the app was asked to stop, for the final message.
        self.note = ""
        #: ``None`` while healthy; the deadline for "exit it by force" once not.
        self.deadline: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="jms-top-supervisor", daemon=True)

    def start(self) -> Supervisor:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def should_quit(self) -> str | None:
        """Why the app ought to stop now, or ``None``. Always ``None`` once stopped.

        Two symptoms point at the terminal having been replaced under us: the output
        pipe has failed (a thread cannot be killed, so a live process with a dead writer
        is exactly that), or writes to the tty raise — which is the same failure seen at
        the other end. A resize to zero is *not* used: on macOS a terminal reports zero
        while the app is suspended, and quitting over that would fight the shell.
        """
        if self._stop.is_set():
            return None
        driver = getattr(self.app, "_driver", None)
        if driver is None or _is_test_driver(driver):
            return None
        writer = getattr(driver, "_writer_thread", None)
        if writer is not None and not writer.is_alive():
            return "the terminal went away (its output could not be written to)"
        if not reachable():
            # The writer is only *eventually* dead: it dies on the next write, and a
            # paused monitor may not write again for a while. Stdin going away (a
            # killed tmux client leaves /dev/null behind, say) never kills it at all.
            #
            # Only *writes* are tested, never whether stdin is still readable: an
            # embedded tty (a pty handed to us by a terminal emulator) reports a closed
            # master as readable-but-empty rather than as EIO, and a working stdin is
            # allowed to do either at any moment.
            return "the terminal went away (it can no longer be written to)"
        return None

    def _loop(self) -> None:
        while not self._stop.wait(self.PERIOD_S):
            reason = self.should_quit()
            if reason is None:
                if self.deadline is not None and time.monotonic() > self.deadline:
                    # Told to quit and it did not: something inside the app is stuck
                    # (a thread on a dead write, a join on that thread). Waiting longer
                    # keeps the process — and the SSH ControlMasters it holds — alive
                    # with nothing to show for it. The terminal is already gone, so
                    # there is nothing left to restore.
                    os._exit(1)  # noqa: SLF001
                continue
            if self.deadline is None:
                self.note = reason
                try:
                    self.app.call_from_thread(self.app.exit, 1)
                except Exception:  # noqa: BLE001 - the app is already going down
                    pass
                self.deadline = time.monotonic() + self.GRACE_S


def reachable() -> bool:
    """Can the terminal we render to still be *written* to?

    The direct test, the same failure Textual's writer meets: one probe write that
    fails with ``EIO`` when the pty has lost its reader. It is an empty OSC 8 hyperlink
    — parsed by the terminal, draws no glyph and moves no cursor — so a healthy screen
    cannot tell it happened.

    Note what is *not* done here: no ``dup``/``select`` first. A ``dup`` of a blocking
    tty is itself blocking, and a probe that blocks on a tty rescued from a suspend is
    how one loses the app to a hang; the write is 16 bytes, so on any pty that still has
    a reader it lands in the buffer immediately.
    """
    for stream in (sys.__stderr__, sys.__stdout__):
        try:
            fd = stream.fileno()  # type: ignore[union-attr]
            if not os.isatty(fd):
                continue
            os.write(fd, _PROBE)
            return True
        except OSError:
            return False
        except (AttributeError, ValueError):
            continue
    return False


#: An empty OSC 8 hyperlink: parsed by the terminal, draws nothing.
_PROBE = b"\x1b]8;;\x1b\\\x1b]8;;\x1b\\"


def _is_test_driver(driver: Any) -> bool:
    """``run_test``'s headless driver has no writer thread and no real terminal."""
    return type(driver).__name__ in {"HeadlessDriver", "WebDriver", "WindowsDriver"}


def say(line: str) -> None:
    """Write one line to the terminal, best effort, accepting a vanished one.

    After the app has exited a broken pipe or ``EIO`` here is normal (the terminal may
    be the reason we are talking at all), and must not turn into a traceback.
    """
    for stream in (sys.__stderr__, sys.__stdout__):
        if stream is None:
            continue
        try:
            fd = stream.fileno()
        except (AttributeError, OSError, ValueError):
            continue
        try:
            if os.isatty(fd):
                return  # the terminal is alive: the write below cannot block on a dead pty
        except OSError:
            continue
        try:
            os.write(fd, (line.rstrip("\n") + "\n").encode())
        except OSError:
            pass
        return


# --- the input thread's selector -------------------------------------------------


class ResilientSelector:
    """A ``selectors`` selector for stdin that copes with a terminal that is gone.

    Same interface as what it replaces (``register``, ``select``, ``unregister``,
    ``close``, ``get_map``), because Textual's input thread calls those on the object
    it holds.

    * Normally it delegates untouched.
    * An exception from the underlying selector (``EIO``/``EBADF`` on a vanished tty) is
      reported once, as EOF, and then swallowed: repeated EOFs would flood the app's
      queue instead of ending it.

    Deliberately *not* a silent return while the output is dead: Textual's input thread
    also uses this select purely as a 100 ms sleep between parser ticks (it runs it even
    with nothing registered), and returning early turns that loop into a busy spin.
    The supervisor ends the process instead, within seconds.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self._eof_reported = False

    def __getattr__(self, name: str) -> Any:
        # Forwarded only for the parts of the selector API not spelled out below
        # (``modify`` and friends). A missing one still raises like a normal attribute;
        # only the *inner* selector knows the difference.
        return getattr(self._inner, name)

    def register(self, fileobj: Any, events: int, data: Any = None) -> Any:
        return self._inner.register(fileobj, events, data)

    def unregister(self, fileobj: Any) -> Any:
        return self._inner.unregister(fileobj)

    def close(self) -> None:
        self._inner.close()

    def get_map(self) -> Any:
        return self._inner.get_map()

    def select(self, timeout: float | None = None) -> list[Any]:
        try:
            return self._inner.select(timeout)
        except Exception:  # noqa: BLE001 - EIO, EBADF, a broken selector: the tty is gone
            if self._eof_reported:
                _sleep(timeout)
                return []
            self._eof_reported = True
            return [(key, selectors.EVENT_READ) for key in self._inner.get_map().values()]


def _sleep(timeout: float | None) -> None:
    time.sleep(min(timeout if timeout is not None else 0.1, 0.1))


def harden_input_selector(app: App) -> bool:
    """Give Textual's input thread a :class:`ResilientSelector`. ``False`` if it cannot."""
    try:
        reader = getattr(app._driver, "_input_reader", None)
        inner = getattr(reader, "_selector", None)
        if reader is None or inner is None:
            return False
        if isinstance(inner, ResilientSelector):
            return True
        reader._selector = ResilientSelector(inner)  # noqa: SLF001
        return True
    except Exception:  # noqa: BLE001 - a Textual upgrade is not a crash
        return False
