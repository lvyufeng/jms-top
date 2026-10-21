"""Live editing: change the code, see it, without losing the connection.

The loop this serves is the one a TUI is worst at: you are watching the screen, you
notice something wrong, and the fix is a dozen lines away. Quitting destroys the state
you were looking at — the multiplexed SSH connections, the last full round, the cursor.
So the app watches its own sources and, when one changes, rebuilds **the interface
only**: the monitor keeps its connections and its data, and the new widgets paint from
the last snapshot.

What is reloaded, and what is not:

* :mod:`jms_top.panels`, :mod:`jms_top.render` and :mod:`jms_top.app` (the stylesheet)
  are reloaded and the screen is recomposed. These are the files a change is usually
  about — layout, wording, colours — and they hold no state.
* :mod:`jms_top.tui` is reloaded too, but the widgets are rebuilt explicitly rather
  than by swapping the class, so the running monitor and its `asyncio` tasks survive.
* :mod:`jms_top.monitor`, :mod:`jms_top.probe` and :mod:`jms_top.families` are **not**
  hot-swapped. A monitor swap would cancel in-flight SSH commands and drop every
  reading on screen; those changes still want a restart, and the footer says so when it
  sees one.

The watcher is a polling ``stat`` loop, not a filesystem-events library: one small
``stat`` per source file per second is cheaper than a dependency, and it behaves the
same on every platform (macOS FSEvents and Linux inotify disagree about editor
save-via-rename, which is exactly how most editors save).
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

#: Modules whose change can be applied without restarting the app.
RELOADABLE = ("jms_top.panels", "jms_top.render", "jms_top.tui")
_RELOADABLE = {"panels", "render", "tui"}

#: Modules that need a restart: their objects are already wired into running tasks.
NEEDS_RESTART = ("jms_top.monitor", "jms_top.probe", "jms_top.families", "jms_top.scan")


@dataclass
class Change:
    """What changed since the last check."""

    reloadable: set[str]
    needs_restart: set[str]
    stylesheet: bool
    #: Watched files outside the package (tests, extra sources). They are reported but
    #: never imported, so they cannot be classified as either of the above.
    other: set[str] = field(default_factory=set)

    @property
    def any(self) -> bool:
        return bool(self.reloadable or self.needs_restart or self.stylesheet or self.other)


def _sources() -> dict[str, Path]:
    """Source file per module name, plus the stylesheet, for the *installed* package."""
    import jms_top

    root = Path(jms_top.__file__).parent
    files = {path.stem: path for path in root.glob("*.py")}
    files["app.tcss"] = root / "app.tcss"
    return files


class SourceWatcher:
    """Polls the package's sources and reports what changed.

    The first :meth:`check` records the baseline, so starting the watcher never fires a
    reload for files that were already on disk.
    """

    def __init__(self, *, extra: list[Path] | None = None):
        self._files = {name: path for name, path in _sources().items() if path.exists()}
        for path in extra or []:
            self._files[f"extra:{path.name}"] = path
        self._stamps: dict[str, tuple[int, int]] = {}
        self._prime()

    def _prime(self) -> None:
        self._stamps = {name: _stamp(path) for name, path in self._files.items()}

    def check(self) -> Change:
        """Compare against the last check; update the baseline when anything moved."""
        changed = {
            name for name, path in self._files.items() if _stamp(path) != self._stamps.get(name)
        }
        if not changed:
            return Change(set(), set(), False)
        self._prime()  # re-baseline so one save does not fire twice
        reloadable = {f"jms_top.{n}" for n in changed if n in _RELOADABLE}
        restart = {f"jms_top.{n}" for n in changed if f"jms_top.{n}" in NEEDS_RESTART}
        # Anything watched but not classified (an extra file, or a package module that
        # is neither reloadable nor restart-only) still counts as a change: silently
        # ignoring it would leave "nothing happened" as the answer after an edit.
        other = {
            n
            for n in changed
            if n not in _RELOADABLE and n != "app.tcss" and f"jms_top.{n}" not in NEEDS_RESTART
        }
        return Change(reloadable, restart, "app.tcss" in changed, other)

    def paths(self) -> dict[str, Path]:
        return dict(self._files)


def _stamp(path: Path) -> tuple[int, int]:
    """A change stamp of ``(mtime_ns, size)``.

    Nanoseconds, not ``st_mtime``: the float is truncated on a coarse filesystem, so two
    saves inside one second compare equal and the edit is never seen. Size is part of
    the stamp because a same-size rewrite with a preserved mtime (``cp -p``, some
    editors' atomic saves) would otherwise be invisible too.
    """
    try:
        stat = path.stat()
    except OSError:  # a file removed under us while the editor saves
        return (-1, -1)
    return (stat.st_mtime_ns, stat.st_size)


def reload_modules(names: set[str]) -> tuple[list[str], list[str]]:
    """Re-execute each module's source, bypassing the bytecode cache.

    ``importlib.reload`` is not enough on its own: it re-reads the source only when the
    cached ``.pyc`` looks stale, and staleness is judged by mtime **and size** — both
    truncated to whole seconds. An edit that keeps the file size and lands in the same
    second therefore reloads the *old* bytecode, which looks exactly like "hot reload
    does nothing". Every source is removed from ``sys.modules`` and re-imported from the
    file instead, which cannot serve a stale cache entry.

    A syntax error while you are half-way through typing is the *normal* case here, so a
    failed reload is reported and the running code is kept, not raised.
    """
    import sys

    reloaded: list[str] = []
    failed: list[str] = []
    swapped: list[tuple[object, object]] = []
    for name in sorted(names):
        old = sys.modules.get(name)
        if old is None:
            failed.append(f"{name} (not imported)")
            continue
        try:
            path = getattr(old, "__file__", None)
            if path:
                importlib.invalidate_caches()
                Path(importlib.util.cache_from_source(path)).unlink(missing_ok=True)
            sys.modules.pop(name, None)
            new = importlib.import_module(name)
            swapped.append((old, new))
            reloaded.append(name)
        except Exception as exc:  # noqa: BLE001 - a half-typed file is not fatal
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    if swapped:
        rebind_references(swapped)
    return reloaded, failed


def rebind_references(swapped: list[tuple[object, object]]) -> None:
    """Point every already-imported module's globals at the new module objects.

    Re-importing puts a *new* module object in ``sys.modules``, but a module that did
    ``from . import panels`` still holds the old one — so the reload would run and the
    screen would keep calling the previous functions. Walk every loaded ``jms_top``
    module and replace globals that point at a superseded module.

    Only *module objects* are rebound. Swapping functions or classes wholesale is what
    produces half-updated state; whole modules are safe to exchange because nothing in
    this package holds state on them between calls.
    """
    import sys

    stale = {id(old): new for old, new in swapped}
    for module in list(sys.modules.values()):
        if module is None or not str(getattr(module, "__name__", "")).startswith("jms_top"):
            continue
        for attr, value in list(vars(module).items()):
            replacement = stale.get(id(value))
            if replacement is not None:
                setattr(module, attr, replacement)


def describe(change: Change) -> str:
    """A one-line summary for the log pane."""
    bits = []
    if change.reloadable:
        bits.append("reloaded " + ", ".join(sorted(n.split(".")[-1] for n in change.reloadable)))
    if change.stylesheet:
        bits.append("stylesheet")
    if change.needs_restart:
        bits.append(
            "needs restart: " + ", ".join(sorted(n.split(".")[-1] for n in change.needs_restart))
        )
    return " · ".join(bits) or "no change"
