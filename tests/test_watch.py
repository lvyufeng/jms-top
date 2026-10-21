"""Hot reload: an edit to the UI lands while the app is running.

The property under test is not "a function returns the new string" — it is that the
*running app*, with its monitor and its data intact, paints the new code without a
restart. These tests drive the real :class:`JmsTopApp` under Textual's headless driver
and edit a real file on disk.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from textual.widgets import DataTable

from jms_top import panels
from jms_top.demo import demo_monitor
from jms_top.tui import JmsTopApp
from jms_top.watch import Change, SourceWatcher, reload_modules


def run(coro):  # noqa: ANN001, ANN201
    return asyncio.run(coro)


@pytest.fixture()
def source(tmp_path):
    """A throwaway module source the tests can rewrite."""
    return tmp_path / "subject.py"


# --- the watcher ----------------------------------------------------------------


def test_watcher_reports_a_rewritten_file(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    watcher = SourceWatcher(extra=[source])
    assert watcher.check().any is False  # the first check is the baseline

    # Same length on purpose, and the mtime pinned to a value the coarse float form
    # cannot distinguish — an ordinary editor save inside one second.
    source.write_text("x = 2\n")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    change = watcher.check()
    assert change.any is True
    # One save must not fire twice: the baseline moves with the report.
    assert watcher.check().any is False


def test_watcher_classifies_reloadable_and_restart_only():
    """A parser change cannot be swapped safely; the app should say so, not guess."""
    watcher = SourceWatcher()
    watcher._stamps = {name: -1.0 for name in watcher._stamps}  # pretend everything moved
    change = watcher.check()
    assert "jms_top.panels" in change.reloadable
    assert "jms_top.monitor" in change.needs_restart
    assert change.stylesheet is True


def test_reload_module_picks_up_a_rewrite(monkeypatch, tmp_path):
    """The reload must not be fooled by the bytecode cache.

    `importlib.reload` re-reads source only when the cached `.pyc` looks stale, and
    that check uses mtime **and size** truncated to seconds — an edit that keeps the
    size, made within the same second, silently reloads the old code. That was the bug
    that made hot reload look like it did nothing.
    """
    module_file = tmp_path / "victim.py"
    module_file.write_text("VALUE = 'before'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = __import__("victim")
    assert module.VALUE == "before"

    # Same length on purpose: this is the case a naive reload misses.
    module_file.write_text("VALUE = 'after!'\n")
    reloaded, failed = reload_modules({"victim"})
    assert failed == []
    assert __import__("victim").VALUE == "after!"


def test_a_broken_edit_keeps_the_running_code(monkeypatch, tmp_path):
    """Half-typed code is the normal case; it must not take the app down."""
    module_file = tmp_path / "fragile.py"
    module_file.write_text("VALUE = 'ok'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    __import__("fragile")

    module_file.write_text("VALUE = 'ok\n")  # unterminated string
    _, failed = reload_modules({"fragile"})
    assert failed and "fragile" in failed[0]
    # The half-typed file left sys.modules without the module; importing it again must
    # still fail (proving it was never silently accepted), and the rest of the app keeps
    # running on the code it already had.
    with pytest.raises(SyntaxError):
        __import__("fragile")


# --- the running app ------------------------------------------------------------


def test_editing_the_panels_updates_the_live_screen():
    """The whole point: a running app shows the new code without losing its data."""
    path = Path(panels.__file__)
    original = path.read_text()

    async def go():
        monitor = demo_monitor(interval=60, full_interval=120)
        app = JmsTopApp(monitor, boot=False, watch=False)
        try:
            async with app.run_test(size=(140, 34)) as pilot:
                await monitor.poll()
                app._render()
                await pilot.pause()
                before = str(app.query_one("#titlebar").content)
                assert "HOTRELOADED" not in before

                path.write_text(original.replace("cards free ", "cards HOTRELOADED ", 1))
                app.apply_change(Change({"jms_top.panels"}, set(), False))
                # The rebuild runs in a worker, and a queued frame may land before or
                # after it — wait for the *rebuild* to finish rather than sampling the
                # screen a fixed number of times, which is what made this flaky.
                for _ in range(200):
                    await asyncio.sleep(0.02)
                    await pilot.pause()
                    if not app._rebuilding and "HOTRELOADED" in str(
                        app.query_one("#titlebar").content
                    ):
                        break

                after = str(app.query_one("#titlebar").content)
                assert "HOTRELOADED" in after, "the screen kept the old code"
                # The data survived the swap: same host count, same monitor object.
                assert app.query_one("#table", DataTable).row_count == len(monitor.view.hosts)
                assert monitor.view.total_cards > 0
        finally:
            path.write_text(original)
            reload_modules({"jms_top.panels"})

    run(go())


def test_a_restart_only_change_is_announced_not_applied():
    """monitor/probe changes are reported in the footer rather than half-applied."""

    async def go():
        monitor = demo_monitor(interval=60, full_interval=120)
        app = JmsTopApp(monitor, boot=False, watch=False)
        async with app.run_test(size=(140, 34)) as pilot:
            await monitor.poll()
            app._render()
            await pilot.pause()
            app.apply_change(Change(set(), {"jms_top.monitor"}, False))
            await pilot.pause()
            assert app._restart_pending is True
            app._render()  # what the repaint timer does a fraction of a second later
            footer = str(app.query_one("#footer-bar").content)
            assert "restart" in footer

    run(go())


def test_reload_while_selected_keeps_the_selection():
    """A reload is a repaint, not a reset: the host you were looking at stays selected."""

    async def go():
        monitor = demo_monitor(interval=60, full_interval=120)
        app = JmsTopApp(monitor, boot=False, watch=False)
        async with app.run_test(size=(140, 34)) as pilot:
            await monitor.poll()
            app._render()
            await pilot.pause()
            wanted = app.visible_hosts()[2].name
            app._selected = wanted
            app._render()
            await pilot.pause()

            app.apply_change(Change({"jms_top.panels"}, set(), False))
            # Wait for the rebuild worker rather than for the table to be non-empty:
            # the table still holds the pre-reload rows while the swap is in flight, so
            # "row_count > 0" is true immediately and proves nothing.
            for _ in range(200):
                await asyncio.sleep(0.02)
                await pilot.pause()
                if not app._rebuilding:
                    break
            await pilot.pause()
            assert app._selected == wanted

    run(go())


def test_watched_app_arms_the_watcher_only_when_asked():
    """The watcher costs a stat per source per second — opt-in, and off by default."""

    async def armed(watch: bool) -> bool:
        monitor = demo_monitor(interval=60, full_interval=120)
        app = JmsTopApp(monitor, boot=False, watch=watch)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            return app._watcher is not None

    assert watch_armed(True) is True
    assert watch_armed(False) is False


def watch_armed(flag: bool) -> bool:
    async def go() -> bool:
        monitor = demo_monitor(interval=60, full_interval=120)
        app = JmsTopApp(monitor, boot=False, watch=flag)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            return app._watcher is not None

    return run(go())


def test_address_column_shows_the_ip_and_degrades_to_dash():
    """The address is what you act on; a missing one is a dash, never blank."""

    async def go():
        monitor = demo_monitor(interval=60, full_interval=120)
        app = JmsTopApp(monitor, boot=False, watch=False)
        async with app.run_test(size=(160, 34)) as pilot:
            await monitor.poll()
            app._render()
            await pilot.pause()
            table = app.query_one("#table", DataTable)
            assert "ADDRESS" in [str(c.label).split()[0] for c in table.columns.values()]
            hosts = app.visible_hosts()
            address_cells = [str(table.get_row_at(row)[1]) for row in range(len(hosts))]
            for host, cell in zip(hosts, address_cells, strict=True):
                if host.address:
                    assert cell == host.address
                else:
                    assert cell == "—"

    run(go())
