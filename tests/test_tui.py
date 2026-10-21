"""UI smoke tests: the app renders from a monitor whose runner answers from fixtures.

These drive the real :class:`JmsTopApp` in Textual's headless test driver, so a
layout that raises on an unexpected host (a dead asset, a host whose vendor tool
failed, a residual fleet) is caught here rather than in front of a user.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import DataTable

from jms_top.demo import demo_monitor
from jms_top.tui import FILTERS, JmsTopApp, verdict_matches


def run(coro):  # noqa: ANN001, ANN201
    return asyncio.run(coro)


def _app():
    monitor = demo_monitor(interval=60, full_interval=120)
    return monitor, JmsTopApp(monitor, boot=False)


async def _settle(pilot, monitor) -> None:  # noqa: ANN001
    await monitor.poll()
    app = pilot.app
    app._rendered_rows = []
    app._render()
    await pilot.pause()


def test_demo_fleet_is_rendered_with_every_tab_column():
    async def go():
        monitor, app = _app()
        async with app.run_test(size=(160, 44)) as pilot:
            await _settle(pilot, monitor)
            table = app.query_one("#table", DataTable)
            assert table.row_count == len(monitor.view.hosts)
            hosts = app.visible_hosts()
            # Free hosts come first: that is the whole point of the ordering.
            assert hosts[0].verdict == "free"
            assert hosts[-1].verdict in {"down", "unknown", "cpu"}

    run(go())


def test_filter_cycle_keeps_only_matching_hosts():
    async def go():
        monitor, app = _app()
        async with app.run_test(size=(160, 44)) as pilot:
            await _settle(pilot, monitor)
            app.filter = "free"
            app._rendered_rows = []
            app._render()
            await pilot.pause()
            table = app.query_one("#table", DataTable)
            assert table.row_count == sum(1 for h in monitor.view.hosts if h.verdict == "free")
            assert all(h.verdict == "free" for h in app.visible_hosts())

    run(go())


def test_search_matches_users_and_process_names():
    async def monitor_names():
        monitor, app = _app()
        async with app.run_test(size=(160, 44)) as pilot:
            await _settle(pilot, monitor)
            app.needle = "sglang"
            app._rendered_rows = []
            app._render_table()
            assert app.visible_hosts(), "sglang runs on at least one fixture host"
            assert all(
                "sglang" in " ".join(p["cmd"] for p in h.node.top).lower()
                or "sglang" in " ".join(v.name for _c, v in h.procs).lower()
                for h in app.visible_hosts()
            )

    run(monitor_names())


def test_process_view_opens_and_closes():
    async def go():
        monitor, app = _app()
        async with app.run_test(size=(160, 44)) as pilot:
            await _settle(pilot, monitor)
            busy = next(h for h in app.visible_hosts() if h.procs)
            app._selected = busy.name
            app._rendered_rows = []
            app.action_inspect()
            await pilot.pause()
            assert app._detail_mode is True
            assert busy.name in str(app.query_one("#detail").content)
            app.action_close_overlay()
            await pilot.pause()
            assert app._detail_mode is False

    run(go())


def test_every_filter_renders_without_error():
    async def go():
        monitor, app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot, monitor)
            for wanted in FILTERS:
                app.filter = wanted
                app._rendered_rows = []
                app._render()
                await pilot.pause()
                assert all(verdict_matches(h.verdict, wanted) for h in app.visible_hosts())

    run(go())


def test_unreachable_and_driver_error_hosts_render():
    async def go():
        monitor, app = _app()
        async with app.run_test(size=(160, 44)) as pilot:
            await _settle(pilot, monitor)
            down = monitor.view.by_name("dcu-bw1000-3")
            assert down is not None and down.reachable is False
            app._selected = "dcu-bw1000-3"
            app._rendered_rows = []
            app._render()
            await pilot.pause()
            detail = str(app.query_one("#detail").content)
            assert "DOWN" in detail

    run(go())


def test_interval_keys_change_the_poll_rate():
    async def go():
        monitor, app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _settle(pilot, monitor)
            monitor.base_interval = 4.0  # a mid-range value, away from both clamps
            app._render()
            app.action_slower()
            assert monitor.base_interval == pytest.approx(8.0)
            app.action_faster()
            assert monitor.base_interval == pytest.approx(4.0)

    run(go())


def test_painting_is_coalesced_not_per_change():
    run(_painting_is_coalesced())


async def _painting_is_coalesced():
    """A streaming scan must not paint once per reply.

    77 replies in under two minutes used to mean 77 full table rebuilds, which is what
    made the UI feel stuck while the CPU number looked acceptable.
    """
    from jms_top.demo import demo_monitor

    monitor = demo_monitor(interval=3600, full_interval=7200)
    app = JmsTopApp(monitor, boot=False, watch=False)
    async with app.run_test(size=(140, 34)) as pilot:
        await monitor.poll()
        await pilot.pause()
        app._draw_frame()
        # A burst of changes inside one frame window must arm exactly one frame.
        for _ in range(20):
            monitor._touch()
        assert app._frame_handle is not None
        first = app._frame_handle
        for _ in range(20):
            monitor._touch()
        assert app._frame_handle is first, "a second frame was queued inside the window"
        await asyncio.sleep(0.4)
        await pilot.pause()
        assert app._frame_handle is None


def test_input_paints_immediately_instead_of_waiting_for_a_frame():
    run(_input_paints_immediately())


async def _input_paints_immediately():
    """A keypress cannot wait behind a scan: that is what "laggy" feels like."""
    from jms_top.demo import demo_monitor

    monitor = demo_monitor(interval=3600, full_interval=7200)
    app = JmsTopApp(monitor, boot=False, watch=False)
    async with app.run_test(size=(140, 34)) as pilot:
        await monitor.poll()
        await pilot.pause()
        monitor._touch()  # arms a frame
        assert app._frame_handle is not None
        app._paint_now()  # a keypress arrives
        assert app._frame_handle is None, "the queued frame outlived the keypress"


def test_sort_cycles_through_availability_address_and_kind():
    """`u` cycles three orderings; availability stays first because it is the point."""
    from jms_top.tui import SORT_MODES

    monitor = demo_monitor(interval=3600, full_interval=7200)

    async def go():
        app = JmsTopApp(monitor, boot=False, watch=False)
        async with app.run_test(size=(140, 34)) as pilot:
            await monitor.poll()
            await pilot.pause()
            seen = []
            for _ in range(4):
                seen.append(app._sort_mode)
                app.action_cycle_sort()
                await pilot.pause()
            return seen

    assert SORT_MODES[0] == "availability"
    assert run(go()) == ["availability", "address", "kind", "availability"]


def test_address_sort_is_numeric_not_lexicographic():
    """`10.0.0.9` before `10.0.0.10` — a text sort gets this backwards."""
    from jms_top.model import HostView
    from jms_top.tui import _address_key

    rows = [HostView(name="n", address=a) for a in ("10.0.0.10", "10.0.0.9", "10.0.1.1")]
    rows.append(HostView(name="undialled", address=None))
    ordered = [h.address for h in sorted(rows, key=_address_key)]
    assert ordered == ["10.0.0.9", "10.0.0.10", "10.0.1.1", None]


def test_kind_sort_groups_by_accelerator_then_puts_free_first():
    monitor = demo_monitor(interval=3600, full_interval=7200)

    async def go():
        app = JmsTopApp(monitor, boot=False, watch=False)
        async with app.run_test(size=(140, 34)) as pilot:
            await monitor.poll()
            await pilot.pause()
            app._sort_mode = "kind"
            rows = app.visible_hosts()
            labels = [h.family_label for h in rows]
            free_first_ok = all(
                group == sorted(group, key=lambda h: h.verdict != "free")
                for group in _grouped(rows)
            )
            return labels, free_first_ok

    labels, free_first_ok = run(go())
    assert labels == sorted(labels), "families are not contiguous"
    assert free_first_ok, "a free host sat below a busy one inside its family"


def _grouped(rows):  # noqa: ANN001, ANN202
    out, cur = [], []
    for row in rows:
        if cur and row.family_label != cur[0].family_label:
            out.append(cur)
            cur = []
        cur.append(row)
    if cur:
        out.append(cur)
    return out
