"""The TUI: a fleet table, the selected host's cards, and a detail block.

Interaction model (nvitop's, adapted to a fleet):

* ``↑/↓`` (or ``j/k``) moves through hosts; the side panel and detail block follow
  the cursor with no round trip — everything is already in memory.
* ``f`` cycles the *available* filter (all → free → partial → residual → busy →
  unavailable). Filtering never triggers a network call.
* ``/`` filters by text: host name, tag, group, platform, user or process name. It
  is a live substring match over the fleet view, so typing feels instant.
* ``enter`` opens the host's accelerator process table; ``esc`` returns.
* ``r`` forces a full round now; ``space`` pauses; ``i``/``o`` change the interval;
  ``u`` sorts by availability or by accelerator family; ``q`` quits.

The UI never blocks: every network call lives in :class:`FleetMonitor`, which runs in
a background task and bumps a generation counter. A 10 Hz repaint timer re-renders
only when that counter (or the cursor, or the filter) has actually moved, so a stalled
tick shows as an ageing "seen" column rather than a frozen screen.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, RichLog, Static

from .model import (
    BUSY,
    DOWN,
    FREE,
    PARTIAL,
    RESIDUAL,
    UNKNOWN,
    FleetView,
    HostView,
)
from .monitor import FleetMonitor
from .panels import VERDICT_STYLE, cards_panel, detail_view, summary_bar, title_bar

#: The filter cycle, in the order ``f`` walks it. ``all`` first, so one keypress from
#: the default view is "show me only what I can take".
FILTERS: list[str] = ["all", FREE, PARTIAL, RESIDUAL, BUSY, "unavailable"]

FILTER_LABELS = {
    "all": "ALL HOSTS",
    FREE: "FREE ONLY",
    PARTIAL: "PARTIAL ONLY",
    RESIDUAL: "RESIDUAL ONLY",
    BUSY: "BUSY ONLY",
    "unavailable": "DOWN / UNKNOWN",
}

#: Column order. DataTable sizes each column to its widest cell.
COLUMNS = ("HOST", "KIND", "CARD #", "CARDS", "STATE", "LOAD", "MEM", "CPU", "UP", "USERS", "AGE")

#: How often the screen is repainted *if* something changed.
REPAINT_HZ = 10

DASH = "—"


def verdict_matches(verdict: str, wanted: str) -> bool:
    if wanted == "all":
        return True
    if wanted == "unavailable":
        return verdict in (DOWN, UNKNOWN, "cpu")
    return verdict == wanted


def row_text(value: str, verdict: str) -> Text:
    """One table cell, coloured by the row's verdict."""
    return Text(value, style=VERDICT_STYLE.get(verdict, ""))


def _fmt_pct(value: float | None) -> str:
    return DASH if value is None else f"{value:.0f}%"


def _fmt_uptime(seconds: float | None) -> str:
    if seconds is None:
        return DASH
    days, rem = divmod(int(seconds), 86400)
    return f"{days}d{rem // 3600:02d}h" if days else f"{rem // 3600}h{rem % 3600 // 60:02d}m"


def age_text(when: datetime | None) -> str:
    """How old a reading is, as the table shows it."""
    if when is None:
        return DASH
    secs = max(0, (datetime.now(timezone.utc) - when).total_seconds())
    if secs < 90:
        return f"{int(secs)}s"
    if secs < 5400:
        return f"{int(secs // 60)}m"
    return f"{int(secs // 3600)}h"


def _map(view: HostView) -> str:
    from .render import card_cells

    return card_cells(view) or DASH


class JmsTopApp(App):
    """The application shell. Owns no networking — that is the monitor's job."""

    CSS_PATH = Path(__file__).with_name("app.tcss")
    TITLE = "jms-top"
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("f", "cycle_filter", "filter"),
        Binding("slash", "search", "search"),
        Binding("escape", "close_overlay", "close"),
        Binding("enter", "inspect", "procs"),
        Binding("r", "refresh", "refresh"),
        Binding("space", "toggle_pause", "pause"),
        Binding("i", "slower", "interval+"),
        Binding("o", "faster", "interval-"),
        Binding("u", "sort_kind", "sort"),
        Binding("j", "cursor_down", "down", show=False),
        Binding("k", "cursor_up", "up", show=False),
    ]

    def __init__(self, monitor: FleetMonitor, *, demo: bool = False, boot: bool = True):
        super().__init__()
        self.monitor = monitor
        self.demo = demo
        self._boot = boot
        self.filter: str = "all"
        self.needle: str = ""
        self._rendered_generation = -1
        self._rendered_rows: list[str] = []
        self._selected: str | None = None
        self._detail_mode = False
        self._sort_mode = "availability"

    # --- composition -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="titlebar")
            yield Static(id="summary")
            with Horizontal(id="body"):
                with Vertical(id="fleet-pane"):
                    yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
                with Vertical(id="side"):
                    yield Static("CARDS", id="side-title")
                    yield Static(id="cards")
                    yield Static(id="detail")
            yield Static(id="footer-bar")
            yield Input(placeholder="name / tag / group / user / process", id="command")
            yield RichLog(id="log", max_lines=200, markup=False, wrap=False)

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        for col in COLUMNS:
            table.add_column(col, key=col)
        self.query_one("#command", Input).display = False
        self.query_one("#log", RichLog).display = False
        self._render()
        self._shown_at = 0.0
        self.set_interval(1.0 / REPAINT_HZ, self._repaint)
        if self._boot:
            self.run_worker(self._boot_loop(), exclusive=True, name="boot")

    async def _boot_loop(self) -> None:
        # Ping first: which hosts are up is known in ~a second, before the first full
        # scan finishes, and it is what the "N/M up" counter needs.
        await self.monitor.ping_all()
        self.monitor.start()

    async def on_unmount(self) -> None:
        # An async handler: `run_worker` would create a worker that the app is already
        # tearing down, and a bare call would leave the coroutine un-awaited.
        await self.monitor.close()

    # --- data → screen ---------------------------------------------------------

    def visible_hosts(self) -> list[HostView]:
        hosts = [
            h
            for h in self.monitor.view.hosts
            if verdict_matches(h.verdict, self.filter) and self._matches(h)
        ]
        if self._sort_mode == "availability":
            return sorted(hosts, key=lambda h: h.sort_key)
        return sorted(hosts, key=lambda h: (h.family_label, h.name))

    def _matches(self, host: HostView) -> bool:
        if not self.needle:
            return True
        needle = self.needle.lower()
        hay = " ".join(
            [
                host.name,
                host.family_label,
                host.group or "",
                " ".join(host.tags),
                " ".join(host.node.logins),
                " ".join(host.users),
                " ".join(p["user"] for p in host.node.top),
                " ".join(p["cmd"] for p in host.node.top),
            ]
        ).lower()
        return needle in hay

    def selected_view(self) -> HostView | None:
        return None if self._selected is None else self.monitor.view.by_name(self._selected)

    def _repaint(self) -> None:
        """Repaint only when the data actually moved, or once a second for the clocks."""
        now = time.monotonic()
        if (
            self.monitor.generation != self._rendered_generation
            or now - getattr(self, "_shown_at", 0.0) >= 1.0
        ):
            self._shown_at = now
            self._render()

    def _render(self) -> None:
        self._rendered_generation = self.monitor.generation
        view = self.monitor.view
        self.query_one("#titlebar", Static).update(self._title(view))
        self.query_one("#summary", Static).update(summary_bar(view))
        self._render_table()
        selected = self.selected_view()
        self.query_one("#cards", Static).update(
            Text("") if self._detail_mode else cards_panel(selected)
        )
        self.query_one("#detail", Static).update(
            self._process_table(selected) if self._detail_mode else detail_view(selected)
        )
        self.query_one("#side-title", Static).update(
            "PROCESSES  (esc)" if self._detail_mode else "CARDS"
        )
        self._render_footer(view)

    def _title(self, view: FleetView) -> Text:
        line = title_bar(view, paused=self.monitor.paused, selected=self._selected)
        if self.filter != "all":
            line.append("  [", style="dim")
            line.append(FILTER_LABELS.get(self.filter, self.filter), style="bold yellow")
            line.append("]", style="dim")
        if self.needle:
            line.append("  /", style="dim")
            line.append(self.needle, style="bold cyan")
        if self.demo:
            line.append("  DEMO", style="reverse")
        return line

    def _render_table(self) -> None:
        table = self.query_one("#table", DataTable)
        hosts = self.visible_hosts()
        names = [h.name for h in hosts]
        cursor_name = self._selected
        if names != self._rendered_rows:
            # Rows moved: rebuild. With a stable order only cells change, and those are
            # updated in place — a rebuild would steal the cursor on every tick.
            table.clear()
            for host in hosts:
                table.add_row(*self._cells(host), key=host.name)
            self._rendered_rows = names
            self._selected = cursor_name if cursor_name in names else (names[0] if names else None)
            if self._selected is not None:
                table.move_cursor(row=names.index(self._selected))
        else:
            for i, host in enumerate(hosts):
                for j, cell in enumerate(self._cells(host)):
                    table.update_cell_at((i, j), cell, update_width=True)

    def _cells(self, host: HostView) -> list[Text]:
        verdict = host.verdict
        cards = f"{host.free_cards}/{host.total}" if host.total else DASH
        users = ",".join(host.users) or ",".join(sorted(set(host.node.logins)))
        load = None if host.node.load_per_cpu is None else host.node.load_per_cpu * 100
        return [
            row_text(host.name, verdict),
            row_text(host.family_label, verdict),
            row_text(_map(host), verdict),
            row_text(cards, verdict),
            row_text(verdict.upper(), verdict),
            row_text(_fmt_pct(load), verdict),
            row_text(_fmt_pct(host.node.mem_pct), verdict),
            row_text(_fmt_pct(host.node.cpu_pct), verdict),
            row_text(_fmt_uptime(host.node.uptime_s), verdict),
            row_text(users[:16] or "-", verdict),
            row_text(age_text(host.updated_at), verdict),
        ]

    def _process_table(self, host: HostView | None) -> Text:
        t = Text()
        if host is None:
            return t
        procs = host.procs
        t.append(f"{host.name} — {len(procs)} accelerator processes\n", style="bold")
        if not procs:
            t.append("no process holds a card on this host", style="dim")
            return t
        t.append(f"{'CARD':>4}  {'PID':>8}  {'MEM':>7}  NAME\n", style="bold")
        for card, proc in procs:
            t.append(f"{card.index:>4}  {str(proc.pid) if proc.pid else DASH:>8}  ", style="")
            mem = f"{proc.mem_mib / 1024:.1f}G" if proc.mem_mib is not None else DASH
            t.append(f"{mem:>7}  ", style="dim")
            t.append(f"{proc.name}\n", style="")
        return t

    def _render_footer(self, view: FleetView) -> None:  # noqa: C901
        keys = [
            "↑↓ select",
            "f filter",
            "/ search",
            "enter procs",
            "r refresh",
            "space pause",
            "i/o interval",
            "u sort",
            "q quit",
        ]
        text = Text("  ".join(keys), style="dim")
        if view.log:
            text.append("   │ ", style="dim")
            text.append(view.log[-1][11:], style="")
        self.query_one("#footer-bar", Static).update(text)

    # --- messages --------------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key
        if key is not None and key.value is not None:
            self._selected = str(key.value)
            self._render()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.needle = event.value.strip()
        self._rendered_rows = []  # membership changed: rebuild rather than update
        self._render_table()
        self.query_one("#titlebar", Static).update(self._title(self.monitor.view))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._hide_command()

    # --- actions ---------------------------------------------------------------

    def action_cycle_filter(self) -> None:
        self.filter = FILTERS[(FILTERS.index(self.filter) + 1) % len(FILTERS)]
        self._rendered_rows = []
        self._render()

    def action_search(self) -> None:
        board = self.query_one("#command", Input)
        board.display = True
        board.value = self.needle
        self.set_focus(board)

    def action_close_overlay(self) -> None:
        if self._detail_mode:
            self._detail_mode = False
            self._render()
            return
        if self.query_one("#command", Input).display:
            self._hide_command()

    def _hide_command(self) -> None:
        board = self.query_one("#command", Input)
        board.display = False
        self.set_focus(self.query_one("#table", DataTable))

    def action_inspect(self) -> None:
        self._detail_mode = True
        self._render()

    def action_refresh(self) -> None:
        if hasattr(self.monitor, "refresh_now"):
            self.run_worker(self.monitor.refresh_now(), exclusive=False, name="refresh")

    def action_toggle_pause(self) -> None:
        if self.monitor.paused:
            self.monitor.resume()
        else:
            self.monitor.pause()
        self._render()

    def action_slower(self) -> None:
        self.monitor.base_interval = min(60.0, self.monitor.base_interval * 2)
        self.monitor.view.interval = self.monitor.base_interval
        self._render()

    def action_faster(self) -> None:
        self.monitor.base_interval = max(1.0, self.monitor.base_interval / 2)
        self.monitor.view.interval = self.monitor.base_interval
        self._render()

    def action_sort_kind(self) -> None:
        self._sort_mode = "kind" if self._sort_mode == "availability" else "availability"
        self._rendered_rows = []
        self._render()

    def action_cursor_down(self) -> None:
        self.query_one("#table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#table", DataTable).action_cursor_up()


def run_tui(monitor: FleetMonitor, *, demo: bool = False) -> None:
    JmsTopApp(monitor, demo=demo).run()


async def screenshot(monitor: FleetMonitor, path: str, *, demo: bool = False) -> None:
    """Render one frame of the real app to ``path`` — how the docs get their pictures.

    It drives :class:`JmsTopApp` through Textual's headless driver at 170x44 and
    exports Textual's own SVG, so the image is what the terminal would show, not a
    re-drawing of it.
    """
    app = JmsTopApp(monitor, demo=demo, boot=False)
    async with app.run_test(size=(170, 44)) as pilot:
        await monitor.ping_all()
        await monitor.poll()
        app._rendered_rows = []
        app._rendered_generation = -1
        app._render()
        await pilot.pause()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(app.export_screenshot())
