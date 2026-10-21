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

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, RichLog, Static

# Imported as a *module*, not as a list of functions: hot reload replaces the module's
# globals, and a `from .panels import summary_bar` binding would keep pointing at the
# function object loaded at startup — the reload would run and change nothing on screen.
from . import panels
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
from .watch import SourceWatcher

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
COLUMNS = (
    "HOST",
    "ADDRESS",
    "KIND",
    "CARD #",
    "CARDS",
    "VRAM",
    "STATE",
    "LOAD",
    "MEM",
    "CPU",
    "UP",
    "USERS",
    "AGE",
)

#: The repaint timer no longer polls at a fixed rate. It used to wake 10 times a second
#: and return early unless the data had moved — which still costs the event loop a
#: wakeup per tick, and measured 44-86% of a core doing nothing at all (with all app
#: timers paused the same app idles at 0.1%). Painting is now driven by the data: the
#: monitor bumps a generation counter when a reply lands, and one task paints per
#: change. The clock in the header still needs to move, but once a second is enough
#: for a countdown, so that is the only thing left on a timer.
#: Retained for the docs/tests that refer to "the repaint path"; painting itself is
#: driven by data now (see :meth:`JmsTopApp._paint`), not by this rate.
REPAINT_HZ = 10

#: Frame rate while data is streaming. 4 fps is enough for numbers that change at
#: seconds' pace, and it is not a smoothness knob: every frame is a full re-read of the
#: 77-row view model (~3 ms), so 10 fps would burn 30% of a core to redraw glyphs
#: nobody can read that fast.
FRAME_INTERVAL = 0.25
#: How often the source watcher stats the package (one ``stat`` per file). Saves are
#: not latency-critical, and a second of lag is unnoticeable while you are reading the
#: screen it is about to change.
WATCH_HZ = 1.0

#: Orderings the table can use. `availability` is the default and what the table
#: returns to, because it is the question the tool exists to answer. The other two are
#: for *finding* a machine rather than a free slot, and are reached by **clicking the
#: column header** — `ADDRESS` sorts by address, `KIND` groups by accelerator — which is
#: where a reader looks to reorder a table. Clicking the same header again goes back to
#: availability, so the useful order is always one click away.
SORT_MODES = ("availability", "address", "kind", "state")

#: Which column header selects which ordering.
SORT_COLUMNS = {"ADDRESS": "address", "KIND": "kind", "STATE": "state"}
#: Fixed widths, wide enough for the longest value each column can hold. Sized from the
#: content would be prettier and would drift mid-scan; a table that resizes under the
#: pointer is not worth it.
COLUMN_WIDTHS = {
    "HOST": 20,
    "ADDRESS": 17,
    "KIND": 13,
    "CARD #": 9,
    "CARDS": 9,
    "VRAM": 9,
    "STATE": 9,
    "LOAD": 6,
    "MEM": 6,
    "CPU": 6,
    "UP": 9,
    "USERS": 16,
    "AGE": 6,
}

#: The same mapping by column position in `COLUMNS`, which is what a click reports.
#: Derived, never written by hand — see the note in `on_data_table_header_selected`.
SORT_BY_INDEX = {list(COLUMNS).index(name): mode for name, mode in SORT_COLUMNS.items()}

DASH = "—"


def _header(col: str) -> Text:
    """A column title; the sortable ones are underlined so they read as clickable.

    `ADDRESS` and `KIND` reorder the table when clicked. Marking just those two costs
    nothing and stops the rest looking broken for not responding.
    """
    if col in SORT_COLUMNS:
        return Text(f"{col} ▾", style="underline")
    return Text(col)


def _vram(host: HostView) -> str:
    """Per-card capacity and how much of it the host is using, e.g. ``144G 73%``.

    Capacity is what distinguishes an 8 x 24 GB box from an 8 x 144 GB one; the
    percentage is what says whether the memory is actually taken. Both, or the column
    answers only half the question a reader has.
    """
    accel = host.accel
    if accel is None or not accel.cards:
        return DASH
    per_card = host.vram_per_card
    if per_card is None:
        return DASH
    used, total = host.vram_used, host.vram_total
    size = f"{per_card / 1024:.0f}G"
    if used is None or not total:
        return size
    return f"{size} {used / total * 100:.0f}%"


def _address_key(host: HostView) -> tuple:
    """Sort by address, numerically within each /24.

    A plain text sort puts `10.0.0.10` before `10.0.0.9`, which reads as random in a
    list of subnets. Sorting the octets as integers groups neighbours the way the rack
    is laid out. Unresolvable addresses sort last rather than first.
    """
    addr = (host.address or "").split(":")[0]  # drop any port
    parts = addr.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return (0, tuple(int(p) for p in parts), host.name)
    return (1, (), host.name)


def verdict_matches(verdict: str, wanted: str) -> bool:
    if wanted == "all":
        return True
    if wanted == "unavailable":
        return verdict in (DOWN, UNKNOWN, "cpu")
    return verdict == wanted


def row_text(value: str, verdict: str) -> Text:
    """One table cell, coloured by the row's verdict."""
    return Text(value, style=panels.VERDICT_STYLE.get(verdict, ""))


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


def _address_cell(host: HostView) -> str:
    """The machine's own address, or dashes when the inventory did not record one.

    Shown next to the name because ``ssh``/``ping``/``curl`` need the address, not the
    inventory key — and the key is often an asset name that resolves to nothing outside
    JumpServer.
    """
    return (host.address or DASH)[:15]


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
        Binding("u", "cycle_sort", "sort", show=False),
        Binding("l", "toggle_log", "log", show=False),
        Binding("ctrl+r", "reload", "reload", show=False),
        Binding("j", "cursor_down", "down", show=False),
        Binding("k", "cursor_up", "up", show=False),
    ]

    def __init__(
        self,
        monitor: FleetMonitor,
        *,
        demo: bool = False,
        boot: bool = True,
        watch: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.demo = demo
        self._boot = boot
        self._watch = watch
        self.filter: str = "all"
        self.needle: str = ""
        self._rendered_generation = -1
        self._rendered_rows: list[str] = []
        self._selected: str | None = None
        self._detail_mode = False
        self._sort_mode = "availability"
        #: name -> current row position. Availability keeps a row in the slot it already
        #: holds instead of re-sorting on every free-card change, so the table does not
        #: jump each time a host's card count moves.
        self._row_rank: dict[str, int] = {}
        #: name -> current row position, so availability can keep rows in place instead
        #: of re-sorting on every free-card change.
        self._row_rank: dict[str, int] = {}
        self._watcher: SourceWatcher | None = None
        self._restart_pending = False
        # True while the widget tree is being rebuilt. The repaint timer keeps firing
        # during that window and would otherwise query widgets that have been removed.
        self._rebuilding = False
        #: Pending coalesced frame (Textual timer handle), or None.
        self._frame_handle = None
        #: Name of the row a programmatic `move_cursor` is heading for. While it is
        #: set, `RowHighlighted` events are ignored — they arrive asynchronously, one
        #: per row the cursor passes, so a flag cleared after the call is too early.
        self._pending_cursor: str | None = None

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
            # Widths are fixed here rather than left to grow with the content. A
            # `CARD #` cell is `—` before a host is scanned and `16/16` afterwards, so
            # content-driven sizing made the columns shift right *during* a scan —
            # which moves every cell under the pointer, and made clicking STATE land
            # on whatever column had slid under the cursor by then.
            table.add_column(_header(col), key=col, width=COLUMN_WIDTHS[col])
        self.query_one("#command", Input).display = False
        self.query_one("#log", RichLog).display = False
        self._render()
        self._shown_at = 0.0
        # Paint on data, not on a clock: the monitor calls this whenever a reply lands.
        # The 1 Hz timer below is only for the "next tick in 2.1s" countdown.
        self.monitor.on_change = self._paint
        # No repaint timer at all. A timer callback in Textual schedules a render even
        # when the handler writes nothing, so a 1 Hz clock cost 22% of a core here and
        # the old 10 Hz one cost most of a core. Everything on screen changes because a
        # reply arrived (which calls `_paint`), a key was pressed, or the selection
        # moved — none of which need a clock.
        if self._watch:
            self._watcher = SourceWatcher()
            # One stat per source per tick: cheap enough to leave on, and it means an
            # edit shows up while you are looking at the screen rather than after a
            # restart you have to remember to do.
            self.set_interval(WATCH_HZ, self._poll_sources)
        if self._boot:
            self.run_worker(self._boot_loop(), exclusive=True, name="boot")

    async def _boot_loop(self) -> None:
        # Ping first: which hosts are up is known in ~a second, before the first full
        # scan finishes, and it is what the "N/M up" counter needs.
        await self.monitor.ping_all()
        self.monitor.start()

    async def on_unmount(self) -> None:
        # The monitor may outlive the app (tests, screenshots); a stale callback into a
        # torn-down widget tree is the one thing worse than a slow repaint.
        self.monitor.on_change = None
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
        # The freeze exists so that a scan *streaming results in* does not reshuffle
        # the table under the cursor — arrivals changing `verdict` would otherwise make
        # rows jump every time one landed. It applies only while the ordering is the
        # one the scan affects, i.e. the default:
        #
        #   * availability: frozen, because that is the order being computed from
        #     data that is still arriving;
        #   * address: sorting is on a field the scan never touches, so it is already
        #     stable and freezing it would only be a no-op;
        #   * kind and state: the *user* just asked for this order. Freezing them was
        #     a bug — it collapsed every mode to inventory order, so clicking STATE
        #     mid-scan changed the label in the footer and nothing else.
        frozen = self.monitor.view.scanning or self._sort_mode == "availability"
        if self._sort_mode == "address":
            return sorted(hosts, key=lambda h: _address_key(h))
        if self._sort_mode == "kind":
            # Within a model, free first: grouping should not bury the useful rows.
            return sorted(hosts, key=lambda h: (h.family_label, h.sort_key()))
        if self._sort_mode == "state":
            # Strictly by verdict. `availability` already puts the bands in order, but
            # it also breaks ties by free-card count, so a busy host with plenty of
            # memory and one at 100% get interleaved across their band. For "show me
            # everything that is DOWN" the bands must be contiguous.
            return sorted(hosts, key=lambda h: (h.sort_key()[0], h.name))
        if frozen and self._rendered_rows:
            # Freeze means *freeze*: keep the exact order the table already shows and
            # append rows that have appeared since. Re-deriving a "stable" order from
            # names still re-sorts, and during a scan the data changes many times a
            # second — every re-sort moved the cursor, which fired a highlight event,
            # which rewrote the selection: 1500 selection changes in a few seconds, and
            # a side panel showing a different host than the highlighted row.
            rank = {n: i for i, n in enumerate(self._rendered_rows)}
            known = [h for h in hosts if h.name in rank]
            known.sort(key=lambda h: rank[h.name])
            fresh = [h for h in hosts if h.name not in rank]
            if fresh:
                fresh.sort(key=lambda h: self._availability_key(h, frozen=False))
            return known + fresh
        return sorted(hosts, key=lambda h: self._availability_key(h, frozen=frozen))

    def _availability_key(self, host: HostView, *, frozen: bool) -> tuple:
        """Band first; inside a band, keep the row where it already is.

        The band comes from the verdict, so a host that frees up moves up — that is
        the whole point of the tool. The *tie-break* is the position the row already
        holds, not its free-card count: sorting on cards means the table re-shuffles
        every time a host's count changes, which happens on every full round (60 s by
        default) as other people start and stop work. That periodic jump is what made
        the default view look like it kept switching to a state-grouped ordering.

        A row that is frozen (mid-scan, or just promoted) keeps its slot by name.
        """
        if frozen and host.name in self._row_rank:
            return (host.sort_key()[0], 0, self._row_rank[host.name], host.name)
        return (host.sort_key()[0], 1, 0, host.name)

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
        """Paint if the data moved. Kept for callers that want a no-op-when-idle paint."""
        if self._rebuilding:
            return
        if self.monitor.generation != self._rendered_generation:
            self._paint()

    def _paint(self) -> None:
        """Schedule a frame; at most one is drawn per :data:`FRAME_INTERVAL`.

        Painting immediately on every change is what made the interface feel stuck: a
        scan over 77 hosts fires ~77 changes in under two minutes, and each one rebuilt
        the whole table. Painting on a timer instead was worse still — Textual schedules
        a render for every timer callback, so a 1 Hz clock sat at 22% and a 10 Hz one at
        most of a core whether or not anything had changed.

        Trailing-edge coalescing gets both: a change arms one delayed frame, later
        changes in the same window join it, and the frame is drawn once. The screen
        updates at `FRAME_INTERVAL` while data streams in (smooth, bounded cost) and
        immediately when nothing is streaming (a keypress paints on the next frame).
        """
        if self._rebuilding:
            return
        if self._frame_handle is not None:
            return  # a frame is already on its way; this change joins it
        self._frame_handle = self.set_timer(FRAME_INTERVAL, self._draw_frame)

    def _paint_now(self) -> None:
        """Paint this instant, cancelling a queued frame — for input that must feel live.

        Keystrokes, cursor moves and mode changes are user-driven and rare; making them
        wait up to a frame for a scan to finish is what "sluggish" means. A streaming
        reply can afford to coalesce, a keypress cannot.
        """
        if self._frame_handle is not None:
            self._frame_handle.stop()
            self._frame_handle = None
        self._draw_frame()

    def _draw_frame(self) -> None:
        self._frame_handle = None
        if self._rebuilding:
            return
        self._painted_at = time.monotonic()
        self._shown_at = self._painted_at
        self._render()

    def _render(self) -> None:
        if self._rebuilding:
            return
        self._rendered_generation = self.monitor.generation
        view = self.monitor.view
        self.query_one("#titlebar", Static).update(self._title(view))
        self.query_one("#summary", Static).update(panels.summary_bar(view))
        self._render_table()
        selected = self.selected_view()
        self.query_one("#cards", Static).update(
            Text("") if self._detail_mode else panels.cards_panel(selected)
        )
        self.query_one("#detail", Static).update(
            self._process_table(selected) if self._detail_mode else panels.detail_view(selected)
        )
        self.query_one("#side-title", Static).update(
            "PROCESSES  (esc)" if self._detail_mode else "CARDS"
        )
        self._render_footer(view)

    def _title(self, view: FleetView) -> Text:
        line = panels.title_bar(view, paused=self.monitor.paused, selected=self._selected)
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
            # A reorder rebuilds. Writing cells into their new positions is cheaper, but
            # `update_cell_at` only replaces *content* — the rows keep the keys they
            # were added with, so `ordered_rows` stays in the old order while the screen
            # shows the new one. Row indices then mean two different things, and
            # `move_cursor(row=i)` lands on the wrong host: the cursor and the side
            # panel disagreed for exactly this reason.
            #
            # The earlier objection to rebuilding was that `clear()` + re-add leaves a
            # window where a click is hit-tested against the old layout. That is now
            # covered: column widths are fixed, and the reorder rate dropped to "only
            # when a band actually changes" when availability stopped re-sorting on
            # every free-card count.
            table.clear()
            for host in hosts:
                table.add_row(*self._cells(host), key=host.name)
            self._rendered_rows = names
            self._row_rank = {n: i for i, n in enumerate(names)}
            self._selected = cursor_name if cursor_name in names else (names[0] if names else None)
            if self._selected is not None:
                self._pending_cursor = self._selected
                table.move_cursor(row=names.index(self._selected))
        else:
            # `update_width=False` is the whole point: each cell update otherwise
            # recomputes the column widths, so 77 rows x 13 columns meant ~1000 width
            # recalculations per frame and a 560 ms repaint — the table felt frozen
            # while a scan streamed results in. The widths only need refreshing once,
            # after every cell is written.
            for i, host in enumerate(hosts):
                for j, cell in enumerate(self._cells(host)):
                    # update_width=False: recomputing widths per cell is what made a
                    # frame cost 560 ms, *and* it changes the layout under the pointer
                    # mid-scan. Widths are set once when the columns are added.
                    table.update_cell_at((i, j), cell, update_width=False)
            # Same rows, same order: only the cells need rewriting. The table's keys
            # already match `names`, so no cursor work is needed.

    def _cells(self, host: HostView) -> list[Text]:
        verdict = host.verdict
        cards = f"{host.free_cards}/{host.total}" if host.total else DASH
        users = ",".join(host.users) or ",".join(sorted(set(host.node.logins)))
        load = None if host.node.load_per_cpu is None else host.node.load_per_cpu * 100
        return [
            row_text(host.name, verdict),
            row_text(_address_cell(host), verdict),
            row_text(host.family_label, verdict),
            row_text(_map(host), verdict),
            row_text(cards, verdict),
            row_text(_vram(host), verdict),
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
        # The current ordering is shown, not just the key that changes it: after a
        # `u` the table looks shuffled for no stated reason otherwise.
        text.append(f"  [{self._sort_mode}]", style="bold cyan")
        if self._watcher is not None:
            text.append("  ^r reload", style="dim")
        if self._restart_pending:
            # Said once, in the place a key would have been: a reload that silently
            # left half the change unapplied is worse than one that asks.
            text.append("   │ ", style="dim")
            text.append("restart to apply monitor/parser changes", style="bold yellow")
        if view.log:
            text.append("   │ ", style="dim")
            text.append(view.log[-1][11:], style="")
        self.query_one("#footer-bar", Static).update(text)

    # --- messages --------------------------------------------------------------

    @property
    def _live(self) -> bool:
        """Whether the widget tree is present and paintable.

        `is_mounted` is not enough: `recompose()` tears the children down while the app
        itself stays mounted, and a highlight emitted during teardown would then paint
        against widgets that no longer exist (`NoMatches: #titlebar`). Asking for one
        of our own widgets is the direct test.
        """
        try:
            self.query_one("#titlebar", Static)
            return True
        except Exception:  # noqa: BLE001 - any failure means "cannot paint"
            return False

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the cursor as the user moves it.

        Guarded against our own `move_cursor` calls: keeping the cursor on the selected
        host after a reorder emits this event too, and if that event were taken at face
        value it would *overwrite* the selection with whatever row we had just moved
        past. That feedback loop made the highlight and the side panel disagree on
        almost every frame of a scan.
        """
        if self._rebuilding or not self._live:
            return
        key = event.row_key
        if key is None or key.value is None:
            return
        name = str(key.value)
        if self._pending_cursor is not None:
            # A programmatic move is settling. Ignore highlights until the cursor
            # actually arrives at the row we asked for, then clear the expectation.
            if name == self._pending_cursor:
                self._pending_cursor = None
            return
        if name != self._selected:
            self._selected = name
            self._render()
        # `name == self._selected` means this highlight was caused by our own
        # `move_cursor` keeping the cursor on the selection — nothing to do, and
        # re-rendering here would loop. A *user* moving the cursor produces a name
        # that differs from `_selected`, which is the case above. That is why this
        # compares names instead of using a "we are moving the cursor" flag: cursor
        # moves deliver `RowHighlighted` asynchronously, so a flag set and cleared
        # around `move_cursor()` is already false by the time the event arrives.

    def on_input_changed(self, event: Input.Changed) -> None:
        self.needle = event.value.strip()
        self._rendered_rows = []  # membership changed: rebuild rather than update
        self._render_table()
        self.query_one("#titlebar", Static).update(self._title(self.monitor.view))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._hide_command()

    # --- live reload -----------------------------------------------------------

    def _poll_sources(self) -> None:
        """Apply a change to the sources mid-run (see :mod:`jms_top.watch`)."""
        if self._watcher is None or self._rebuilding:
            # A rebuild in flight will re-stat on its own next tick; queueing here would
            # stack reloads on top of each other while the widgets are half-swapped.
            return
        change = self._watcher.check()
        if change.any:
            self.apply_change(change)

    def apply_change(self, change) -> None:  # noqa: ANN001 - watch.Change
        """Reload the changed modules and repaint, keeping the monitor and its data.

        The monitor is deliberately untouched: its SSH connections and its last round
        are the state worth keeping, and rebuilding the widgets around them costs one
        frame. Modules that cannot be swapped safely (the poller, the parsers) are
        reported instead of reloaded — a half-swapped parser reading a live reply is a
        much worse outcome than a line of text asking for a restart.
        """
        from .watch import describe, reload_modules

        reloaded, failed = reload_modules(change.reloadable)
        if change.stylesheet:
            self.refresh_css()
        if failed:
            self.log(f"reload failed (still running the old code): {failed[0]}"[:160])
            return
        if change.needs_restart:
            self._restart_pending = True
        self.log(f"hot reload: {describe(change)}"[:160])
        # Not `run_worker(exclusive=True)`: a second save while the first rebuild is
        # still mounting would cancel it mid-swap and leave an empty screen. The
        # `_rebuilding` guard above keeps them serial instead.
        self.run_worker(self._rebuild(), exclusive=False, name="reload")

    async def _rebuild(self) -> None:
        """Recompose every widget from the live monitor state.

        ``recompose()`` removes the children and calls ``compose`` again — and because
        ``compose`` builds *new* widget objects each time, the new tree is drawn by the
        reloaded ``panels`` module. (Driving ``remove``/``mount_all`` by hand does not
        work: ``compose`` uses ``with`` blocks that need the stack ``recompose`` sets
        up, and calling it outside that raises ``IndexError``.)

        Painting is suppressed for the swap (a paint would query widgets that have just
        been removed), and one is forced at the end: with no repaint timer there is
        nothing else to schedule it.
        """
        self._rendered_rows = []
        self._rendered_generation = -1
        self._rebuilding = True
        try:
            await self.recompose()
            await self._after_rebuild()
        finally:
            # A failed rebuild must leave a repaintable screen behind, not a frozen one;
            # the widgets that did mount are the ones just composed.
            self._rebuilding = False
            self._rendered_generation = -1
            # Nobody else will paint this frame: the swap happened outside any timer.
            self._render()

    async def _after_rebuild(self) -> None:
        """Re-apply mount-time wiring on the fresh widgets, then paint.

        Idempotent on purpose: ``recompose()`` re-runs ``compose``, which returns the
        *same* widget objects, so a DataTable that already has its columns is not
        "fresh" — it is the one that was on screen a moment ago. Clearing first makes
        the rebuild safe to run any number of times.
        """
        table = self.query_one("#table", DataTable)
        table.clear(columns=True)
        for col in COLUMNS:
            # Widths are fixed here rather than left to grow with the content. A
            # `CARD #` cell is `—` before a host is scanned and `16/16` afterwards, so
            # content-driven sizing made the columns shift right *during* a scan —
            # which moves every cell under the pointer, and made clicking STATE land
            # on whatever column had slid under the cursor by then.
            table.add_column(_header(col), key=col, width=COLUMN_WIDTHS[col])
        self.query_one("#command", Input).display = False
        self.query_one("#log", RichLog).display = False
        self._rendered_rows = []
        self._render()
        # One idle tick so the freshly mounted widgets are laid out before the paint
        # that follows; without it the first frame after a reload can be a frame of
        # empty cells.
        await asyncio.sleep(0)

    def action_reload(self) -> None:
        """Force a reload now — for an edit the watcher missed, or after a checkout."""
        from .watch import Change

        self.apply_change(Change({"jms_top.panels", "jms_top.render", "jms_top.tui"}, set(), True))

    # --- actions ---------------------------------------------------------------

    def action_cycle_filter(self) -> None:
        self.filter = FILTERS[(FILTERS.index(self.filter) + 1) % len(FILTERS)]
        self._rendered_rows = []
        self._paint_now()

    def action_search(self) -> None:
        board = self.query_one("#command", Input)
        board.display = True
        board.value = self.needle
        self.set_focus(board)

    def action_close_overlay(self) -> None:
        if self._detail_mode:
            self._detail_mode = False
            self._paint_now()
            return
        if self.query_one("#command", Input).display:
            self._hide_command()

    def _hide_command(self) -> None:
        board = self.query_one("#command", Input)
        board.display = False
        self.set_focus(self.query_one("#table", DataTable))

    def action_inspect(self) -> None:
        self._detail_mode = True
        self._paint_now()

    def action_refresh(self) -> None:
        if hasattr(self.monitor, "refresh_now"):
            self.run_worker(self.monitor.refresh_now(), exclusive=False, name="refresh")

    def action_toggle_pause(self) -> None:
        if self.monitor.paused:
            self.monitor.resume()
        else:
            self.monitor.pause()
        self._paint_now()

    def action_slower(self) -> None:
        self.monitor.base_interval = min(60.0, self.monitor.base_interval * 2)
        self.monitor.view.interval = self.monitor.base_interval
        self._paint_now()

    def action_faster(self) -> None:
        self.monitor.base_interval = max(1.0, self.monitor.base_interval / 2)
        self.monitor.view.interval = self.monitor.base_interval
        self._paint_now()

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Clicking a header sorts by that column; clicking it again goes back.

        The table headers are the affordance a reader already knows for "reorder this",
        so `ADDRESS` and `KIND` are clickable. Any other header, or the same one twice,
        returns to availability — which has to stay reachable without a second key,
        because "where are the free machines" is the question this tool answers.
        """
        # Match on the column *position*, not the rendered label. The label carries
        # the `▾` clickable marker, so keying on text means the decoration and the
        # lookup have to be kept in sync by hand — which they were not, and the first
        # version of this handler silently did nothing on every click. The index is
        # stable because `COLUMNS` is what built the table.
        wanted = (
            SORT_BY_INDEX.get(event.column_index)
            if 0 <= event.column_index < len(COLUMNS)
            else None
        )
        label = COLUMNS[event.column_index] if wanted else ""
        # Clicking the active column restores the default. That is fine between rounds,
        # but during one it bites: the table is frozen to inventory order, so the first
        # click looks like it did nothing (the rows do not move yet), the second click
        # cancels it, and the user is left believing the header is broken. So while a
        # round is in flight the mode only ever *advances* — it cannot be cancelled by
        # an impatient double click. Availability is still one click away once the scan
        # finishes.
        if wanted is None:
            self._sort_mode = "availability"
        elif self.monitor.view.scanning and wanted == self._sort_mode:
            pass  # already chosen: leave it, do not toggle off mid-scan
        else:
            self._sort_mode = "availability" if wanted == self._sort_mode else wanted
        self._rendered_rows = []
        self._paint_now()
        # Logged whether or not the mode changed: a click that lands on a header and
        # does nothing looks identical to no click at all, and the difference is
        # exactly what is needed to diagnose "it reordered itself".
        self.log(f"header clicked: {label or '?'} -> sort={self._sort_mode}")

    def action_toggle_log(self) -> None:
        """Show the event log — every sort change, reload and error lands here.

        It was mounted and hidden with no way to reveal it, so `sorted by state` (and
        any reload failure) went somewhere the user could not look.
        """
        log = self.query_one("#log", RichLog)
        log.display = not log.display
        self._paint_now()

    def action_cycle_sort(self) -> None:
        """Cycle the ordering: availability -> address -> kind -> availability.

        Availability stays the default because it is the question the tool exists to
        answer. The other two are for finding a machine rather than a free slot:
        `address` sorts numerically inside each subnet (so `.9` precedes `.10`, which a
        text sort gets wrong), `kind` groups by accelerator model.
        """
        self._sort_mode = SORT_MODES[(SORT_MODES.index(self._sort_mode) + 1) % len(SORT_MODES)]
        self._rendered_rows = []
        self._paint_now()
        self.log(f"sort: {self._sort_mode}")

    def action_sort_kind(self) -> None:
        """Kept for the `u` binding's history; cycles like :meth:`action_cycle_sort`."""
        self.action_cycle_sort()

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
