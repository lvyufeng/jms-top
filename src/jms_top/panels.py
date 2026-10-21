"""Per-host panels: the title bar, the card side panel and the detail block.

Rendered with Rich rather than as more DataTable columns, because these are small
fixed-width figures arranged by position — a card is a *picture* of one device, not a
row in a table, and 16 cards fit on one line where 16 rows would not.

Style rules shared with the fleet table (``render.py``): a value that was not
reported is ``—``, never ``0``; ``unknown`` is its own colour, never green; and the
card strip's letters mean the same thing everywhere.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.text import Text

from .model import FREE, RESIDUAL, FleetView, HostView

#: Colour per verdict and per card state — one table, so the fleet table, the card
#: strip and the detail block can never disagree about what orange means.
VERDICT_STYLE = {
    FREE: "bold green",
    "partial": "bold yellow",
    "busy": "bold red",
    RESIDUAL: "bold dark_orange",
    "unknown": "bold magenta",
    "cpu": "bold cyan",
    "down": "dim",
}

CARD_STYLE = {
    "free": "green",
    "busy": "red",
    "unknown": "magenta",
    "residual": "dark_orange",
}

DASH = "—"


def _pct(v: float | None) -> str:
    return DASH if v is None else f"{v:.0f}%"


def _gib(mib: float | None) -> str:
    return DASH if mib is None else f"{mib / 1024:.0f}G"


def _age(view: HostView, now: datetime | None = None) -> str:
    if view.updated_at is None:
        return DASH
    now = now or datetime.now(timezone.utc)
    secs = max(0, int((now - view.updated_at).total_seconds()))
    if secs < 90:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    return f"{secs // 3600}h"


def card_state(card) -> str:  # noqa: ANN001 - Card from families
    if card.residual:
        return "residual"
    return card.state


# --- title bar ----------------------------------------------------------------


def title_bar(view: FleetView, *, paused: bool | None = None, selected: str | None = None) -> Text:
    """One line: what is being watched, how fresh it is, and what is free."""
    t = Text()
    t.append(" jms-top ", style="bold reverse")
    t.append(" ")
    pct = (view.free_cards / view.total_cards * 100) if view.total_cards else 0.0
    t.append(f"{view.free_cards}", style="bold green")
    t.append(f"/{view.total_cards} cards free ", style="dim")
    t.append(f"({pct:.0f}%)", style="green" if pct else "red")
    t.append("  ")
    t.append(f"{len(view.free_hosts)}", style="bold green")
    t.append(" free hosts  ", style="dim")
    if view.residual_cards:
        t.append(f"{view.residual_cards} residual  ", style="dark_orange")
    if view.unknown_cards:
        t.append(f"{view.unknown_cards} unknown  ", style="magenta")
    t.append("│ ", style="dim")
    t.append(f"{view.reachable}/{len(view.hosts)} up", style="dim")
    t.append(" │ ", style="dim")
    # The countdown is what tells a glance whether the view is live or wedged: a tick
    # that hangs leaves the number frozen at 0, where a "last updated 4s ago" would
    # also be true of a view that is simply paused.
    now = datetime.now(timezone.utc)
    # A pump that never ran has no timestamps at all; a *paused* pump keeps its last
    # countdown, which would read as "about to tick" while nothing is scheduled.
    stalled = view.next_refresh is None or (
        not paused and (now - view.next_refresh).total_seconds() > view.interval * 2
    )
    if paused:
        t.append("PAUSED", style="bold yellow on grey23")
    elif view.next_refresh is None:
        t.append("starting…", style="dim")
    else:
        left = (view.next_refresh - datetime.now(timezone.utc)).total_seconds()
        fresh = (
            (datetime.now(timezone.utc) - view.round_finished).total_seconds()
            if (view.round_finished)
            else None
        )
        if stalled:
            age = (now - view.next_refresh).total_seconds() if view.next_refresh else 0
            t.append(f"tick overdue {age:.0f}s", style="bold red")
        elif left >= 0.5:
            t.append(f"next tick in {left:.1f}s", style="dim")
        else:
            t.append("polling…", style="bold yellow")
        if fresh is not None:
            t.append(f" · full {_fmt_secs(int(fresh))} ago", style="dim")
    if selected:
        t.append(" │ ", style="dim")
        t.append(selected, style="bold")
    return t


def _fmt_secs(secs: int) -> str:
    secs = max(0, secs)
    if secs < 90:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    return f"{secs // 3600}h"


def summary_bar(view: FleetView) -> Text:
    """Second line: the fleet's composition by accelerator family."""
    t = Text(" ")
    fams = sorted(view.families.items(), key=lambda kv: -kv[1])
    for label, count in fams:
        t.append(f"{label}", style="bold")
        t.append(f" {count}  ", style="dim")
    if not fams:
        t.append("no accelerator tools answered — every host is reported CPU-only", style="dim")
    return t


# --- card side panel ----------------------------------------------------------


def cards_panel(view: HostView | None) -> Text:
    """The selected host's cards, one per line, nvitop-style.

    Each line: ``#0  ████░░░░  98%  75/80G  52C  544W  3p  python3.12``. The bar is
    the card's *commit* (memory held) rather than instantaneous utilisation: on a
    shared cluster a card is "taken" the moment its memory is allocated, which is
    what decides whether you can have it, not whether a kernel is running this
    millisecond.
    """
    t = Text()
    if view is None:
        t.append("no host selected", style="dim")
        return t
    head = Text()
    head.append(f"{view.name}", style="bold")
    head.append(f"  {view.family_label}", style="dim")
    if view.cards:
        head.append(f"  {view.free_cards}/{view.total} free", style="dim")
    t.append(head)
    t.append("\n")
    if view.reachable is False:
        t.append(f"unreachable: {view.error or 'no reply'}", style="red")
        return t
    if view.accel is not None and view.accel.error:
        t.append(f"{view.family_label}: {view.accel.error}", style="magenta")
        t.append("\n")
    if not view.cards:
        t.append("no accelerator table reported", style="dim")
        return t
    for card in view.cards:
        state = card_state(card)
        style = CARD_STYLE[state]
        t.append(f"{card.index:>2}", style="dim")
        t.append(" ")
        t.append(_bar(card.mem_frac), style=style)
        t.append(" ")
        t.append(_pct(card.util).rjust(4), style=style)
        t.append(" ")
        t.append(f"{_gib(card.mem_used)}/{_gib(card.mem_total)}".rjust(9), style="")
        t.append(" ")
        temp = f"{card.temp:.0f}C" if card.temp is not None else DASH
        power = f"{card.power:.0f}W" if card.power is not None else DASH
        t.append(temp.rjust(4), style="dim")
        t.append(" ")
        t.append(power.rjust(5), style="dim")
        if card.procs:
            names = sorted({p.name for p in card.procs})
            t.append(f"  {len(card.procs)}p ", style="bold")
            t.append(",".join(names)[:22], style="")
        elif state == "residual":
            t.append("  memory held, no process", style="dark_orange")
        t.append("\n")
    return t


def _bar(frac: float | None, width: int = 10) -> str:
    if frac is None:
        return DASH * width
    filled = max(0, min(width, round(frac * width)))
    # Half-block for the last partial cell: a 10-cell bar then shows 5% steps.
    return "█" * filled + "░" * (width - filled)


# --- detail block -------------------------------------------------------------


def detail_view(view: HostView | None, now: datetime | None = None) -> Text:
    """The selected host's health: node facts, holders, and the top processes."""
    t = Text()
    if view is None:
        return t
    t.append("VERDICT  ", style="dim")
    t.append(view.verdict.upper(), style=VERDICT_STYLE.get(view.verdict, ""))
    t.append("   ")
    if view.verdict == FREE and view.total:
        t.append("all cards available", style="green")
    elif view.verdict == "partial":
        t.append(f"{view.free_cards} of {view.total} cards available", style="yellow")
    elif view.verdict == RESIDUAL:
        t.append("no compute, but memory is held on every card", style="dark_orange")
    elif view.verdict == "busy":
        t.append("every card is in use", style="red")
    elif view.verdict == "unknown":
        reason = (view.accel.error if view.accel else None) or view.error or "no reading yet"
        t.append(reason[:70], style="magenta")
    elif view.verdict == "down":
        t.append((view.error or "unreachable")[:70], style="red")
    t.append("\n")

    # The address sits directly under the verdict because it is what you act on: copy
    # it into ssh, ping, or curl. Dashed when the inventory has no address for the host.
    t.append("ADDRESS  ", style="dim")
    t.append(view.address or "—", style="bold")
    if view.group:
        t.append(f"   group {view.group}", style="dim")
    t.append("\n")

    n = view.node
    t.append("NODE     ", style="dim")
    t.append(f"{n.cpus or DASH}c", style="")
    t.append("  load ", style="dim")
    load = f"{n.load1:.1f}" if n.load1 is not None else DASH
    if n.load5 is not None:
        load += f"/{n.load5:.1f}"
    t.append(load, style=_load_style(n.load_per_cpu))
    per_core = _pct(None if n.load_per_cpu is None else n.load_per_cpu * 100)
    t.append(f" ({per_core} per core)", style="dim")
    t.append("  mem ", style="dim")
    t.append(f"{_pct(n.mem_pct)}", style=_pct_style(n.mem_pct))
    t.append(f" of {_gib(n.mem_total_mib)}", style="dim")
    t.append("  cpu ", style="dim")
    t.append(_pct(n.cpu_pct), style="")
    if n.disk_pct is not None:
        t.append("  disk ", style="dim")
        t.append(_pct(n.disk_pct), style=_pct_style(n.disk_pct))
    t.append("\n")

    t.append("HOLDERS  ", style="dim")
    users = view.users
    logins = sorted(set(view.node.logins))
    if users:
        t.append(", ".join(users)[:60], style="")
    elif logins:
        t.append("no card holders; logged in: " + ", ".join(logins)[:40], style="dim")
    else:
        t.append("none detected", style="dim")
    t.append("\n")

    if n.top:
        t.append("TOP CPU  ", style="dim")
        for i, proc in enumerate(n.top[:3]):
            if i:
                t.append("   ", style="dim")
            t.append(f"{proc['cpu']:.0f}% ", style=_pct_style(proc["cpu"]))
            t.append(proc["user"], style="dim")
            t.append(f" {proc['cmd'][:44]}", style="")
        t.append("\n")
    if view.cards:
        t.append("CARDS    ", style="dim")
        t.append(card_legend(view))
        t.append("\n")
    t.append("SEEN     ", style="dim")
    t.append(f"{_age(view, now)} ago", style="dim")
    if view.stale:
        t.append("  (stale: this host was not polled last round)", style="yellow")
    t.append(f"   tags: {', '.join(view.tags) or DASH}", style="dim")
    return t


def card_legend(view: HostView) -> Text:
    t = Text()
    for card in view.cards:
        state = card_state(card)
        t.append(f"{card.index}", style=CARD_STYLE[state])
        t.append(" ")
    return t


def _load_style(lp: float | None) -> str:
    if lp is None:
        return ""
    if lp >= 1.0:
        return "bold red"
    if lp >= 0.75:
        return "yellow"
    return "green"


def _pct_style(p: float | None) -> str:
    if p is None:
        return ""
    if p >= 90:
        return "bold red"
    if p >= 70:
        return "yellow"
    return ""
