"""Plain-text rendering: the ``--once`` table and the shared formatting helpers.

The TUI and this module must agree on what a value *means* — ``—`` for "not
reported", ``partial`` only when some cards are genuinely available — so the
formatting lives here and both call into it.
"""

from __future__ import annotations

from .model import FREE, RESIDUAL, FleetView, HostView

#: Shown wherever a value was not reported. Deliberately not ``0``: a monitor that
#: prints 0% memory for a card it could not read is lying, and this is the one lie
#: that gets work scheduled onto a busy machine.
DASH = "—"


def fmt_pct(value: float | None, width: int = 5) -> str:
    return DASH.rjust(width) if value is None else f"{value:.0f}%".rjust(width)


def fmt_gib(mib: float | None) -> str:
    if mib is None:
        return DASH
    return f"{mib / 1024:.0f}G"


def fmt_mib_pair(used: float | None, total: float | None) -> str:
    if used is None or total is None:
        return DASH
    return f"{used / 1024:.0f}/{total / 1024:.0f}G"


def fmt_uptime(seconds: float | None) -> str:
    if seconds is None:
        return DASH
    s = int(seconds)
    days, rem = divmod(s, 86400)
    hours = rem // 3600
    if days:
        return f"{days}d{hours}h"
    minutes = (rem % 3600) // 60
    return f"{hours}h{minutes}m" if hours else f"{minutes}m"


def card_cells(view: HostView) -> str:
    """The one-cell-per-card strip: ``F`` free, ``B`` busy, ``?`` unknown, ``~`` residual.

    This is the fleet view's most valuable column — at 8 cards per host it shows more
    state per character than nvitop's bars do per line, and a glance tells you which
    of a node's cards are actually obtainable.
    """
    if not view.cards:
        return ""
    return "".join(
        "~" if c.residual else ("F" if c.busy is False else ("?" if c.busy is None else "B"))
        for c in view.cards
    )


def verdict_cell(view: HostView) -> str:
    return {
        FREE: "FREE",
        "partial": "PARTIAL",
        "busy": "BUSY",
        RESIDUAL: "RESIDUAL",
        "unknown": "UNKNOWN",
        "cpu": "CPU",
        "down": "DOWN",
    }.get(view.verdict, view.verdict.upper())


def _users_cell(view: HostView) -> str:
    """Who is on the host: accelerator holders first, else whoever is logged in."""
    accel = ",".join(view.users)
    logins = ",".join(sorted(set(view.node.logins)))
    return (accel or logins or "-")[:17]


_COLUMNS = [
    ("HOST", 26, lambda v: v.name),
    # The address, not the inventory key: `ssh`/`ping` need something that resolves.
    ("ADDRESS", 15, lambda v: v.address or DASH),
    ("PLATFORM", 14, lambda v: v.family_label),
    ("VERDICT", 10, lambda v: verdict_cell(v)),
    ("CARDS", 8, lambda v: f"{v.free_cards}/{v.total}" if v.total else DASH),
    ("MAP", 17, lambda v: card_cells(v) or DASH),
    (
        "LOAD",
        7,
        lambda v: fmt_pct(None if v.node.load_per_cpu is None else v.node.load_per_cpu * 100, 4),
    ),  # noqa: E501
    ("MEM", 5, lambda v: fmt_pct(v.node.mem_pct, 4)),
    ("CPU", 5, lambda v: fmt_pct(v.node.cpu_pct, 4)),
    ("UP", 9, lambda v: fmt_uptime(v.node.uptime_s)),
    ("USERS", 18, _users_cell),
]


def render_table(view: FleetView, *, color: bool = False, group: bool = True) -> str:
    """The ``--once`` table: one host per line, free hosts first.

    No box drawing: this output is meant to be grepped and awk'd as much as read, and
    a fixed-width layout with aligned columns gives both.
    """
    rows = sorted(view.hosts, key=lambda h: h.sort_key)
    head = "".join(
        name.ljust(w) if i == 0 else name.rjust(w) for i, (name, w, _) in enumerate(_COLUMNS)
    )
    lines = [head, "-" * len(head)]
    current_group = None
    for host in rows:
        if group and host.group and host.group != current_group:
            current_group = host.group
            lines.append(f"[{host.group}]")
        cells = []
        for i, (_name, w, fn) in enumerate(_COLUMNS):
            text = str(fn(host))
            if len(text) > w:
                text = text[: w - 1] + "~" if w > 2 else text[:w]
            cells.append(text.ljust(w) if i == 0 else text.rjust(w))
        line = "".join(cells)
        if color:
            line = _colour(line, host)
        lines.append(line)
    lines.append("")
    lines.append(_summary(view))
    lines.append(
        f"map: F free  B busy  ~ residual (memory held, idle)  ? unknown   {DASH} not reported"
    )
    return "\n".join(lines)


def _summary(view: FleetView) -> str:
    fams = "  ".join(f"{k} {v}" for k, v in sorted(view.families.items()))
    return (
        f"{len(view.hosts)} hosts, {view.reachable} reachable | "
        f"cards {view.free_cards} free / {view.busy_cards} busy / {view.unknown_cards} unknown "
        f"of {view.total_cards} | {len(view.free_hosts)} fully free{(' | ' + fams) if fams else ''}"
    )


def _colour(line: str, host: HostView) -> str:
    code = {
        FREE: "32",  # green
        "partial": "33",  # yellow
        "busy": "31",  # red
        RESIDUAL: "91",  # dark orange: idle, but its memory is not available
        "unknown": "35",  # magenta: we do not know, do not pretend
        "cpu": "36",  # cyan
        "down": "90",  # bright black
    }.get(host.verdict, "0")
    return f"\x1b[{code}m{line}\x1b[0m"
