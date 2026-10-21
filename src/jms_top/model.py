"""What one host looks like at one instant, and whether it is worth taking.

This is the model the whole UI reads. Two independent rolls-up matter here:

* **Card roll-up** — ``free / busy / unknown`` chips, and the *usable* count, which
  excludes unknown ones. A host is "free" only when its whole card set is free; a
  partially free host is reported with the count that is actually available (the
  question a user asks is "can I get 4 cards", not "is it idle").
* **Node roll-up** — the CPU side: load per core, memory, logins, used disk. On a
  shared cluster node this is the honest signal for CPU-only jobs, and it is also
  what tells you whether a card-free host is really yours to take: eight idle cards
  next to a load of 600 belong to somebody running a CPU-side data pipeline.

Both are kept side by side rather than merged into one score, because the answer
depends on what you are about to run. The one place they *are* combined is
:attr:`HostView.verdict`, which is deliberately the conservative reading: any
trouble at all lands in ``busy`` before ``free``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from .cpu import CpuSample, load_per_cpu
from .families import AccelResult, Card

#: Verdict labels, in the order the fleet table groups them.
FREE = "free"
PARTIAL = "partial"
BUSY = "busy"
RESIDUAL = "residual"
UNKNOWN = "unknown"
DOWN = "down"


@dataclass
class NodeStats:
    """The CPU half of a host, as ``/proc`` and ``ps`` report it."""

    hostname: str | None = None
    cpus: int | None = None
    cpus_online: int | None = None
    load1: float | None = None
    load5: float | None = None
    load15: float | None = None
    mem_total_mib: float | None = None
    mem_avail_mib: float | None = None
    mem_pct: float | None = None
    swap_total_mib: float | None = None
    uptime_s: float | None = None
    boot_time: float | None = None
    cpu_pct: float | None = None
    cpu_sample: CpuSample | None = None
    #: ``[(user, sessions_or_count)]`` — who is logged in right now.
    logins: list[str] = field(default_factory=list)
    disk_pct: float | None = None
    disk_free_gib: float | None = None
    #: ``[{pid, user, elapsed, cpu, mem, cmd}]`` — the heaviest processes, top-first.
    top: list[dict] = field(default_factory=list)

    @property
    def load_per_cpu(self) -> float | None:
        return load_per_cpu(self.load1, self.cpus)

    def to_dict(self) -> dict:
        d = {
            "hostname": self.hostname,
            "cpus": self.cpus,
            "load": [self.load1, self.load5, self.load15],
            "load_per_cpu": self.load_per_cpu,
            "mem_pct": self.mem_pct,
            "mem_avail_mib": self.mem_avail_mib,
            "mem_total_mib": self.mem_total_mib,
            "cpu_pct": self.cpu_pct,
            "uptime_s": self.uptime_s,
            "logins": self.logins,
            "disk_pct": self.disk_pct,
            "top": self.top,
        }
        return d


@dataclass
class HostView:
    """One inventory host, as seen at ``updated_at``."""

    name: str
    platform: str = "generic"
    tags: list[str] = field(default_factory=list)
    address: str | None = None
    group: str | None = None
    inventory_platform: str | None = None

    # --- liveness -------------------------------------------------------------
    #: ``True`` reached, ``False`` refused/unreachable, ``None`` not yet probed.
    reachable: bool | None = None
    error: str | None = None
    latency_s: float | None = None

    # --- payload --------------------------------------------------------------
    accel: AccelResult | None = None
    node: NodeStats = field(default_factory=NodeStats)
    #: The raw reply, kept so the detail pane can show what the parser saw.
    raw: str = ""
    updated_at: datetime | None = None
    #: Set when a poll has started but not finished (the row shows a stale marker).
    stale: bool = False

    # --- derived --------------------------------------------------------------

    @property
    def cards(self) -> list[Card]:
        return self.accel.cards if self.accel else []

    @property
    def total(self) -> int:
        return len(self.cards)

    @property
    def vram_used(self) -> float | None:
        """MiB held across every card, or None if no card reported memory."""
        vals = [c.mem_used for c in self.cards if c.mem_used is not None]
        return sum(vals) if vals else None

    @property
    def vram_total(self) -> float | None:
        """MiB of accelerator memory on the host, or None if unreported.

        Shown in the table so the size of a machine is visible at a glance: a host
        with 8 x 144 GB and one with 8 x 24 GB look identical when only the card
        count is displayed.
        """
        vals = [c.mem_total for c in self.cards if c.mem_total]
        return sum(vals) if vals else None

    @property
    def vram_per_card(self) -> float | None:
        """MiB per card — the number people quote ("an 80G card")."""
        vals = [c.mem_total for c in self.cards if c.mem_total]
        return max(vals) if vals else None

    @property
    def free_cards(self) -> int:
        return sum(1 for c in self.cards if c.busy is False)

    @property
    def busy_cards(self) -> int:
        return sum(1 for c in self.cards if c.busy is True)

    @property
    def unknown_cards(self) -> int:
        return sum(1 for c in self.cards if c.busy is None)

    @property
    def residual_cards(self) -> int:
        return sum(1 for c in self.cards if c.residual)

    @property
    def family(self) -> str:
        return self.accel.family if self.accel else "generic"

    @property
    def family_label(self) -> str:
        return self.accel.label if self.accel else "—"

    @property
    def card_name(self) -> str:
        names = {c.name for c in self.cards if c.name}
        if not names:
            return ""
        first = sorted(names)[0]
        return first if len(names) == 1 else f"{first} +{len(names) - 1}"

    @property
    def procs(self) -> list[tuple[Card, object]]:
        """Every accelerator process on the host, as ``(card, proc)`` pairs."""
        return [(c, p) for c in self.cards for p in c.procs]

    @property
    def users(self) -> list[str]:
        """Users holding an accelerator on this host, by pid (else just the name)."""
        seen: set[str] = set()
        out = []
        for p in (proc for _c, proc in self.procs):
            label = p.user or f"pid {p.pid}"
            if label not in seen:
                seen.add(label)
                out.append(label)
        return out

    @property
    def residual_only(self) -> bool:
        """Every card is held by memory alone: no compute, a leaked/preloaded stack.

        This is the state a "free card" search most wants to know about, because the
        cards are not free *and* nobody is using them — asking is cheap, guessing is
        expensive.
        """
        return bool(self.total) and self.residual_cards == self.total

    @property
    def verdict(self) -> str:
        """One word for the whole host: free / partial / busy / unknown / down.

        ``residual`` is busier than ``busy`` in practice, so it is ranked beside it:
        its cards cannot be used without someone freeing memory they still hold.
        """
        if self.reachable is False:
            return DOWN
        if self.accel is not None and self.accel.total:
            free, unknown = self.free_cards, self.unknown_cards
            if free == self.total:
                # All cards free — but a loaded CPU side means the node is not idle.
                lpc = self.node.load_per_cpu
                if lpc is not None and lpc >= 0.75:
                    return PARTIAL
                return FREE
            if free > 0:
                return PARTIAL
            if unknown == self.total:
                return UNKNOWN
            return RESIDUAL if self.residual_only else BUSY
        if self.reachable is True and self.accel is not None:
            # Reachable, no accelerators (CPU-only, or an unparsed vendor tool).
            return UNKNOWN if self.accel.error else "cpu"
        return UNKNOWN

    @property
    def pending(self) -> bool:
        """Has not answered yet — not "answered with something unknown".

        The two look identical in the data (``reachable is None`` and no reading), but
        they must not sort the same: a pending host is one the scan has not reached, and
        it is about to become FREE, BUSY or DOWN. Sorting it with the UNKNOWN band would
        make rows jump *out* of the middle of the table as the scan advanced.
        """
        return self.reachable is None and self.accel is None and self.updated_at is None

    def sort_key(self, *, stable: bool = False) -> tuple:
        """Free hosts first, then partial; within a band, most free cards first.

        ``stable=True`` keeps the inventory's order instead, which is what the table
        uses *while a round is in flight*: a host that reports FREE halfway through
        would otherwise jump to the top, past whatever the cursor was pointing at.
        The availability order returns when the round is over — that is when a re-sort
        is wanted, because the whole point of the tool is to bring the free machines up.

        Hosts that have not answered yet go last, below DOWN: they are not a verdict,
        and putting them anywhere in the order means every one of them jumps rows when
        its answer arrives.
        """
        if stable:
            return (0, 0, 0, self.name)
        if self.pending:
            return (9, 0, 0, self.name)
        order = {FREE: 0, PARTIAL: 1, "cpu": 2, BUSY: 3, RESIDUAL: 4, UNKNOWN: 5, DOWN: 6}
        return (
            order.get(self.verdict, 8),
            -self.free_cards,
            0,
            self.name,
        )

    def with_node(self, node: NodeStats) -> HostView:
        """A copy carrying a new CPU reading, keeping the accelerator data.

        Used by the fast, cheap pass: CPU/load/memory are refreshed every round while
        the card tables are re-read on the slower one. A copy rather than in-place
        mutation so a reader holding the old view never sees half-updated numbers.
        """
        return replace(self, node=node)

    def to_dict(self) -> dict:
        return {
            "host": self.name,
            "platform": self.platform,
            "tags": self.tags,
            "address": self.address,
            "reachable": self.reachable,
            "error": self.error,
            "latency_s": self.latency_s,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "verdict": self.verdict,
            "accel": {
                "family": self.family,
                "label": self.family_label,
                "card_name": self.card_name,
                "total": self.total,
                "free": self.free_cards,
                "busy": self.busy_cards,
                "unknown": self.unknown_cards,
                "residual": self.residual_cards,
                "version": self.accel.version if self.accel else None,
                "error": self.accel.error if self.accel else None,
                "cards": [c.to_dict() for c in self.cards],
            },
            "node": self.node.to_dict(),
        }


@dataclass
class FleetView:
    """Every host, plus the fleet-level roll-up the header shows."""

    hosts: list[HostView] = field(default_factory=list)
    round_started: datetime | None = None
    round_finished: datetime | None = None
    next_refresh: datetime | None = None
    paused: bool = False
    interval: float = 3.0
    #: ``(done, total)`` while a full round is in flight, else ``None``. Shown in the
    #: footer so a slow round over a large inventory reads as progress, not as a hang.
    scan_progress: tuple[int, int] | None = None
    #: True while a full round is in flight. The table keeps the order the round
    #: started with for its duration: a host that reports FREE halfway through would
    #: otherwise jump to the top, past whatever the cursor was pointing at.
    scanning: bool = False
    #: How often a host with free capacity gets its card tables re-read.
    full_interval: float = 6.0
    #: One line per completed round, newest last — the activity log.
    log: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> HostView | None:
        for h in self.hosts:
            if h.name == name:
                return h
        return None

    @property
    def reachable(self) -> int:
        return sum(1 for h in self.hosts if h.reachable is True)

    @property
    def free_hosts(self) -> list[HostView]:
        return [h for h in self.hosts if h.verdict == FREE]

    @property
    def free_cards(self) -> int:
        return sum(h.free_cards for h in self.hosts)

    @property
    def total_cards(self) -> int:
        return sum(h.total for h in self.hosts)

    @property
    def busy_cards(self) -> int:
        return sum(h.busy_cards for h in self.hosts)

    @property
    def unknown_cards(self) -> int:
        return sum(h.unknown_cards for h in self.hosts)

    @property
    def residual_cards(self) -> int:
        return sum(h.residual_cards for h in self.hosts)

    @property
    def families(self) -> dict[str, int]:
        """``{family_label: card count}`` — the one-line fleet composition."""
        out: dict[str, int] = {}
        for h in self.hosts:
            if h.total:
                out[h.family_label] = out.get(h.family_label, 0) + h.total
        return out
