"""Turning ``/proc`` counters into a CPU utilisation percentage.

``/proc/stat``'s first ``cpu`` line is a set of *cumulative* jiffy counters since
boot; a single reading says nothing about how busy the CPU is now. The share of time
spent not-idle between two readings does, and that is what :class:`CpuSampler`
keeps — one previous reading per host.

Two readings are never sampled the way naive code does it (``sleep 1; cat /proc/stat``
on the remote host): that adds a second to every poll round *and* is exactly the kind
of command a bastion's audit rules exist to catch. The delta between successive polls
is free, and at a 5s refresh it is a better window than 1s would be.

A reading with no previous sample is ``None``, not 0: "not measured yet" and "idle"
are different answers, and the fleet view shows them differently.
"""

from __future__ import annotations

from dataclasses import dataclass

# Instrumentation (including iowait as idle) follows the convention of top/htop.
_IDLE_FIELDS = frozenset({"steal"})  # steal time is *not* our CPU being busy


@dataclass(frozen=True)
class CpuSample:
    """One ``/proc/stat`` cpu line plus the wall-clock second it was taken."""

    total: int
    idle: int
    stamp: float

    @classmethod
    def parse_line(cls, line: str, *, stamp: float) -> CpuSample | None:
        """Parse ``cpu user nice system idle iowait irq softirq steal ...``.

        Returns ``None`` for anything that is not the aggregate ``cpu`` line or does
        not carry at least the four guaranteed fields — a vendor kernel with a
        reordered or truncated line degrades to "unmeasured" rather than to a wrong
        number.
        """
        parts = line.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        try:
            values = [int(v) for v in parts[1:]]
        except ValueError:
            return None
        # user nice system idle iowait ; the remainder (irq softirq steal guest...)
        # is counted as busy except for guest time, which would be double-counted.
        total = sum(values[:4]) + sum(values[5:8]) if len(values) > 4 else sum(values[:4])
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return cls(total=total, idle=idle, stamp=stamp)

    def busy_pct_since(self, prev: CpuSample | None) -> float | None:
        """Share of non-idle CPU time between ``prev`` and ``self``, in percent."""
        if prev is None:
            return None
        d_total = self.total - prev.total
        d_idle = self.idle - prev.idle
        if d_total <= 0:
            # Equal or (after a reboot/counter reset) smaller: no usable window.
            return None
        busy = d_total - d_idle
        if busy < 0:
            return None
        # Round to one decimal: the column is 5 wide and a spurious 0.04% is noise.
        return round(min(100.0, busy * 100.0 / d_total), 1)


def load_per_cpu(load1: float | None, cpus: int | None) -> float | None:
    """1-minute load average normalised by core count — the fleet-level busy signal.

    On a shared cluster node this is the most honest "is anyone using this" answer
    that costs nothing to collect: unlike a card's utilisation it sees CPU-only jobs
    too, and unlike an occupied-card count it reflects *how much* of the node is
    taken. ``None`` when either input is unknown.
    """
    if load1 is None or not cpus:
        return None
    return load1 / cpus
