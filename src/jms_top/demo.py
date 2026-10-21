"""Demo mode: the real UI, the real parsers, no bastion.

This replays a **recorded snapshot of the fleet** (``tests/fixtures/demo/fleet.json``,
captured 2026-10-21 from the 22-machine inventory) through the same ``parse_reply`` →
``HostView`` → TUI path as live data. Two reasons it exists:

* the interface can be looked at, screenshotted and reviewed without a JumpServer
  account — which is also how the docs get their pictures;
* the tests drive the whole app against it, so they exercise the real render path
  rather than a mock of it.

The one thing that is synthesised is the passage of time: the recorded CPU counters
are from capture day, so they are advanced by the wall time since the previous
reading, at a per-host utilisation that depends only on the host name (so a
screenshot is reproducible). Every other number — cards, memory, temperatures,
processes — is exactly what the cluster reported.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

from jms_bridge.config import BastionConfig
from jms_bridge.inventory import Host, Inventory
from jms_bridge.ssh import ExecResult

from .monitor import FleetMonitor
from .probe import SENTINEL
from .tui import JmsTopApp

DEMO_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "demo"
SNAPSHOT = DEMO_DIR / "fleet.json"

_COUNTERS = re.compile(r"(?m)^(\d+ \d+)$")


class ReplayRunner:
    """An ``SSHRunner`` stand-in serving recorded replies."""

    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        self.calls = 0
        self.closed = False
        #: host -> (monotonic time of the previous reading, its total jiffies)
        self._last: dict[str, tuple[float | None, int | None]] = {}

    async def exec_on(self, host, script, *, timeout=60.0, workdir=True, env_setup=True):  # noqa: ANN001
        self.calls += 1
        entry = self.snapshot["hosts"].get(host.name, {})
        if "##" not in script:  # the liveness sentinel
            ok = entry.get("reachable", True)
            return ExecResult(host.name, 0, f"{SENTINEL}\n" if ok else "", "")
        if entry.get("reachable") is False:
            return ExecResult(host.name, 0, entry.get("error", "unreachable"), "")
        kind = "cpu" if "##CPU" in script and "##NVIDIA" not in script else "fleet"
        body = self._advance(entry.get(kind, ""), host.name)
        return ExecResult(host.name, 0, f"{SENTINEL}\n{body}", "")

    def _advance(self, body: str, host: str) -> str:
        """Push the recorded CPU counters forward so utilisation means something."""
        prev_time, prev_total = self._last.get(host, (None, None))
        now = time.monotonic()
        match = _COUNTERS.search(body)
        if match is None:
            self._last[host] = (now, None)
            return body
        total, idle = (int(x) for x in match.group(1).split())
        if prev_time is not None and prev_total is not None:
            elapsed = max(1.0, now - prev_time)
            cpus = max(1, os.cpu_count() or 8)
            busy = (sum(ord(c) for c in host) % 86) / 100.0
            jiffies = int(elapsed * 100 * cpus)  # USER_HZ = 100 on Linux
            total += jiffies
            idle += jiffies - int(jiffies * busy)
        self._last[host] = (now, total)
        return body[: match.start(1)] + f"{total} {idle}" + body[match.end(1) :]

    async def close(self) -> None:
        self.closed = True


def demo_inventory() -> Inventory:
    snapshot = _snapshot()
    hosts = {
        name: Host(
            name=name,
            platform=row.get("platform", "generic"),
            tags=list(row.get("tags", [])),
            address=row.get("address"),
        )
        for name, row in snapshot["hosts"].items()
    }
    return Inventory(
        hosts=hosts, groups={k: list(v) for k, v in snapshot.get("groups", {}).items()}
    )


def _snapshot() -> dict:
    if not SNAPSHOT.exists():  # pragma: no cover - ships with the tests
        raise FileNotFoundError(f"demo snapshot missing: {SNAPSHOT}")
    return json.loads(SNAPSHOT.read_text())


def demo_monitor(*, interval: float = 3.0, full_interval: float | None = None) -> FleetMonitor:
    """A :class:`FleetMonitor` wired to the recorded fleet instead of SSH."""
    return FleetMonitor(
        cfg=BastionConfig(url="https://demo.invalid"),
        inventory=demo_inventory(),
        runner=ReplayRunner(_snapshot()),
        interval=interval,
        full_interval=full_interval,
    )


def run_demo(*, interval: float = 3.0, once: bool = False, as_json: bool = False) -> int:
    monitor = demo_monitor(interval=interval)
    if once or as_json:
        from .cli import run_monitor_once

        return asyncio.run(run_monitor_once(monitor, as_json=as_json))
    JmsTopApp(monitor, demo=True).run()
    return 0
