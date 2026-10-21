"""The poll loop: keep a :class:`FleetView` up to date without blocking the UI.

:class:`FleetMonitor` owns everything async. It is deliberately *not* a Textual
object: the same class backs ``jms-top --once``, the TUI, and the tests (which drive
it against a fake runner), so a bug in refresh behaviour is reproducible without a
terminal.

The scheduling is two-speed, mirroring what makes nvitop feel live:

* **Fast tick — CPU only.** Load, cores, memory, logins: a handful of file reads, no
  ``ps``, no disk, no vendor tool. It runs every ``interval`` seconds (3s by default)
  and costs one round trip per host.
* **Full tick — every card table.** The vendor SMI, ``ps``, ``df``. It runs per host
  on its own timer: every ``full_interval`` seconds while that host has free capacity
  to show, and ``IDLE_FACTOR`` times slower once nothing on it is free — a busy card
  stays busy, and there is nothing to watch. A host that was free and is now busy
  gets a full tick on the *next* tick regardless of its timer: that transition is the
  one that matters for "can I take this machine".

Both write into the same :class:`HostView`: the fast tick replaces the node stats on
a *copy* of the view (see :meth:`HostView.with_node`), so a reader never sees a
half-updated host, and card tables shown between full ticks are at most one full
interval old — which the age column makes visible.

Four more decisions, and why:

* **Ping first.** ``ping`` is one guarded ``echo`` per host: on a warm ControlMaster it
  answers in milliseconds, so the fleet shows *which hosts are up* before the first
  full scan instead of after it.
* **One fan-out cap for both rounds.** The bastion limits concurrent connections per
  account, so every round goes through one :class:`asyncio.Semaphore` — an overlapping
  full tick then cannot double the fleet's connection count.
* **Per-host failure backoff.** A host that keeps failing is full-polled less often,
  and a positive answer from either round clears the count, so one unreachable asset
  cannot dominate every tick.
* **Stale-while-revalidate.** A tick that fails on every host keeps the previous
  reading on screen (marked stale by age) rather than blanking the table.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from .config import Bastion, ConfigError, load_settings
from .cpu import CpuSample, load_per_cpu
from .inventory import Host, Inventory, InventoryError
from .model import FREE, FleetView, HostView, NodeStats
from .probe import PING_SCRIPT, SENTINEL, ProbeLimits, build_script, with_sentinel
from .scan import parse_reply, reached
from .ssh import ExecResult, SSHRunner, failure_reason

#: How many times slower a full tick runs on a host with nothing free.
IDLE_FACTOR = 3.0

#: How many hosts one full round probes at once. Low on purpose: the point is a screen
#: that fills in visibly, not the shortest possible round — and every concurrent command
#: holds a bastion connection the account has a limit on. Raise it for a large fleet.
SCAN_BATCH = 4
#: Consecutive failures after which a host is only polled every Nth round.
BACKOFF_ROUNDS = 4
#: Consecutive fast-tick failures that promote a host to a full tick: the cheap probe
#: cannot tell "unreachable" from "reachable, but /proc is odd".
SUSPECT_AFTER = 3


class MonitorError(RuntimeError):
    """Configuration or inventory problem — the message is meant for the user."""


class FleetMonitor:
    """Builds the host list from the inventory and polls it."""

    def __init__(
        self,
        *,
        targets: list[str] | None = None,
        interval: float = 3.0,
        full_interval: float | None = None,
        bastion: Bastion | None = None,
        inventory: Inventory | None = None,
        runner: SSHRunner | None = None,
        limits: ProbeLimits | None = None,
        nvidia_tile: bool = False,
        timeout: float = 45.0,
        log: Callable[[str], None] | None = None,
    ):
        try:
            if bastion is None:
                bastion = load_settings().bastion
            self.bastion = bastion
            self.inventory = inventory or Inventory.load()
        except (ConfigError, InventoryError) as exc:
            # One line of advice, not a traceback: these are first-run mistakes
            # (no config, no inventory), which every user hits exactly once.
            raise MonitorError(str(exc)) from exc
        self._owns_runner = runner is None
        self.runner = runner or SSHRunner(bastion, max_parallel=bastion.max_parallel)
        self.limits = limits or ProbeLimits()
        self.nvidia_tile = nvidia_tile
        self.timeout = timeout
        self.base_interval = max(0.5, float(interval))
        self.full_interval = max(self.base_interval, float(full_interval or self.base_interval * 2))
        self._log_sink = log

        selectors = targets or ["all"]
        try:
            hosts = self.inventory.resolve(selectors)
        except InventoryError as exc:
            raise MonitorError(str(exc)) from exc
        if not hosts:
            raise MonitorError(f"selector(s) {selectors!r} matched no hosts")

        self.hosts: list[Host] = sorted(hosts, key=lambda h: h.name)
        self.groups = _groups_of(self.inventory, self.hosts)
        self.view = FleetView(
            hosts=[self._blank(h) for h in self.hosts],
            interval=self.base_interval,
            full_interval=self.full_interval,
        )
        self._prev_cpu: dict[str, CpuSample] = {}
        self._failures: dict[str, int] = {}
        #: Consecutive cheap-tick failures per host, which promote it to a full tick.
        self._suspect: dict[str, int] = {}
        #: When each host last got a full tick (monotonic seconds).
        self._last_full: dict[str, float] = {}
        #: Progress of the round in flight, for the footer. ``_scan_total`` is 0 outside
        #: a full round.
        self._scanned = 0
        self._scan_total = 0
        #: Bumped whenever anything the UI shows changes; the UI diffs it instead of
        #: re-rendering on a timer.
        self.generation = 0
        self._round_no = 0
        self._running = False
        self._current: asyncio.Task | None = None
        self._pinged = False
        #: ``time.monotonic()`` of the last tick, for the countdown.
        self.last_tick: float = 0.0

    # --- construction ----------------------------------------------------------

    def _blank(self, host: Host) -> HostView:
        from .families import family_hint

        return HostView(
            name=host.name,
            platform=family_hint(host.platform, host.tags),
            inventory_platform=host.platform,
            tags=list(host.tags),
            address=host.address,
            group=self.groups.get(host.name),
        )

    def log(self, line: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{stamp}  {line}"
        self.view.log.append(entry)
        del self.view.log[:-200]
        if self._log_sink:
            self._log_sink(entry)

    def _touch(self) -> None:
        self.generation += 1

    # --- one round -------------------------------------------------------------

    async def ping_all(self) -> None:
        """Mark every host reachable/unreachable before the first full poll."""
        results = await asyncio.gather(
            *(self._run(h, PING_SCRIPT, timeout=min(self.timeout, 20.0)) for h in self.hosts)
        )
        up = 0
        for host, res in zip(self.hosts, results, strict=True):
            view = self.view.by_name(host.name)
            assert view is not None
            if isinstance(res, Exception):
                view.reachable, view.error = False, f"{type(res).__name__}: {res}"[:200]
            elif reached(res.stdout):
                view.reachable, view.error = True, None
                up += 1
            else:
                view.reachable = False
                view.error = failure_reason(res)[:200] or "no reply"
        self._pinged = True
        self._touch()
        down = len(self.hosts) - up
        suffix = f", {down} refused/down" if down else ""
        self.log(f"ping: {up}/{len(self.hosts)} hosts reachable{suffix}")

    def due_full(self, host: Host, now: float | None = None) -> bool:
        """Whether ``host`` is due a full tick now (cards, processes, disk)."""
        now = time.monotonic() if now is None else now
        if self._suspect.get(host.name, 0) >= SUSPECT_AFTER:
            return True
        last = self._last_full.get(host.name)
        if last is None:
            return True
        view = self.view.by_name(host.name)
        if view is None or view.reachable is False:
            return True
        budget = self.full_interval if view.free_cards > 0 else self.full_interval * IDLE_FACTOR
        return now - last >= budget

    async def poll(self, *, force: bool = False) -> None:
        """One **full** round over every host (the ``--once`` path and manual refresh)."""
        started = time.time()
        self.view.round_started = datetime.now(timezone.utc)
        self._round_no += 1
        round_no = self._round_no
        targets: list[Host] = []
        for host in self.hosts:
            fails = self._failures.get(host.name, 0)
            if force or fails < 3 or round_no % BACKOFF_ROUNDS == 0:
                targets.append(host)
        for host in self.hosts:
            view = self.view.by_name(host.name)
            if view is not None:
                view.stale = host not in targets

        await self._full(targets)

        self.view.round_finished = datetime.now(timezone.utc)
        elapsed = time.time() - started
        self.log(
            f"full round: {len(targets)} polled, {elapsed:.1f}s"
            + (
                f" ({len(self.hosts) - len(targets)} backing off)"
                if len(targets) < len(self.hosts)
                else ""
            )
        )

    def plan(self, now: float | None = None) -> tuple[list[Host], list[Host]]:
        """Split the fleet into ``(cheap targets, full targets)`` for the next tick."""
        now = time.monotonic() if now is None else now
        cheap: list[Host] = []
        full: list[Host] = []
        for host in self.hosts:
            (full if self.due_full(host, now) else cheap).append(host)
        return cheap, full

    async def tick(self) -> None:
        """One tick: cheap stats for everyone due, full reads for the hosts due.

        Both halves run concurrently under the shared connection cap, so a full tick
        on two hosts does not delay the cheap tick on the other twenty.
        """
        cheap, full = self.plan()
        tasks = []
        if cheap:
            tasks.append(self._cheap(cheap))
        if full:
            tasks.append(self._full(full))
        if tasks:
            await asyncio.gather(*tasks)
        self.last_tick = time.monotonic()
        self.view.next_refresh = datetime.now(timezone.utc) + timedelta(seconds=self.base_interval)

    async def _cheap(self, hosts: list[Host]) -> None:
        script = with_sentinel(build_script("cpu"))
        results = await asyncio.gather(
            *(self._run(h, script, timeout=min(self.timeout, 20.0)) for h in hosts)
        )
        changed = False
        stamp = datetime.now(timezone.utc).timestamp()
        for host, res in zip(hosts, results, strict=True):
            view = self.view.by_name(host.name)
            if view is None:
                continue
            if isinstance(res, Exception) or not reached(res.stdout):
                self._suspect[host.name] = self._suspect.get(host.name, 0) + 1
                continue
            self._suspect.pop(host.name, None)
            node = self._node_from(host, res.stdout, stamp)
            self._prev_cpu[host.name] = node.cpu_sample  # type: ignore[assignment]
            updated = view.with_node(node)
            updated.reachable, updated.error = True, None
            # A node that has become busy is worth a fresh card table before the next
            # scheduled one: "free" in the table is the claim this tool exists to make,
            # and the CPU side has just changed the grounds for it.
            if node_load_busy(node) and not _load_was_busy(view.node):
                self._last_full.pop(host.name, None)
            # Swap the whole view object: a reader holding the old one sees a
            # consistent (if older) host, never a half-updated one.
            self.view.hosts[self.view.hosts.index(view)] = updated
            if _signature(view) != _signature(updated):
                changed = True
        if changed:
            self._touch()

    async def _full(self, hosts: list[Host]) -> None:
        if not hosts:
            return
        self._scanned, self._scan_total = 0, len(hosts)
        self.view.scan_progress = (0, len(hosts))
        script = with_sentinel(
            build_script("fleet", limits=self.limits, nvidia_tile=self.nvidia_tile)
        )
        # Hosts go out in small batches, and the round repaints as each one answers.
        #
        # Pure serial is easy to watch but wastes the round: the latencies are ~1-12 s
        # each, so 77 machines take a couple of minutes, almost all of it waiting. A
        # full fan-out is fast but pointless to watch — nothing appears until the
        # slowest host lands, and then everything does. Batches of `SCAN_BATCH` keep
        # both properties: rows keep arriving in inventory order, and the round costs
        # roughly `ceil(n / SCAN_BATCH)` slowest-host waits instead of `n` of them.
        changed = False
        for start in range(0, len(hosts), SCAN_BATCH):
            batch = hosts[start : start + SCAN_BATCH]
            pending = {
                asyncio.ensure_future(self._run(h, script, timeout=self.timeout)): h for h in batch
            }
            # Explicit wait/FIRST_COMPLETED rather than `as_completed`: that helper yields
            # wrappers which (3.13+) no longer identify the future handed in, and on 3.14
            # they are coroutines with nothing to look up. Waiting for "one is done" keeps
            # the task -> host mapping trivially correct.
            while pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    host = pending.pop(task)
                    res = task.result()
                    view = self.view.by_name(host.name)
                    if view is None:
                        continue
                    before = _signature(view)
                    self._apply_full(host, view, res, stamp=datetime.now(timezone.utc).timestamp())
                    self._scanned += 1
                    self.view.scan_progress = (self._scanned, self._scan_total)
                    if _signature(view) != before:
                        changed = True
                    # Bumped per host so the TUI repaints mid-round: the app compares
                    # generations, so one bump at the end would put every row on screen
                    # at once — the thing this avoids.
                    self._touch()
        self._scan_total = 0  # round over: the footer stops showing progress
        self.view.scan_progress = None
        if changed:
            self._touch()

    async def _run(self, host: Host, script: str, *, timeout: float) -> ExecResult | Exception:
        """Run one script on one host; a failure is returned, never raised.

        The runner runs the script as-is: no ``env_setup``, no ``cd`` into a workdir.
        Looking at a host is not running a job on it, and the poll must succeed on a
        host whose dev environment is broken — which is exactly when you want the
        screen to still tell you the machine is up and idle.
        """
        try:
            return await self.runner.exec_on(host, script, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - one host's failure is that host's
            return exc

    def _node_from(self, host: Host, stdout: str, stamp: float) -> NodeStats:
        """Parse the node stats out of a cheap-tick reply (no accelerator section)."""
        body = stdout
        if body.startswith(f"{SENTINEL}\n"):
            body = body[len(SENTINEL) + 1 :]
        scratch = HostView(name=host.name)
        parse_reply(scratch, body, stamp=stamp)
        node = scratch.node
        prev = self._prev_cpu.get(host.name)
        if node.cpu_sample is not None:
            # Two counters and a delta: a percentage is only available from the second
            # reading onwards, and until then the column shows "—" rather than a
            # made-up zero.
            node.cpu_pct = node.cpu_sample.busy_pct_since(prev)
            self._prev_cpu[host.name] = node.cpu_sample
        return node

    def _apply_full(
        self, host: Host, view: HostView, res: ExecResult | Exception, *, stamp: float
    ) -> None:
        view.stale = False
        if isinstance(res, Exception):
            self._fail_host(host, view, f"{type(res).__name__}: {res}")
            return
        if not reached(res.stdout):
            self._fail_host(host, view, failure_reason(res) or "no reply")
            return
        self._failures.pop(host.name, None)
        self._suspect.pop(host.name, None)
        # The sentinel is the first line the far side prints; drop that one line only,
        # so the probe's most valuable content (``##SYS``) becomes the new first line.
        body = res.stdout
        if body.startswith(f"{SENTINEL}\n"):
            body = body[len(SENTINEL) + 1 :]
        # Start from a clean view: a card that disappeared must not survive as a stale
        # entry copied over from the previous reply.
        view.accel = None
        parse_reply(view, body, stamp=stamp)
        node: NodeStats = view.node
        prev = self._prev_cpu.get(host.name)
        if node.cpu_sample is not None:
            node.cpu_pct = node.cpu_sample.busy_pct_since(prev)
            self._prev_cpu[host.name] = node.cpu_sample
        view.reachable = True
        view.error = None
        self._last_full[host.name] = time.monotonic()

    def _fail_host(self, host: Host, view: HostView, reason: str) -> None:
        self._failures[host.name] = self._failures.get(host.name, 0) + 1
        view.reachable = False
        view.error = reason[:200]
        view.latency_s = None
        self._last_full.pop(host.name, None)

    # --- the loop --------------------------------------------------------------

    def start(self) -> None:
        """Start the background loop (the TUI calls this on mount)."""
        if self._current is None or self._current.done():
            self._running = True
            self._current = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._current is not None:
            self._current.cancel()
            try:
                await self._current
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._current = None

    def pause(self) -> None:
        self._running = False

    def resume(self) -> None:
        if self._current is None or self._current.done():
            self.start()
        else:
            self._running = True

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return not self._running

    async def refresh_now(self) -> None:
        """Manual refresh: a full round on everything, whatever the timers say."""
        await self.poll(force=True)

    async def _loop(self) -> None:
        self._touch()
        # Tick *immediately*, then sleep: waiting a whole interval before the first
        # round leaves the table blank for `interval` + (ping's own time) after mount,
        # which on a bastion is 10 s of empty screen. The schedule keeps to the same
        # wall-clock grid (`ceil(now / interval) * interval`) as the display countdown,
        # so "next tick in 2.1s" stays true even when a round overruns.
        while self._running:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive a bad tick
                self.log(f"tick failed: {type(exc).__name__}: {exc}"[:200])
            if not self._running:
                break
            now = time.monotonic()
            wait = math.ceil(now / self.base_interval) * self.base_interval - now
            await asyncio.sleep(max(0.2, wait))

    async def run_forever(self) -> None:
        """Ping once, then tick forever. Cancelling the task stops the loop."""
        await self.ping_all()
        self.start()
        assert self._current is not None
        await self._current

    async def close(self) -> None:
        await self.stop()
        if self._owns_runner:
            await self.runner.close()


def node_load_busy(node: NodeStats, threshold: float = 0.75) -> bool:
    """Is the CPU side loaded enough that the node is no longer idle?"""
    lpc = load_per_cpu(node.load1, node.cpus)
    return lpc is not None and lpc >= threshold


def _load_was_busy(node: NodeStats, threshold: float = 0.75) -> bool:
    lpc = load_per_cpu(node.load1, node.cpus)
    return lpc is not None and lpc >= threshold


def _signature(view: HostView) -> tuple:
    """Cheap change detector: did anything the table shows change at all?"""
    return (
        view.reachable,
        view.verdict,
        view.total,
        view.free_cards,
        view.busy_cards,
        tuple((c.index, c.util, c.mem_used, len(c.procs)) for c in view.cards),
        view.node.hostname,
        view.node.cpus,
        view.node.load1,
        view.node.mem_pct,
        view.node.cpu_pct,
        tuple(view.node.logins),
        tuple(view.node.top),
        tuple(view.users),
        view.node.disk_pct,
    )


def _groups_of(inventory: Inventory, hosts: list[Host]) -> dict[str, str]:
    """Each host's first group name, for the group column and the group filter."""
    out: dict[str, str] = {}
    wanted = {h.name for h in hosts}
    for group, members in inventory.groups.items():
        for name in members:
            if name in wanted:
                out.setdefault(name, group)
    return out


__all__ = ["FleetMonitor", "MonitorError", "FREE"]
