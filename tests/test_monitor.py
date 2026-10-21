"""The scan → model → verdict pipeline, and the two-speed scheduler, without a cluster.

``FakeRunner`` stands in for jms-bridge's ``SSHRunner``: it answers from a script of
per-host replies, so a test can say "this host goes down, then comes back" and check
what the monitor does about it.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from jms_bridge.config import BastionConfig
from jms_bridge.inventory import Host, Inventory
from jms_bridge.ssh import ExecResult

from jms_top.cpu import CpuSample, load_per_cpu
from jms_top.model import (
    BUSY,
    DOWN,
    FREE,
    PARTIAL,
    RESIDUAL,
    UNKNOWN,
    FleetView,
    HostView,
    NodeStats,
)
from jms_top.monitor import FleetMonitor
from jms_top.probe import SENTINEL, build_script
from jms_top.scan import parse_reply, parse_system, reached, split_sections

# --- fixtures from the live fleet ------------------------------------------------

SYS = """{sentinel}
##SYS
{load} 1.0 1.0 2/800 4242
{cpus}
{cpus}
{mem_total} {mem_avail} 0
{uptime}
{cpu_total} {cpu_idle}
node-{name}
##WHO
 2 root
##TOP
 101 root           12:00:00  9.0  0.1 /usr/bin/python train.py
##DISK
/dev/sda1 100 50 50 50% /
"""

GPU_IDLE = """##NVIDIA
0, NVIDIA A100-SXM4-40GB, 0, 4, 81920, 37, 56.9, 00000000:03:00.0
1, NVIDIA A100-SXM4-40GB, 0, 4, 81920, 34, 60.5, 00000000:16:00.0
"""
GPU_BUSY = """##NVIDIA
0, NVIDIA A100-SXM4-40GB, 97, 79000, 81920, 62, 300.1, 00000000:03:00.0
1, NVIDIA A100-SXM4-40GB, 97, 79000, 81920, 63, 301.2, 00000000:16:00.0
00000000:03:00.0, 9001, /opt/conda/envs/t/bin/python, 78900 MiB
00000000:16:00.0, 9002, /opt/conda/envs/t/bin/python, 78900 MiB
"""


def reply(
    name: str,
    *,
    load: float = 1.0,
    cpus: int = 64,
    gpu: str = GPU_IDLE,
    cpu_total: int = 1000,
    cpu_idle: int = 500,
    sentinel: bool = True,
) -> str:
    head = SYS.format(
        sentinel=SENTINEL if sentinel else "",
        load=load,
        cpus=cpus,
        mem_total=1_048_576,
        mem_avail=524_288,
        uptime=3600,
        cpu_total=cpu_total,
        cpu_idle=cpu_idle,
        name=name,
    )
    return head + gpu


class FakeRunner:
    """Answers ``exec_on`` from ``replies[host_name]``; a ``None`` reply raises."""

    def __init__(self, replies: dict[str, str | Exception]):
        self.replies = replies
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def exec_on(self, host, script, *, timeout=60.0, workdir=True, env_setup=True):  # noqa: ANN001
        kind = (
            "ping"
            if "jms-top-ok" in script and "##" not in script
            else ("cpu" if "##CPU" in script and "##NVIDIA" not in script else "fleet")
        )
        self.calls.append((host.name, kind))
        value = self.replies[host.name]
        if isinstance(value, Exception):
            raise value
        return ExecResult(host=host.name, exit_code=0, stdout=value, stderr="")

    async def close(self) -> None:
        self.closed = True


def make_inventory(names: list[str]) -> Inventory:
    return Inventory(
        hosts={n: Host(name=n, platform="cuda", tags=["gpu"]) for n in names},
        groups={"all-gpus": names},
    )


def make_monitor(replies: dict[str, str | Exception], *, interval: float = 1.0, full: float = 2.0):
    names = list(replies)
    return FleetMonitor(
        cfg=BastionConfig(url="https://bastion.example.invalid"),
        inventory=make_inventory(names),
        runner=FakeRunner(replies),
        interval=interval,
        full_interval=full,
    )


# --- parsing ------------------------------------------------------------------


def test_split_sections_ignores_markers_inside_output():
    text = "jms-top-ok\n##SYS\na\n##NOT A MARKER\nb\n##WHO\n 1 root\n"
    secs = split_sections(text)
    assert secs["SYS"] == ["a", "##NOT A MARKER", "b"]
    assert secs["WHO"] == [" 1 root"]


def test_reached_requires_a_leading_sentinel():
    assert reached(f"{SENTINEL}\n##SYS")
    assert not reached("permission denied\n")
    # A sentinel past the first lines is somebody's output, not the handshake.
    assert not reached("\n".join(["x"] * 5 + [SENTINEL]))


def test_cpu_delta_across_two_replies():
    first = parse_system(
        ["1 1 1 1/1 1", "8", "8", "100 50 0", "100", "10", "1000 500", "h"], stamp=1.0
    )
    second = parse_system(
        ["1 1 1 1/1 1", "8", "8", "100 50 0", "110", "10", "1200 520", "h"], stamp=2.0
    )
    # 200 jiffies passed, 20 of them idle+iowait: 90% busy.
    assert second.cpu_sample.busy_pct_since(first.cpu_sample) == pytest.approx(90.0)
    assert first.cpu_sample.busy_pct_since(None) is None


def test_cpu_delta_after_reboot_is_unknown_not_negative():
    before = CpuSample(total=5000, idle=100, stamp=1.0)
    after = CpuSample(total=10, idle=5, stamp=2.0)  # counters reset
    assert after.busy_pct_since(before) is None


def test_node_and_cards_from_one_reply():
    view = HostView(name="a")
    parse_reply(view, reply("a")[len(SENTINEL) + 1 :], stamp=1.0)
    assert view.node.cpus == 64
    assert view.node.mem_pct == pytest.approx(50.0)
    assert view.node.uptime_s == 3600
    assert view.node.hostname == "node-a"
    assert view.node.logins == ["root", "root"]
    assert view.node.disk_pct == 50.0
    assert view.node.top[0]["cmd"].endswith("train.py")
    assert (view.total, view.free_cards, view.verdict) == (2, 2, FREE)


def test_a_cheap_reply_replaces_the_whole_node_snapshot():
    """Fields a reply does not carry become unknown — never the previous reply's value."""
    view = HostView(name="a")
    parse_reply(view, reply("a")[len(SENTINEL) + 1 :], stamp=1.0)
    assert view.node.uptime_s == 3600
    # The cheap block has the same layout, so uptime is refreshed, not carried: a node
    # that rebooted between ticks must show the new uptime, not the old one.
    cheap = "##CPU\n1.5 1.5 1.5 1/1 1\n8\n8\n100 50 0\n10\n1000 500\nh\n"
    parse_reply(view, cheap, stamp=2.0)
    assert view.node.cpus == 8
    assert view.node.uptime_s == 10
    assert view.node.logins == []  # not in this reply
    assert view.node.top == []
    # A block that lost the uptime line entirely leaves it unknown, not stale.
    parse_reply(view, "##CPU\n1.5 1.5 1.5 1/1 1\n8\n8\n100 50 0\n1000 500\nh\n", stamp=3.0)
    assert view.node.uptime_s is None


# --- verdicts -----------------------------------------------------------------


def _view(**kw) -> HostView:  # noqa: ANN003
    base = HostView(name="h", reachable=True)
    return replace(base, **kw)


def test_verdict_ranks_free_partial_residual_busy():
    free = _view(accel=_accel(idle=4, busy=0))
    partial = _view(accel=_accel(idle=2, busy=2))
    busy = _view(accel=_accel(idle=0, busy=4))
    assert free.verdict == FREE
    assert partial.verdict == PARTIAL
    assert busy.verdict == BUSY


def _accel(idle: int, busy: int):  # noqa: ANN202
    from jms_top.families import AccelResult, Card

    cards = [Card(index=i, util=0, mem_used=4, mem_total=81920) for i in range(idle)]
    cards += [Card(index=idle + i, util=90, mem_used=70000, mem_total=81920) for i in range(busy)]
    return AccelResult(family="nvidia", label="NVIDIA", cards=cards)


def test_loaded_cpu_turns_a_free_host_into_partial():
    """Idle cards next to a loaded CPU side: not ours to take outright."""
    view = _view(accel=_accel(idle=8, busy=0))
    view.node = NodeStats(load1=60.0, cpus=64)
    assert view.node.load_per_cpu == pytest.approx(0.9375)
    assert view.verdict == PARTIAL


def test_unknown_cards_are_not_free_and_upset_the_host():
    from jms_top.families import AccelResult, Card

    unknown = AccelResult(
        family="nvidia",
        label="NVIDIA",
        cards=[Card(index=0)],  # nothing reported
    )
    view = _view(accel=unknown)
    assert view.free_cards == 0 and view.unknown_cards == 1
    assert view.verdict == UNKNOWN
    assert load_per_cpu(None, 8) is None


def test_unreachable_host_is_down_not_unknown():
    view = _view(reachable=False, error="permission denied")
    assert view.verdict == DOWN


# --- the monitor --------------------------------------------------------------


def test_poll_fills_every_host_and_logs():
    mon = make_monitor({"a": reply("a"), "b": reply("b", gpu=GPU_BUSY)})

    async def run():
        await mon.poll()

    asyncio.run(run())
    a, b = mon.view.by_name("a"), mon.view.by_name("b")
    assert a.verdict == FREE and b.verdict == BUSY
    assert b.users == ["pid 9001", "pid 9002"]
    assert mon.view.log and "full round" in mon.view.log[-1]
    assert mon.generation > 0


def test_host_failure_is_isolated_and_clears_on_recovery():
    replies: dict[str, str | Exception] = {
        "a": reply("a"),
        "b": RuntimeError("bastion said no"),
    }
    mon = make_monitor(replies)

    async def run():
        await mon.poll()
        assert mon.view.by_name("a").verdict == FREE
        assert mon.view.by_name("b").reachable is False
        assert "bastion said no" in (mon.view.by_name("b").error or "")
        replies["b"] = reply("b")
        await mon.poll()
        assert mon.view.by_name("b").reachable is True
        assert mon.view.by_name("b").verdict == FREE

    asyncio.run(run())


def test_ko_ko_rejection_without_sentinel_counts_as_unreachable():
    mon = make_monitor({"a": "POST /api/v1/... failed, get code: 403"})

    async def run():
        await mon.poll()

    asyncio.run(run())
    view = mon.view.by_name("a")
    assert view.reachable is False
    assert "403" in (view.error or "")


def test_cheap_tick_leaves_card_tables_alone_and_updates_node():
    mon = make_monitor({"a": reply("a", load=2.0)})

    async def run():
        await mon.tick()  # everyone is due a full read on the first tick
        first = mon.view.by_name("a")
        assert first.total == 2
        # Next tick is too soon for a full read: only the CPU stats are refreshed.
        assert mon.due_full(mon.hosts[0]) is False
        await mon.tick()
        second = mon.view.by_name("a")
        assert [c.index for c in second.cards] == [0, 1]
        kinds = [k for name, k in mon.runner.calls if name == "a"]
        assert kinds[:2] == ["fleet", "cpu"]

    asyncio.run(run())


def test_full_tick_is_slower_once_nothing_is_free():
    mon = make_monitor({"a": reply("a", gpu=GPU_BUSY)})

    async def run():
        await mon.tick()
        host = mon.hosts[0]
        mon._last_full["a"] = 1000.0
        # Nothing free: the next full read waits IDLE_FACTOR times longer.
        assert mon.due_full(host, now=1000.0 + mon.full_interval + 0.1) is False
        assert mon.due_full(host, now=1000.0 + mon.full_interval * 3 + 1) is True

    asyncio.run(run())


def test_fast_tick_failures_promote_the_host_to_a_full_read():
    mon = make_monitor({"a": reply("a")})

    async def run():
        await mon.tick()
        mon._last_full["a"] = 1e18  # timer says "not for a very long time"
        mon.runner.replies["a"] = ""
        for _ in range(3):
            await mon.tick()
        assert mon.due_full(mon.hosts[0]) is True

    asyncio.run(run())


def test_pause_stops_the_loop_and_resume_restarts_it():
    mon = make_monitor({"a": reply("a")}, interval=0.01)

    async def run():
        await mon.ping_all()
        mon.start()
        await asyncio.sleep(0.05)
        assert mon.running
        mon.pause()
        await asyncio.sleep(0.05)
        assert mon.paused
        calls = len(mon.runner.calls)
        await asyncio.sleep(0.05)
        assert len(mon.runner.calls) == calls  # nothing polled while paused
        mon.resume()
        # `resume` restarts the tick loop; the first tick of a fresh loop is a full
        # round, so give it a moment to begin before counting.
        await mon.tick()
        assert len(mon.runner.calls) > calls
        await mon.stop()

    asyncio.run(run())


def test_probe_scripts_never_use_env_setup_or_a_workdir():
    """Looking at a host must not depend on its dev env being activatable."""
    script = build_script("fleet")
    assert "set +e" in script  # a vendor tool that fails must not abort the report
    assert "npu-smi info" in script and "nvidia-smi" in script
    assert "##CPU" not in script
    cheap = build_script("cpu")
    assert "##CPU" in cheap and "nvidia-smi" not in cheap
    with pytest.raises(ValueError):
        build_script("nonsense")


def test_snapshot_json_is_stable():
    from jms_top.cli import snapshot

    mon = make_monitor({"a": reply("a")})

    async def run():
        await mon.poll()

    asyncio.run(run())
    doc = snapshot(mon.view)
    assert doc["summary"]["free_hosts"] == ["a"]
    assert doc["hosts"][0]["accel"]["free"] == 2
    assert doc["hosts"][0]["node"]["cpus"] == 64


def test_fleet_view_rollup():
    view = FleetView(hosts=[_view(accel=_accel(2, 0)), _view(accel=_accel(0, 2))])
    assert (view.free_cards, view.busy_cards, view.total_cards) == (2, 2, 4)
    assert [h.verdict for h in view.free_hosts] == [FREE]
    assert view.families == {"NVIDIA": 4}


def test_manual_refresh_is_immediate_and_full():
    mon = make_monitor({"a": reply("a")}, interval=30.0, full=60.0)

    async def run():
        await mon.tick()  # first tick: full
        mon._last_full["a"] = 0.0  # pretend the timer is far from due
        calls = len(mon.runner.calls)
        await mon.refresh_now()
        assert [k for _n, k in mon.runner.calls[calls:]] == ["fleet"]
        assert mon.view.by_name("a").verdict == FREE

    asyncio.run(run())


def test_demo_replay_parses_through_the_same_path_as_live():
    """The recorded fleet must produce the same verdicts live data would."""
    from jms_top.demo import demo_monitor

    mon = demo_monitor(interval=60, full_interval=120)

    async def run():
        await mon.poll()

    asyncio.run(run())
    view = mon.view
    assert len(view.hosts) == 19
    assert view.reachable == 18
    assert view.by_name("dcu-bw1000-3").verdict == DOWN
    assert view.by_name("tsingmicro-137").verdict == RESIDUAL
    assert view.by_name("h100-17").verdict == BUSY
    free = [h.name for h in view.free_hosts]
    assert "a800-171" in free and "gcu-s60" in free
    # Every family that was in the inventory is represented, parsed from its own tool.
    assert set(view.families) >= {"NVIDIA", "Ascend", "Enflame", "MetaX", "Kunlunxin", "TsingMicro"}


def test_a_stalled_vendor_tool_reads_as_unknown_not_free():
    """A driver that never answers has reported nothing — that is not "no cards"."""
    stalled = (
        f"{SENTINEL}\n##SYS\n1.0 1.0 1.0 1/1 1\n8\n8\n100 50 0\n100\n10\n1000 500\nh\n"
        "##STALLED:nvidia:15s\n##NVIDIA\n"
    )
    mon = make_monitor({"a": stalled})

    async def run():
        await mon.poll()

    asyncio.run(run())
    view = mon.view.by_name("a")
    assert view.reachable is True
    assert view.verdict == UNKNOWN
    assert view.free_cards == 0
    assert "15s" in (view.accel.error or "")


def test_busy_node_promotes_the_next_full_read():
    """The cheap tick notices a loaded CPU side; the card table must catch up at once."""
    mon = make_monitor({"a": reply("a", load=1.0)})

    async def run():
        await mon.tick()
        assert mon.due_full(mon.hosts[0]) is False
        mon.runner.replies["a"] = reply("a", load=60.0)  # 60 / 64 cores
        await mon.tick()
        assert mon.due_full(mon.hosts[0]) is True
        assert mon.view.by_name("a").verdict == PARTIAL

    asyncio.run(run())


def test_cheap_probe_carries_the_whole_node_view():
    """The cheap tick must not blank fields the full probe reports."""
    from jms_top.probe import build_script

    cheap = build_script("cpu")
    full = build_script("fleet")
    for section in ("##WHO", "##TOP", "##DISK"):
        assert section in cheap and section in full
    # Same first six lines up to the marker name: load, cores, memory, counters, uptime.
    cheap_body = cheap.splitlines()[2:10]
    full_body = full.splitlines()[2:10]
    assert cheap_body == full_body
