"""Parser tests, pinned to output captured from the live fleet on 2026-10-21.

Every fixture in ``fixtures/smi`` is verbatim ``*smi`` output from a real host; the
file name says which. The assertions are about *judgements*, not just numbers —
which cards are free, which are held by memory alone, which host must be reported as
unknown — because those are the ones that send work onto a machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jms_top.families import (
    BUSY_MEM_FRACTION,
    FAMILIES,
    Nvidia,
    family_hint,
)

FIX = Path(__file__).parent / "fixtures" / "smi"


def load(name: str) -> str:
    return (FIX / name).read_text()


# --- NVIDIA -------------------------------------------------------------------


def test_nvidia_a800_is_genuinely_idle():
    """4 MiB of driver context, 0% util, no processes: this host is free."""
    r = FAMILIES["nvidia"].parse(load("nvidia_a800.txt"))
    assert (r.total, r.free, r.busy) == (8, 8, 0)
    assert r.version == "580.126.20"
    card = r.cards[0]
    assert card.name == "NVIDIA A800-SXM4-80GB"
    assert card.mem_used == pytest.approx(4)
    assert card.mem_total == pytest.approx(81920)
    assert card.procs == []


def test_nvidia_h100_running_training():
    """Eight cards at ~94% memory with one python each: every card is busy."""
    r = FAMILIES["nvidia"].parse(load("nvidia_h100_tile.txt"))
    assert (r.total, r.free, r.busy) == (8, 0, 8)
    card = r.cards[0]
    assert card.procs[0].pid == 1062720
    assert card.procs[0].name.startswith("python")


def test_nvidia_query_csv_with_and_without_uuid():
    """The uuid column is honoured by some drivers and silently dropped by others."""
    with_uuid = (
        "0, NVIDIA H800, GPU-abc, 96 %, 76000 MiB, 81559 MiB, 55, 500.1\n"
        "GPU-abc, 4242, /usr/bin/torchrun, 75998 MiB\n"
    )
    without = (
        "0, NVIDIA A100-SXM4-40GB, 0, 31823, 40960, 31, 57.25, 00000000:03:00.0\n"
        "1, NVIDIA A100-SXM4-40GB, 0, 4, 40960, 30, 58.82, 00000000:16:00.0\n"
        "00000000:03:00.0, 111, /x/python, 31800\n"
    )
    a = FAMILIES["nvidia"].parse(with_uuid)
    assert (a.total, a.busy) == (1, 1)
    assert a.cards[0].procs[0].pid == 4242
    assert a.cards[0].util == 96.0

    b = FAMILIES["nvidia"].parse(without)
    assert (b.total, b.free, b.busy) == (2, 1, 1)
    # 77% of memory is held, far above the commit threshold: busy even at 0% util.
    assert b.cards[0].busy is True
    assert b.cards[0].procs[0].pid == 111
    assert b.cards[1].free is True


def test_nvidia_driver_failure_is_an_error_not_a_free_host():
    """``metax-mc550-121`` prints nvidia-smi's driver error — must not read 'idle'."""
    r = FAMILIES["nvidia"].parse(
        "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
    )
    assert r.total == 0
    assert r.error and "NVIDIA driver" in r.error


# --- Ascend -------------------------------------------------------------------


def test_ascend_910c_two_chips_per_card():
    r = FAMILIES["ascend"].parse(load("ascend_910c_108.txt"))
    assert r.total == 16  # 8 cards x 2 chips, not 8
    assert r.version == "25.5.0"
    first = r.cards[0]
    assert first.name == "Ascend910"
    assert first.health == "OK"
    assert first.temp == 58.0 and first.power == pytest.approx(513.9)
    assert first.procs and first.procs[0].name == "python"
    # The process table's rows are keyed by (NPU card, chip-in-card); chip 1 of card 0
    # is the *second* card in the list, and it holds a different pid.
    assert [p.pid for p in r.cards[1].procs] == [2510885]
    # "No running processes found in NPU 7" — both of card 7's chips are free.
    assert [c.index for c in r.cards if c.free][-2:] == [14, 15]


def test_ascend_memory_held_without_process_is_residual():
    r = FAMILIES["ascend"].parse(load("ascend_910c_110_idle.txt"))
    assert r.total == 16
    assert r.residual >= 1
    residual = [c for c in r.cards if c.residual]
    assert all(c.util == 0 for c in residual)


# --- Kunlunxin ----------------------------------------------------------------


def test_kunlunxin_two_kinds_of_busy():
    """Card 0 is repointed at 100% by a test script; card 3 is held by five vLLM procs."""
    r = FAMILIES["kunlunxin"].parse(load("kunlunxin_p800_13.txt"))
    assert (r.total, r.free, r.busy) == (8, 6, 2)
    assert r.cards[0].busy is True and r.cards[0].util == 100.0
    assert [c.index for c in r.cards if c.procs] == [0, 3]
    assert {p.name for p in r.cards[3].procs} == {"VLLM::EngineCore"}


def test_kunlunxin_0_percent_util_is_not_idle():
    """p800-15 prints 0% util on cards holding 87% of memory: busy, not free."""
    r = FAMILIES["kunlunxin"].parse(load("kunlunxin_p800_15_mem.txt"))
    assert r.total == 8
    assert r.free == 0
    assert all(c.util == 0 for c in r.cards)
    assert all(c.mem_frac > BUSY_MEM_FRACTION for c in r.cards)


# --- MetaX --------------------------------------------------------------------


def test_metax_util_lives_on_the_name_row():
    r = FAMILIES["metax"].parse(load("metax_mc550_111.txt"))
    assert (r.total, r.free, r.busy) == (8, 0, 8)
    card = r.cards[0]
    assert card.util == 97.0  # not None, not the stats line's "Available"
    assert card.mem_total == pytest.approx(65536)
    assert card.procs[0].name == "VLLM::Worker_TP"


# --- Enflame ------------------------------------------------------------------


def test_enflame_idle_cards():
    r = FAMILIES["enflame"].parse(load("enflame_s60.txt"))
    assert (r.total, r.free) == (8, 8)
    card = r.cards[0]
    assert card.name == "ZIXIAOC200"
    assert card.temp == 51.0
    assert card.procs == [] and card.procs_known


# --- Moore Threads ------------------------------------------------------------


def test_mthreads_idle_and_busy():
    idle = FAMILIES["mthreads"].parse(load("mthreads_16_idle.txt"))
    assert (idle.total, idle.free) == (8, 8)
    assert idle.cards[0].temp == 35.0

    busy = FAMILIES["mthreads"].parse(load("mthreads_s5000_17.txt"))
    assert (busy.total, busy.free, busy.busy) == (8, 0, 8)
    assert any(p.name.startswith("sglang") for p in busy.cards[0].procs)


# --- TsingMicro ---------------------------------------------------------------


def test_tsingmicro_is_residual_not_free():
    """0% util but 90% memory and no process: the memory is not actually available."""
    r = FAMILIES["tsingmicro"].parse(load("tsingmicro_137.txt"))
    assert r.total == 8  # one chip per card on this SKU
    assert r.free == 0
    assert r.residual == 8
    assert all(c.residual for c in r.cards)


# --- cross-family invariants --------------------------------------------------


@pytest.mark.parametrize(
    "family,fixture,expect_total",
    [
        ("nvidia", "nvidia_a800.txt", 8),
        ("nvidia", "nvidia_h100_tile.txt", 8),
        ("ascend", "ascend_910c_108.txt", 16),
        ("kunlunxin", "kunlunxin_p800_13.txt", 8),
        ("metax", "metax_mc550_111.txt", 8),
        ("enflame", "enflame_s60.txt", 8),
        ("mthreads", "mthreads_16_idle.txt", 8),
        ("mthreads", "mthreads_s5000_17.txt", 8),
        ("tsingmicro", "tsingmicro_137.txt", 8),
    ],
)
def test_every_card_is_classified(family, fixture, expect_total):
    """No card may end up in an undefined state: free, busy or unknown — never blank."""
    r = FAMILIES[family].parse(load(fixture))
    assert r.total == expect_total
    assert r.free + r.busy + r.unknown == r.total
    for card in r.cards:
        assert card.state in {"free", "busy", "unknown"}
        if card.busy is None:
            # Unknown is only legitimate when the vendor reported nothing at all.
            assert card.util is None and card.mem_frac is None and not card.procs


def test_empty_output_never_reports_a_free_host():
    for fam in FAMILIES.values():
        r = fam.parse("")
        assert r.total == 0
        assert r.free == 0


def test_family_hint_follows_platform_then_tags():
    assert family_hint("cuda") == "nvidia"
    assert family_hint("ascend-910c") == "ascend"
    assert family_hint("generic", ["metax"]) == "metax"
    assert family_hint("generic", ["npu"]) == "ascend"
    assert family_hint("generic", []) == "generic"
    # hy-smi has no parser yet: still a CPU-usable host, never mislabelled as NVIDIA.
    assert family_hint("dcu", ["dcu"]) == "generic"


def test_nvidia_parser_survives_truncated_output():
    """The output budget can cut the last row; it must still parse, not vanish."""
    body = "0, NVIDIA A100, 0, 31823, 40960, 31, 57.2\n1, NVIDIA A100, 0, 31823, 40960, 31"
    r = Nvidia().parse(body)
    assert r.total == 2
    # Half a row parses as far as it goes and reports the rest as unknown — never as a
    # neighbour's number, and never fabricated as zero.
    assert r.cards[1].mem_total == pytest.approx(40960)
    assert r.cards[1].power is None
    assert r.cards[1].temp is None


# --- MetaX both layouts ---------------------------------------------------------


def test_metax_wide_layout_from_the_per_type_queries():
    """沐曦 ``mx-smi --show-memory/--show-usage/--show-process`` (driver 2.3.x).

    Regression: the three queries repeat the same ``GPU#n`` blocks, and the process
    table repeats "GPU" as a column heading. Reading either as a card turned an 8-GPU
    host into a 15-card one with no processes on any of them.
    """
    r = FAMILIES["metax"].parse(load("metax_wide_c550.txt"))
    assert (r.total, r.free, r.busy) == (8, 0, 8)
    assert [c.index for c in r.cards] == list(range(8))
    assert r.cards[0].mem_total == pytest.approx(65536)
    assert r.cards[0].mem_frac > 0.5  # held through the VLLM workers
    assert r.cards[0].procs[0].pid == 2744908


def test_metax_default_table_still_parses():
    """A driver that predates the per-type flags falls back to the table layout."""
    r = FAMILIES["metax"].parse(load("metax_mc550_111.txt"))
    assert (r.total, r.busy) == (8, 8)
    assert r.cards[0].util == 97.0


def test_metax_empty_output_is_an_error_not_a_free_host():
    r = FAMILIES["metax"].parse("")
    assert r.total == 0 and r.error


# --- T-Head PPU ---------------------------------------------------------------


def test_thead_summary_table_from_the_live_fleet():
    """平头哥 ZW810E summary table: 16 PPUs, all idle, memory reserved at 2 MiB.

    Captured from ``ppu-smi`` 1.28 on driver 1.3.2. The rows wrap onto three lines per
    device, so a parser that reads one line per card sees nothing.
    """
    r = FAMILIES["thead"].parse(load("thead_zw810e_16_idle.txt"))
    assert r.total == 16
    assert r.free == 16
    assert r.cards[0].name == "PPU-ZW810E"
    assert r.cards[0].temp == pytest.approx(33)
    assert r.cards[0].power == pytest.approx(82)
    assert r.cards[0].mem_total == pytest.approx(97920)
    assert r.cards[0].mem_used == pytest.approx(2)
    assert r.cards[0].util == 0.0


def test_thead_is_probed_from_the_sdk_path():
    """The SDK does not put `ppu-smi` on PATH — 16 PPUs were present and 0 tools found."""
    from jms_top.probe import build_script

    script = build_script("fleet")
    assert "/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi" in script
    # Falls back to the SDK path only after `command -v` came up empty.
    assert script.index("command -v ppu-smi") < script.index("/usr/local/PPU_SDK")


def test_every_vendor_stall_marker_names_its_own_family():
    """A copy-pasted snippet once made every vendor report `STALLED:ascend`.

    The marker is what the UI reads to say *which* tool went quiet, so a mislabelled
    one sends you to the wrong driver.
    """
    import re

    from jms_top.probe import ACCEL_SNIPPETS

    for family, template in ACCEL_SNIPPETS.items():
        labels = re.findall(r"##STALLED:([a-z]+):", template)
        assert labels, f"{family} snippet has no stall marker"
        assert set(labels) == {family}, f"{family} snippet is labelled {labels}"
