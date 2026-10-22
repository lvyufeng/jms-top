"""Task description -> ranked machines.

The rules here are all about what must *not* happen: recommending a node with a
neighbour to a distributed job, treating unreadable memory as free, or silently
dropping a stated requirement because it could not be parsed.
"""

from __future__ import annotations

import pytest

from jms_top.advise import Want, advise, parse_task, reject, score
from jms_top.families import AccelResult, Card
from jms_top.model import HostView, NodeStats


def host(
    name,
    *,
    free,
    total,
    mem_gib=80,
    load=0.0,
    verdict=None,
    family="nvidia",
    model="NVIDIA A800-SXM4-80GB",
    temp=40.0,
):
    """A host with ``free`` idle cards out of ``total``."""
    cards = []
    for i in range(total):
        busy = i >= free
        cards.append(
            Card(
                index=i,
                name=model,
                util=80.0 if busy else 0.0,
                mem_used=(mem_gib * 1024 * 0.9) if busy else 0.0,
                mem_total=mem_gib * 1024,
                temp=temp,
            )
        )
    accel = AccelResult(family=family, label=family.title(), cards=cards)
    # `load_per_cpu` is derived; give it the raw load that produces the wanted value.
    node = NodeStats(cpus=64, load1=load * 64)
    v = HostView(name=name, platform=family, reachable=True, accel=accel, node=node)
    if verdict:
        v.verdict = verdict
    return v


# --- parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,field,value",
    [
        ("我要跑 8 卡 A100 微调", "cards", 8),
        ("要八卡", "cards", 8),
        ("两卡就行", "cards", 2),
        ("单卡调试", "cards", 1),
        ("显存至少 60G", "min_mem_gib", 60.0),
        ("80G 的显存", "min_mem_gib", 80.0),
        ("显存 32000M", "min_mem_gib", 31.25),
        ("预计 6 小时", "hours", 6.0),
        ("跑 2.5 h", "hours", 2.5),
        ("半小时", "hours", 0.5),
        ("要 910B", "family", "910b"),
        ("用 A100", "family", "a100"),
    ],
)
def test_each_phrase_is_parsed(text, field, value):
    assert getattr(parse_task(text), field) == value


def test_longest_family_word_wins():
    """`910B` must not collapse to `910`.

    Dicts have no order contract; matching `910` first and then searching for it would
    also hit `910B` — correct by luck there, and wrong when someone asks for the older
    model specifically.
    """
    assert parse_task("要 910B").family == "910b"
    assert parse_task("要 910C").family == "910c"


def test_unstated_stays_none_rather_than_zero():
    """Guessing a card count would silently narrow the fleet."""
    want = parse_task("跑一个大点的任务")
    assert want.cards is None
    assert want.min_mem_gib is None
    assert want.hours is None


def test_parallel_wording_means_exclusive_and_scratch_wording_does_not():
    assert parse_task("分布式训练 4 卡").exclusive is True
    assert parse_task("vLLM 推理服务").exclusive is True
    assert parse_task("单卡调试一下").exclusive is False
    # Asking for several cards at all is parallel work by another name.
    assert parse_task("要 8 卡").exclusive is True


# --- rejection ----------------------------------------------------------------


def test_unknown_is_never_recommended():
    """`verdict` is derived, so the states are built out of their real causes."""
    unreachable = HostView(name="down", reachable=False)
    assert reject(unreachable, Want()) is not None

    broken = host("unknown", free=8, total=8)
    broken.accel = None  # what a stalled vendor tool leaves behind
    assert reject(broken, Want()) is not None


def test_a_host_with_too_few_cards_is_rejected_with_the_count():
    why = reject(host("h", free=3, total=8), Want(cards=8))
    assert why and "3" in why and "8" in why


def test_a_host_whose_cards_are_too_small_is_rejected():
    why = reject(host("h", free=8, total=8, mem_gib=40), Want(cards=1, min_mem_gib=60))
    assert why and "40" in why


def test_unknown_memory_is_not_a_rejection_but_costs_rank():
    """A card that reports no memory cannot be checked — so it must not be dropped."""
    h = host("h", free=8, total=8)
    for card in h.accel.cards:
        card.mem_total = None
    assert reject(h, Want(cards=1, min_mem_gib=60)) is None
    cand = score(h, Want(cards=1, min_mem_gib=60, exclusive=True))
    assert any("显存未知" in r for r in cand.reasons)


def test_a_busy_cpu_node_is_not_offered_for_an_interactive_job():
    why = reject(host("h", free=8, total=8, load=1.4), Want(cards=1))
    assert why and "负载" in why


# --- ranking ------------------------------------------------------------------


def test_exclusive_prefers_a_whole_node_over_one_with_a_neighbour():
    """The point of exclusivity: a neighbour shares the interconnect, not a slot.

    A 7/8-free node must not outrank an 8/8 one even though 7 cards are ample —
    somebody else is computing on that board.
    """
    neighbour = host("neighbour", free=7, total=8)
    whole = host("whole", free=8, total=8)
    result = advise([neighbour, whole], Want(cards=4, exclusive=True))
    assert [c.name for c in result.candidates] == ["whole", "neighbour"]
    assert result.best.score - result.candidates[1].score > 10


def test_shared_prefers_the_node_somebody_is_already_on():
    """A scratch run should not take a pristine node a large job may need whole."""
    neighbour = host("neighbour", free=7, total=8)
    whole = host("whole", free=8, total=8)
    result = advise([neighbour, whole], Want(cards=1, exclusive=False))
    assert result.candidates[0].name == "neighbour"


def test_among_whole_nodes_the_one_that_just_fits_wins():
    small = host("small", free=4, total=4)
    big = host("big", free=8, total=8)
    result = advise([big, small], Want(cards=4, exclusive=True))
    assert result.candidates[0].name == "small"


def test_a_quiet_node_beats_a_size_match():
    """The just-fits bonus is a tie-break, never a reason to accept a noisy node."""
    noisy_but_sized = host("noisy", free=4, total=4, load=0.6)
    quiet_but_large = host("quiet", free=8, total=8, load=0.0)
    result = advise([noisy_but_sized, quiet_but_large], Want(cards=4, exclusive=True))
    assert result.candidates[0].name == "quiet"


# --- the whole answer ---------------------------------------------------------


def test_advice_explains_a_shortfall_instead_of_saying_nothing():
    """`no machines` is useless; `6 wrong model, 1 too small` says whether to wait."""
    hosts = [
        host("a", free=8, total=8, model="NVIDIA H100"),
        host("b", free=8, total=8, model="NVIDIA H100"),
        host("c", free=2, total=8),
    ]
    result = advise(hosts, Want(cards=8, family="a100"))
    assert result.candidates == []
    shortfall = result.shortfall()
    assert "型号不符" in shortfall
    assert "3 台" in shortfall


def test_candidates_never_exceed_the_limit():
    hosts = [host(f"h{i}", free=8, total=8) for i in range(20)]
    assert len(advise(hosts, Want(cards=1), limit=5).candidates) == 5


def test_a_rejected_host_never_appears_in_the_ranking():
    hosts = [host("big", free=8, total=8), host("tiny", free=1, total=8)]
    result = advise(hosts, Want(cards=6))
    assert [c.name for c in result.candidates] == ["big"]
    assert any(r.host == "tiny" for r in result.rejected)


def test_the_scope_word_decides_not_the_verb():
    """`跑个大任务` and `跑个脚本` share a verb and mean opposite things.

    A bare `跑个` hint read both as scratch runs, which for the first one is exactly
    backwards: it would send a real job to a noisy shared node.
    """
    assert parse_task("跑个大任务").exclusive is True
    assert parse_task("跑个脚本试试").exclusive is False
    assert parse_task("长时间训练").exclusive is True
