"""The negotiation message: what it says, and what it refuses to say."""

from __future__ import annotations

from datetime import datetime

from jms_top.advise import Want
from jms_top.notify import compose, kill_hint, post


def _host(free=8, total=8, procs=()):
    from jms_top.families import AccelResult, Card, Proc
    from jms_top.model import HostView, NodeStats

    cards = []
    for i in range(total):
        busy = i >= free
        card = Card(
            index=i,
            name="NVIDIA A100-SXM4-40GB",
            util=90.0 if busy else 0.0,
            mem_used=31846.0 if busy else 0.0,
            mem_total=40960.0,
        )
        if busy:
            card.procs = [p for p in procs] or [
                Proc(pid=2882417, user="root", name="python3", mem_mib=31846.0)
            ]
        cards.append(card)
    accel = AccelResult(family="nvidia", label="NVIDIA", cards=cards)
    return HostView(
        name="a100-42",
        platform="nvidia",
        reachable=True,
        accel=accel,
        node=NodeStats(cpus=64, load1=1.0),
    )


def test_a_free_enough_host_gets_a_claim_not_a_negotiation():
    """Nothing to ask for when the cards are there: announce, do not negotiate."""
    draft = compose(
        host=_host(free=8),
        want=Want(cards=4, note="推理服务", exclusive=False),
        hours=3,
        requester="张三",
    )
    assert draft.kind == "claim"
    assert "算力占用" in draft.title
    assert "3 小时" in draft.body
    assert "@张三" in draft.body
    assert draft.blockers == []


def test_an_occupied_host_names_the_process_that_is_in_the_way():
    """`pid 2882417` alone is anonymous; the owner must recognise their own job."""
    from jms_top.families import Proc

    proc = Proc(
        pid=2882417,
        user="root",
        name="python3",
        mem_mib=31846.0,
        cmdline="python3 /workspace/train.py --tag lora-v2",
        elapsed="17:37:36",
    )
    draft = compose(host=_host(free=0, procs=[proc]), want=Want(cards=8, note="微调"))
    assert draft.kind == "negotiate"
    assert "算力协商" in draft.title
    assert "2882417" in draft.body
    assert "root" in draft.body
    assert "17:37:36" in draft.body
    assert "train.py --tag lora-v2" in draft.body
    assert "8 卡" in draft.body


def test_a_negotiation_does_not_invent_a_deadline():
    """The requester's run time says nothing about when the current job ends.

    An earlier version reused `hours` here and produced "please free these by 17:53"
    for a job that had just started.
    """
    draft = compose(
        host=_host(free=0),
        want=Want(cards=8, hours=4.0),
        hours=4.0,
        now=datetime(2026, 10, 22, 17, 53),
    )
    assert "17:53" not in draft.body
    assert "立即" in draft.body or "释放" in draft.body


def test_the_kill_hint_is_a_suggestion_never_an_action():
    draft = compose(host=_host(free=0), want=Want(cards=8))
    hint = kill_hint(draft)
    assert "kill -TERM 2882417" in hint
    assert "确认后再执行" in hint


def test_a_claim_has_no_kill_hint():
    assert kill_hint(compose(host=_host(free=8), want=Want(cards=4))) == ""


def test_sending_to_a_dead_webhook_fails_without_raising():
    """The draft is still on screen; a broken bot must not take the command down."""
    ok, detail = post("http://127.0.0.1:9/nope", "hi", timeout=1.0)
    assert ok is False
    assert detail


def test_a_missing_command_line_degrades_instead_of_inventing_one():
    from jms_top.families import Proc

    proc = Proc(pid=1234, user=None, name="python3", mem_mib=None)
    draft = compose(host=_host(free=0, procs=[proc]), want=Want(cards=8))
    assert "1234" in draft.body
    assert "python3" in draft.body
