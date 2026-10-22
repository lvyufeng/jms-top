"""Turn a task description into a ranked list of machines that can run it.

Four stages, and the split matters more than any one of them:

1. :func:`parse_task` — natural language to a :class:`JobSpec`. Rules, no model.
2. :meth:`JobSpec.reject` — hard constraints. A machine that cannot hold the job never
   reaches the ranking, so no later stage can recommend one.
3. :func:`score` — a number plus the reasons behind it.
4. an optional model pass that explains the result (:mod:`jms_top.ai`).

The model never sees the fleet and never does arithmetic. It sees three to five
pre-filtered candidates with their facts already computed, because a language model
counting `F` characters in `FFFFBBBB` gets it wrong often enough to matter, and being
wrong here means telling someone a machine has room it does not have.

**Exclusive vs shared** is the axis everything else hangs on:

* exclusive — distributed work. What matters is how much of the *node* is free, since
  a neighbour on the same board competes for the same interconnect, not for the same
  card slot.
* shared — a scratch run. A neighbour costs nothing, so a machine that already has
  someone on it is a better pick than a pristine one, which should stay pristine for
  the next job that needs it whole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import HostView

#: A node at or above this load per core cannot be trusted with an interactive job,
#: whether or not its cards are free.
BUSY_LOAD_PER_CORE = 0.75


@dataclass
class Want:
    """What the task needs. ``None`` means "not stated", never "zero"."""

    cards: int | None = None
    min_mem_gib: float | None = None
    family: str | None = None
    hours: float | None = None
    exclusive: bool | None = None
    note: str = ""


# --- stage 1: parsing ---------------------------------------------------------

_CN_NUM = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十六": 16,
}

#: `8卡`, `8 卡`, `八卡`, `8 张卡`, `8张`
_CARDS = re.compile(r"(?P<n>\d+|[一两二三四五六七八九十])\s*(?:张|块)?\s*卡")
#: `单卡`, `一张`
_ONE_CARD = re.compile(r"单卡|一张")

#: `60G 显存`, `显存要 60G`, `显存至少 60G`, `≥60G`
_MEM = re.compile(r"显存[^\d]{0,6}(?P<n>\d+(?:\.\d+)?)\s*(?P<u>G|M|T)?", re.IGNORECASE)
_MEM_AFTER = re.compile(r"(?P<n>\d+(?:\.\d+)?)\s*(?P<u>[GMT])B?\s*(?:的)?\s*显存", re.IGNORECASE)

#: `6 小时`, `6h`, `半小时`, `预计 3 小时`
_HOURS = re.compile(r"(?P<n>\d+(?:\.\d+)?)\s*(?:个)?\s*(?:小时|h|hours?)", re.IGNORECASE)
_HALF_HOUR = re.compile(r"半\s*(?:个)?\s*小时")

#: Vendor and model words people actually type. Matched against `Card.name` and the
#: family label, so `a100` finds `NVIDIA A100-SXM4-40GB` and `910` finds `Ascend910`.
_FAMILY_WORDS = {
    "a100": "a100",
    "a800": "a800",
    "h100": "h100",
    "h800": "h800",
    "v100": "v100",
    "4090": "4090",
    "3090": "3090",
    "a6000": "a6000",
    "l40": "l40",
    "910": "910",
    "910b": "910b",
    "910c": "910c",
    "310p": "310p",
    "v150": "v150",
    "bi-v150": "bi-v150",
    "biv150": "v150",
    "c550": "c550",
    "mc550": "c550",
    "s5000": "s5000",
    "mtt": "s5000",
    "zw810": "zw810",
    "810e": "810e",
    "p800": "p800",
    "pxp": "pxp",
    "tx8110": "tx8110",
    "8110": "8110",
    "bw1100": "bw1100",
    "bw1101": "bw1101",
    "bw1000": "bw1000",
    "c200": "c200",
    "s60": "s60",
    "nvidia": "nvidia",
    "ascend": "ascend",
    "昇腾": "ascend",
    "华为": "ascend",
    "天数": "iluvatar",
    "iluvatar": "iluvatar",
    "海光": "hygon",
    "hygon": "hygon",
    "沐曦": "metax",
    "metax": "metax",
    "清微": "tsingmicro",
    "tsingmicro": "tsingmicro",
    "平头哥": "thead",
    "thead": "thead",
    "燧原": "enflame",
    "enflame": "enflame",
    "摩尔": "mthreads",
    "摩尔线程": "mthreads",
    "mthreads": "mthreads",
    "昆仑": "kunlunxin",
    "昆仑芯": "kunlunxin",
    "kunlunxin": "kunlunxin",
    "寒武纪": "cambricon",
}

#: Words that mean "this is a parallel job". Checked before the shared words, because
#: "分布式调试" is still distributed.
_EXCLUSIVE_HINTS = (
    "分布式",
    "多机",
    "多卡",
    "并行",
    "训练",
    "微调",
    "pretrain",
    "预训练",
    "ddp",
    "deepspeed",
    "torchrun",
    "mpirun",
    "hccl",
    "nccl",
    "推理服务",
    "vllm",
)
#: Words that mean "a scratch run is fine". Deliberately narrow, and only consulted
#: when no exclusive hint is present.
_SHARED_HINTS = (
    "调试",
    "试一下",
    "试跑",
    "跑个",
    "小任务",
    "单卡",
    "单元测试",
    "smoke",
    "调参",
    "玩一下",
    "验证一下",
    "hello",
)


def parse_task(text: str) -> Want:
    """Extract what can be extracted, and leave the rest unset.

    Every field stays ``None`` when the sentence does not say — guessing a card count
    would silently narrow the fleet, and guessing a memory floor would silently rule
    out machines that can run the job.
    """
    want = Want(note=text.strip())
    low = text.lower()

    if m := _CARDS.search(text):
        raw = m.group("n")
        want.cards = int(raw) if raw.isdigit() else _CN_NUM.get(raw)
    elif _ONE_CARD.search(text):
        want.cards = 1

    if m := _MEM.search(text):
        want.min_mem_gib = _to_gib(m.group("n"), m.group("u"))
    elif m := _MEM_AFTER.search(text):
        want.min_mem_gib = _to_gib(m.group("n"), m.group("u"))

    if m := _HOURS.search(text):
        want.hours = float(m.group("n"))
    elif _HALF_HOUR.search(text):
        want.hours = 0.5

    # Longest match first: `910b` must win over `910`, and dict order is not a
    # contract. Without this, "要 910B" matched the bare `910` entry and then looked
    # for `910` anywhere in the host description, which also matches `910B` — right by
    # accident there, wrong the moment someone asks for the older model specifically.
    for word in sorted(_FAMILY_WORDS, key=len, reverse=True):
        if word in low:
            want.family = _FAMILY_WORDS[word]
            break

    # Exclusive wins ties: mistaking an exclusive job for a shared one puts it on a
    # noisy node, while the reverse only leaves out some candidates.
    #
    # "N 卡" with N >= 2 on its own reads as parallel work — nobody asks for eight
    # cards to run a one-liner, and that is the shape most requests take.
    if want.cards is not None and want.cards >= 2:
        want.exclusive = True
    elif any(h in low for h in _EXCLUSIVE_HINTS):
        want.exclusive = True
    elif any(h in low for h in _SHARED_HINTS):
        want.exclusive = False
    else:
        want.exclusive = None  # undecided; the caller asks
    return want


def _to_gib(number: str, unit: str | None) -> float:
    value = float(number)
    unit = (unit or "G").upper()
    if unit == "M":
        return value / 1024
    if unit == "T":
        return value * 1024
    return value


# --- stage 2: rejection -------------------------------------------------------


@dataclass
class Rejected:
    """Why a host did not make it, in a form a person can act on."""

    host: str
    reason: str


def reject(host: HostView, want: Want) -> str | None:
    """``None`` if the host can take the job, else the reason it cannot."""
    accel = host.accel
    if host.verdict in ("down", "unknown"):
        # Unknown is not free. A card we could not read might be mid-job.
        return "无法探测" if host.verdict == "unknown" else "连不上"
    if accel is None or not accel.total:
        return "没有加速卡"

    console = [c for c in host.accel.cards if c.busy is False]
    if want.family and not _family_matches(host, want.family):
        return f"型号不符（是 {host.card_name or accel.label}）"

    need = want.cards or 1
    if len(console) < need:
        return f"只有 {len(console)} 张卡空闲，需要 {need}"

    if want.min_mem_gib:
        # A card that reports no memory cannot be checked, so it is not a reason to
        # reject — it is a reason to rank lower and say so.
        too_small = [c for c in console if c.mem_total and c.mem_total / 1024 < want.min_mem_gib]
        if len(too_small) == len(console):
            biggest = max((c.mem_total or 0) for c in console) / 1024
            return f"显存最大 {biggest:.0f}G，需要 {want.min_mem_gib:.0f}G"

    load = host.node.load_per_cpu
    if load is not None and load >= BUSY_LOAD_PER_CORE:
        return f"CPU 负载 {load:.1f}/核，机器在忙"
    return None


def _family_matches(host: HostView, want: str) -> bool:
    accel = host.accel
    if accel is None:
        return False
    hay = f"{accel.family} {accel.label} {host.card_name or ''}".lower()
    for card in accel.cards:
        if card.name:
            hay += " " + card.name.lower()
    return want in hay


# --- stage 3: ranking ---------------------------------------------------------


@dataclass
class Candidate:
    host: HostView
    score: float
    reasons: list[str] = field(default_factory=list)
    #: Cards the job would take, when it needs fewer than the host has free.
    free_cards: int = 0
    total_cards: int = 0
    per_card_gib: float | None = None

    @property
    def name(self) -> str:
        return self.host.name

    @property
    def occupancy(self) -> float:
        """Fraction of the node's cards that are free."""
        return self.free_cards / self.total_cards if self.total_cards else 0.0


def score(host: HostView, want: Want) -> Candidate:
    """Rank a host that survived :func:`reject`.

    Exclusive jobs weight *how much of the node is free*, because a neighbour shares
    the interconnect with them. Shared jobs weight the opposite: they are the ones
    that should land on a node somebody is already using, leaving whole nodes for the
    jobs that need them whole.
    """
    accel = host.accel
    cards = [c for c in accel.cards if c.busy is False]
    total = accel.total
    free = len(cards)
    exclusive = want.exclusive is not False
    points = 0.0
    reasons: list[str] = []

    occupancy = free / total if total else 0.0
    need = want.cards or 1
    if exclusive:
        # Whole-node first, as a *tier* rather than a coefficient. A node with even one
        # card in someone else's hands is a node whose traffic this job shares, and no
        # amount of spare capacity on it makes up for that — a 7/8-free node must not
        # outrank an 8/8 one. Scale within the tier stays smooth so partially-free
        # nodes still order sensibly among themselves.
        whole = occupancy >= 1.0
        points += 50.0 if whole else 20.0 * occupancy
        if whole:
            reasons.append("整机空闲")
            # Among whole-free nodes, prefer the one that just fits: an 8-card node
            # for a 4-card job is capacity somebody else may need whole later. This is
            # a tie-break *within* the tier, so it can never lift a shared node up.
            if total > need:
                spare = total - need
                points -= min(15.0, 5.0 * spare)
                reasons.append(f"{total} 卡机跑 {need} 卡，余 {spare} 张")
        elif occupancy >= 0.5:
            reasons.append(f"节点占用过半（{free}/{total} 空闲），有邻居")
        else:
            reasons.append(f"同节点有他人任务（{free}/{total} 空闲）")
    else:
        # Prefer a node that is already in use, so pristine nodes stay pristine.
        points += 40.0 if free < total else 20.0
        reasons.append("与人共享" if free < total else "整机空闲（可能被留给大任务）")

    per_card: float | None = None
    sized = [c.mem_total for c in cards if c.mem_total]
    if sized:
        per_card = max(sized) / 1024
        if want.min_mem_gib:
            headroom = per_card - want.min_mem_gib
            points += min(20.0, max(0.0, headroom))
            reasons.append(f">= {want.min_mem_gib:.0f}G 需求，单卡 {per_card:.0f}G")
        else:
            points += 10.0
            reasons.append(f"单卡 {per_card:.0f}G")
    elif want.min_mem_gib:
        # Unknown memory is a real risk for a job that stated a floor; it should not
        # be rejected, but it must not outrank a host we could verify.
        points -= 10.0
        reasons.append("显存未知")

    load = host.node.load_per_cpu
    if load is not None:
        # Steep on purpose. `BUSY_LOAD_PER_CORE` already rejects the worst nodes, so
        # what is left is a grey band where "quiet enough" matters more than any
        # capacity tie-break: a node running at 0.6/core will make an interactive job
        # feel slow long before it runs out of cards. 36 points at the rejection
        # threshold, which outweighs the 15-point just-fits bonus.
        points -= min(36.0, load * 60.0)
        if load > 0.3:
            reasons.append(f"CPU 负载 {load:.1f}/核")

    temps = [c.temp for c in cards if c.temp is not None]
    if temps:
        hot = max(temps)
        if hot >= 60:
            points -= min(10.0, (hot - 60) / 2)
            reasons.append(f"温度 {hot:.0f}C")

    return Candidate(
        host=host,
        score=round(points, 1),
        reasons=reasons,
        free_cards=free,
        total_cards=total,
        per_card_gib=per_card,
    )


@dataclass
class Advice:
    want: Want
    candidates: list[Candidate] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    explanation: str | None = None

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def shortfall(self) -> str:
        """Why nothing fit — the difference between "wait" and "ask differently"."""
        if not self.rejected:
            return "机队里没有任何机器"
        counts: dict[str, int] = {}
        for r in self.rejected:
            key = r.reason.split("（")[0]
            counts[key] = counts.get(key, 0) + 1
        parts = [f"{k} {v} 台" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
        return "；".join(parts[:4])


def advise(hosts: list[HostView], want: Want, *, limit: int = 5) -> Advice:
    """Filter, rank, and keep the best ``limit``."""
    result = Advice(want=want)
    survivors: list[Candidate] = []
    for host in hosts:
        why = reject(host, want)
        if why is None:
            survivors.append(score(host, want))
        else:
            result.rejected.append(Rejected(host=host.name, reason=why))
    survivors.sort(key=lambda c: (-c.score, c.name))
    result.candidates = survivors[:limit]
    return result
