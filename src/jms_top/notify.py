"""Compose the message you send to a compute group, and send it to Feishu.

This is not a notification channel. It is how you say, in a room where people can
answer back, *I intend to take these cards for this long* — or *I need these cards,
here is whose job is holding them*. Two shapes, because those are two different
conversations:

``claim``
    The host is free enough. Announce the intention, so anyone planning on it knows.

``negotiate``
    The host is occupied. Name the processes that would have to stop, with enough
    detail that the owner recognises their own job — a pid alone is anonymous.

The tool never kills anything. It produces the message and, if asked, the command to
run; the decision to stop someone else's work belongs to the people in the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .families import Card, Proc

#: How long the message says you will hold the cards, when the task did not say.
DEFAULT_HOURS = 4.0


@dataclass
class Group:
    """One room to send to. There is no host→room mapping on purpose."""

    name: str
    webhook: str
    secret: str = ""


@dataclass
class Draft:
    """A message, ready to be looked at before it is looked at by anyone else."""

    title: str
    body: str
    kind: str  # "claim" | "negotiate"
    host: str
    #: Processes that would have to stop, for a negotiate draft.
    blockers: list[Proc] = field(default_factory=list)

    def text(self) -> str:
        return f"{self.title}\n{self.body}".strip()


def _fmt_proc(proc: Proc, cards_held: int) -> str:
    """One blocker, described so its owner recognises it.

    The vendor tools report a bare executable (`python3`), which identifies nothing in
    a negotiation. The command line, the user and the run time come from a follow-up
    `/proc` read; when that is missing this degrades to what is known rather than
    inventing a description.
    """
    bits: list[str] = []
    bits.append(f"pid {proc.pid}" if proc.pid is not None else "pid ?")
    if proc.user:
        bits.append(proc.user)
    if proc.elapsed:
        bits.append(f"已跑 {proc.elapsed}")
    head = "  ".join(bits)
    task = proc.cmdline or proc.name or "?"
    if len(task) > 96:
        task = task[:93] + "…"
    held = f"{cards_held} 卡" if cards_held > 1 else "1 卡"
    mem = f"，{proc.mem_mib / 1024:.1f}G/卡" if proc.mem_mib else ""
    return f"  · {head}\n     {task}\n     占用 {held}{mem}"


def _holders(cards: list[Card]) -> dict[int, tuple[Proc, int]]:
    """``{pid: (proc, cards held)}`` for every process occupying a card."""
    out: dict[int, tuple[Proc, int]] = {}
    for card in cards:
        for proc in card.procs:
            if proc.pid is None:
                continue
            prev = out.get(proc.pid)
            if prev is None:
                out[proc.pid] = (proc, 1)
            else:
                out[proc.pid] = (prev[0], prev[1] + 1)
    return out


def compose(
    *,
    host,
    want,
    hours: float | None = None,
    requester: str = "",
    now: datetime | None = None,
) -> Draft:
    """Build the message for one host.

    Picks ``claim`` or ``negotiate`` from the host's own state: if every card the job
    would use is already free, there is nothing to ask for.
    """
    accel = host.accel
    need = want.cards or 1
    total = accel.total if accel else 0
    free_cards = [c for c in (accel.cards if accel else []) if c.busy is False]
    model = host.card_name or (accel.label if accel else "?")
    per_card = ""
    sized = [c.mem_total for c in free_cards if c.mem_total]
    if sized:
        per_card = f"，单卡 {max(sized) / 1024:.0f}G"

    hold = hours if hours is not None else (want.hours or DEFAULT_HOURS)
    start = now or datetime.now()
    end = start + timedelta(hours=hold)
    when = f"{start:%m-%d %H:%M} → {end:%m-%d %H:%M}"
    who = f"@{requester} " if requester else ""
    mode = (
        "整机独占" if (want.exclusive and need >= total) else ("独占" if want.exclusive else "共享")
    )

    # `need` defaults to 1 when the task was not described, which would otherwise
    # produce "需要 1 张卡" for a job nobody has defined yet — a message asking someone
    # to give up a card on behalf of no stated task.
    described = bool(want.note.strip()) or want.cards is not None

    if len(free_cards) >= need:
        body = "\n".join(
            [
                f"任务：{want.note or '（未填写）'}",
                f"占用：{host.name} 的 {need} 张卡（{mode}）",
                f"时长：预计 {hold:g} 小时，{when}",
                f"配置：{model}{per_card}",
                "",
                f"申请：{who}{host.name} {need}×{model}",
                "如有冲突请在此回复，我会调整。",
            ]
        )
        return Draft(
            title=f"【算力占用】{host.name} {need} 卡",
            body=body,
            kind="claim",
            host=host.name,
        )

    # Not enough free: name whoever is in the way. `need` defaults to 1 when nothing
    # was described, which would otherwise ask somebody to free a card on behalf of a
    # job that does not exist yet — say so instead of pretending.
    described = bool(want.note.strip()) or want.cards is not None
    if not described:
        title = f"【算力协商】{host.name} 资源不足"
        task_line = "（未描述任务——先用 `a` 说明要跑什么再发）"
    else:
        title = f"【算力协商】{host.name} 需要 {need} 卡，当前被占用"
        task_line = want.note
    holders = _holders(list(accel.cards))
    lines = list(holders.values())
    described = "\n".join(_fmt_proc(p, n) for p, n in lines) if lines else "  （进程列表不可用）"
    body = "\n".join(
        [
            f"任务：{task_line}",
            f"需求：{host.name} 的 {need} 张卡，当前只有 {len(free_cards)} 张空闲",
            f"含：{model}{per_card}",
            "",
            "占用情况：",
            described,
            "",
            # No deadline derived from the requester's own run time: how long *they*
            # need the cards says nothing about when the current job finishes, and
            # "please free these by 17:53" invited by a 4-hour default is nonsense.
            f"请求：能否释放 {need - len(free_cards)} 张？"
            "若仍在训练，请回复预计结束时间，我再决定是否另找机器。",
        ]
    )
    return Draft(
        title=title,
        body=body,
        kind="negotiate",
        host=host.name,
        blockers=[p for p, _ in lines],
    )


def kill_hint(draft: Draft) -> str:
    """The command that would free the cards, for a person to run if they decide to.

    Not executed by anything here. Handing someone a command to review is a different
    act from running it on their behalf.
    """
    pids = [str(p.pid) for p in draft.blockers if p.pid is not None]
    if not pids:
        return ""
    return f"ssh <host> 'kill -TERM {' '.join(pids)}'   # 确认后再执行"


# --- transport ----------------------------------------------------------------


def post(webhook: str, text: str, *, secret: str = "", timeout: float = 10.0) -> tuple[bool, str]:
    """Send one text message to a Feishu bot. Never raises.

    Returns ``(ok, detail)``. A webhook that is down must not take the command down
    with it: the draft is still on screen, and the user can copy it.
    """
    import base64
    import hashlib
    import hmac
    import time

    payload: dict = {"msg_type": "text", "content": {"text": text}}
    if secret:
        stamp = str(int(time.time()))
        key = f"{stamp}\n{secret}".encode()
        sign = base64.b64encode(hmac.new(key, b"", hashlib.sha256).digest()).decode()
        payload["timestamp"] = stamp
        payload["sign"] = sign

    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(webhook, json=payload)
    except Exception as exc:  # noqa: BLE001 - any transport failure is the same result
        return False, f"{type(exc).__name__}: {exc}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001 - a 200 that is not JSON still counts as sent
        return True, "sent"
    code = data.get("code", data.get("StatusCode", 0))
    if code not in (0, None):
        return False, f"feishu code {code}: {data.get('msg', data)}"
    return True, "sent"
