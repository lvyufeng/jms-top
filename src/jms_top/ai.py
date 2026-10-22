"""Ask a model to explain a recommendation — and nothing else.

The contract is deliberately narrow. The model receives a handful of candidates whose
facts were computed locally, and returns prose. It cannot add a machine, remove one, or
change an ordering that the ranking already decided: a language model counting free
cards in `FFFFBBBB` is unreliable, and acting on that unreliability means telling
someone a machine has room that it does not have.

Every failure path — no configuration, a timeout, a malformed reply, an endpoint that
answers 500 — degrades to "no explanation". The recommendation itself never depends on
this module being reachable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .advise import Advice

#: Kept short on purpose: the model is writing one or two sentences about a list it
#: has been handed, not reasoning about a fleet.
_SYSTEM = (
    "你在帮我从一批候选机器里挑一台跑任务。只做两件事："
    "一是从候选里选最合适的，二是用一到两句话说明为什么。"
    "候选的顺序和字段都是我已经算好的，不要重新计算、不要推测没有给出的信息、"
    "不要提到候选之外的机器。如果候选里明显有更合适的一台，可以指出它并说明理由，"
    "但只能从给出的 host 名字里选。"
)


@dataclass
class Explanation:
    text: str
    pick: str | None = None
    ok: bool = True


def _payload(advice: Advice) -> str:
    """The candidates, with the facts the model is allowed to mention."""
    want = advice.want
    rows = []
    for c in advice.candidates:
        rows.append(
            {
                "host": c.name,
                "model": c.host.accel.card_name if c.host.accel else None,
                "free_cards": c.free_cards,
                "total_cards": c.total_cards,
                "vram_per_card_gib": c.per_card_gib,
                "cpu_load_per_core": (
                    round(c.host.node.load_per_cpu, 2)
                    if c.host.node.load_per_cpu is not None
                    else None
                ),
                "rank": advice.candidates.index(c) + 1,
                "notes": c.reasons,
            }
        )
    return json.dumps(
        {
            "task": want.note,
            "needs": {
                "cards": want.cards,
                "min_vram_gib": want.min_mem_gib,
                "exclusive": want.exclusive,
                "hours": want.hours,
            },
            "candidates": rows,
        },
        ensure_ascii=False,
        indent=2,
    )


def explain(
    advice: Advice, *, base_url: str, api_key: str, model: str, timeout: float = 20.0
) -> Explanation:
    """One OpenAI-compatible chat completion. Never raises.

    Returns ``ok=False`` with the reason in ``text`` when the call did not produce a
    usable answer, so a caller can decide whether to show it at all.
    """
    if not advice.candidates:
        return Explanation(text="", ok=False)
    if not base_url or not api_key:
        return Explanation(text="", ok=False)

    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": _payload(advice)},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400,
                },
            )
        if resp.status_code != 200:
            return Explanation(text=f"AI 返回 HTTP {resp.status_code}", ok=False)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - any failure is "no explanation"
        return Explanation(text=f"AI 调用失败: {type(exc).__name__}", ok=False)

    text = (content or "").strip()
    if not text:
        return Explanation(text="AI 返回了空内容", ok=False)

    # The pick is only accepted if it is one of the candidates: a model that names a
    # machine outside the list is not allowed to introduce it.
    names = {c.name for c in advice.candidates}
    pick = next((n for n in names if n in text), None)
    return Explanation(text=text, pick=pick, ok=True)
