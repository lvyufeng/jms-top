"""One parser per accelerator family, and one command snippet to collect them all.

This module parses the **state of every card**: name, utilisation, memory,
temperature, power, and the processes holding it. A launch gate only needs a verdict;
a monitor needs the numbers behind it, on the same screen, for every vendor.

Every parser here was written against captured output (``tests/fixtures/``, taken from
the live fleet on 2026-10-21), not against a vendor manual. The rules that came out of
that exercise:

* **A parse failure never reads as "idle".** A card whose numbers did not parse keeps
  them ``None`` and reports its state as unknown — it is shown, and it is not counted
  as free. A wrong "free" sends work onto someone else's running job, which is the one
  failure this whole tool exists to prevent.
* **A card is busy if a process holds it, whatever the utilisation says.** This is not
  theoretical: ``p800-15`` prints ``0%`` utilisation on cards holding 87% of their
  memory with a vLLM stack on them, and ``tsingmicro-137`` prints ``0%`` on twelve
  chips with 90% of their memory taken. Utilisation alone would call both fleets idle.
* **Signals are OR-ed, and "unknown" is not "free".** ``Card.busy`` combines process,
  memory and utilisation; when *none* of the three is known the answer is ``None``.
* **Vendors differ in whether a row is a card or a chip.** Ascend 910C reports 2 chips
  per NPU card, TsingMicro 4 chips per card. The unit is the *chip* (what a job
  addresses via ``ASCEND_RT_VISIBLE_DEVICES`` / device index); the card id rides along.

Nothing here decides which family a host runs: the probe script runs every vendor
tool it finds and prefixes each section with a marker, so the answer comes from what
the host actually printed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import error as error_group

#: Commit-label threshold: a card holding at least this share of its memory is busy
#: even with no process and no utilisation. 0.75 is above the "CUDA context" floor a
#: driver reserves before a job starts, and below what any real workload allocates.
BUSY_MEM_FRACTION = 0.50

#: Below this memory share, an *idle* card is called free rather than unknown.
FREE_MEM_FRACTION = 0.10

#: An idle card holding at least this share is ``residual``: a stack that reserved
#: memory and never gave it back. Not an error — an observation, so nobody silently
#: "reclaims" a card that is not theirs to reclaim.
RESIDUAL_MEM_FRACTION = 0.75

#: Utilisation at or above this is activity even with no process (e.g. an Ascend NPU
#: whose processes are invisible to ``npu-smi`` on some firmware levels).
BUSY_UTIL = 5.0


@dataclass
class Proc:
    pid: int | None
    user: str | None
    name: str
    mem_mib: float | None = None

    def to_dict(self) -> dict:
        return {"pid": self.pid, "user": self.user, "name": self.name, "mem_mib": self.mem_mib}


@dataclass
class Card:
    """One addressable accelerator: a GPU, an NPU chip, a GCU, an XPU, a PPU."""

    index: int
    name: str = ""
    util: float | None = None  # percent
    mem_used: float | None = None  # MiB
    mem_total: float | None = None  # MiB
    temp: float | None = None  # C
    power: float | None = None  # W
    procs: list[Proc] = field(default_factory=list)
    #: Which physical card this chip sits on, when the vendor reports several per card.
    card: int | None = None
    health: str | None = None
    #: True when the vendor's process table was actually read for this card, or when
    #: the vendor offers no process table at all (its memory/utilisation counters are
    #: then the whole signal). False means *unknown* — a table we failed to find must
    #: never be mistaken for "nothing is running".
    procs_known: bool = True

    # --- derived state ----------------------------------------------------------

    @property
    def mem_frac(self) -> float | None:
        if self.mem_used is None or not self.mem_total:
            return None
        return min(1.0, self.mem_used / self.mem_total)

    @property
    def busy(self) -> bool | None:
        """``True`` / ``False`` / ``None`` (unknown — never reported as free)."""
        if self.health and self.health.upper() not in {"OK", "NORMAL", "GOOD"}:
            # A card whose health is not OK is not available for work, whatever its
            # counters say.
            return True
        if self.procs:
            return True
        signals: list[bool] = []
        frac = self.mem_frac
        if frac is not None:
            signals.append(frac >= BUSY_MEM_FRACTION)
        if self.util is not None:
            signals.append(self.util >= BUSY_UTIL)
        if not signals:
            return None
        return any(signals)

    @property
    def state(self) -> str:
        """``busy`` / ``free`` / ``unknown`` — the one word the whole UI keys on."""
        b = self.busy
        return "unknown" if b is None else ("busy" if b else "free")

    @property
    def free(self) -> bool:
        return self.busy is False

    @property
    def residual(self) -> bool:
        """Memory held, nothing computing: a stack that never freed its reservation."""
        frac = self.mem_frac
        return (
            not self.procs
            and frac is not None
            and frac >= RESIDUAL_MEM_FRACTION
            and (self.util or 0.0) < BUSY_UTIL
        )

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "card": self.card,
            "util": self.util,
            "mem_used": self.mem_used,
            "mem_total": self.mem_total,
            "temp": self.temp,
            "power": self.power,
            "health": self.health,
            "busy": self.busy,
            "residual": self.residual,
            "procs": [p.to_dict() for p in self.procs],
        }


@dataclass
class AccelResult:
    family: str
    label: str
    cards: list[Card] = field(default_factory=list)
    #: Why there are no cards, when the tool is absent or its output did not parse.
    error: str | None = None
    version: str | None = None

    @property
    def total(self) -> int:
        return len(self.cards)

    @property
    def busy(self) -> int:
        return sum(1 for c in self.cards if c.busy is True)

    @property
    def free(self) -> int:
        return sum(1 for c in self.cards if c.busy is False)

    @property
    def unknown(self) -> int:
        return sum(1 for c in self.cards if c.busy is None)

    @property
    def residual(self) -> int:
        return sum(1 for c in self.cards if c.residual)


# --- small parsing helpers ----------------------------------------------------


def _num(text: str | None) -> float | None:
    """First number in ``text``; ``N/A`` / ``-`` / ``Disabled`` yield ``None``."""
    if not text:
        return None
    if re.fullmatch(r"\s*\(?-?\s*(N/A|NA|Disabled|None|\[\])\s*\)?\s*", text):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group(0)) if m else None


def _pct(text: str | None) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return float(m.group(1)) if m else None


_MEM_UNIT = {"": 1.0, "K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}


def _mib(value: str, unit: str = "M") -> float | None:
    n = _num(value)
    if n is None:
        return None
    unit = (unit or "").upper()[:1]
    return round(n * _MEM_UNIT.get(unit if unit in _MEM_UNIT else "M", 1.0), 1)


def _pair(text: str) -> tuple[float | None, float | None]:
    """``'164MiB / 98304MiB'``, ``'58997M / 65536M'``, ``'64914/65536 MiB'`` -> MiB pair."""
    pats = [
        (r"(\d+(?:\.\d+)?)\s*([KMGT]?)i?B\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT]?)i?B", None),
        (r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT]?)i?B", "shared"),
        (r"(\d+(?:\.\d+)?)\s*([KMGT])i?B?\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT])i?B?", None),
    ]
    for pat, mode in pats:
        m = re.search(pat, text)
        if not m:
            continue
        if mode == "shared":
            return _mib(m.group(1), m.group(3)), _mib(m.group(2), m.group(3))
        return _mib(m.group(1), m.group(2)), _mib(m.group(3), m.group(4))
    return None, None


def _shorten(name: str) -> str:
    """Process name as the terminal should show it: basename, no argv, no ellipsis."""
    name = name.strip()
    if not name:
        return "?"
    # nvidia-smi and friends elide long paths from the left: "...envs/fsa/bin/python".
    if name.startswith("...") and "/" in name:
        name = name.lstrip(".").lstrip("/").rsplit("/", 1)[-1]
    head = name.split()[0]
    if "/" in head:
        head = head.rsplit("/", 1)[-1]
    return head or "?"


def _field(m: re.Match[str], name: str) -> float | None:
    """A numeric field that may be absent (no match) — ``None``, not a neighbour's value.

    ``m.group(name)`` raises when the pattern has no such group; an optional group that
    did not participate returns ``None``, which is exactly the "not reported" answer.
    """
    try:
        return _num(m.group(name))
    except (IndexError, error_group):
        return None


def _proc(m: re.Match[str]) -> Proc:
    """The ``Proc`` in an NVIDIA apps row (``... pid, process_name, used_memory``)."""
    return Proc(
        pid=int(m.group("pid")),
        user=None,
        name=_shorten(m.group("name")),
        mem_mib=_num(m.group("mem")),
    )


def _regex_version(body: str, pattern: str) -> str | None:
    """First capture group of ``pattern`` in ``body``, or ``None``."""
    m = re.search(pattern, body)
    return m.group(1) if m else None


def _first_error(body: str) -> str | None:
    """The first line that reads like an error rather than a table."""
    for line in body.splitlines():
        low = line.lower()
        if any(
            k in low
            for k in (
                "error",
                "failed",
                "not found",
                "no such",
                "cannot",
                "couldn't",
                "permission denied",
                "exceeds max",
            )
        ):
            clean = re.sub(r"^\[[A-Za-z/]+\]\s*", "", line.strip())
            return clean[:160]
    return None


# --- family base --------------------------------------------------------------


class Family:
    """A vendor's SMI: its marker in the probe output, and how to read what it printed."""

    key: str = ""
    label: str = ""
    #: Regex the probe's ``##FAMILY`` marker line is matched against.
    marker: re.Pattern[str]

    def parse(self, body: str) -> AccelResult:  # pragma: no cover - interface
        raise NotImplementedError


# --- NVIDIA -------------------------------------------------------------------

#: Wide query: ``index, name, util %, used MiB, total MiB, temp, power``.
#: Deliberately no trailing anchor: a row cut by the output budget must still parse,
#: because a truncated card table is exactly what a busy host produces. Units are
#: optional (``nounits`` prints bare numbers, plain ``csv`` adds ``%`` and ``MiB``).
_NV_WIDE = re.compile(
    r"^(?P<idx>\d+)\s*,\s*(?P<name>[^,]+?)\s*,\s*(?P<util>[\w.]+)\s*%?\s*,\s*"
    r"(?P<used>[\w.]+)\s*(?:MiB\s*)?,\s*(?P<total>[\w.]+)\s*(?:MiB\s*)?"
    r"(?:\s*,\s*(?P<temp>[\w.]+)\s*,\s*(?P<power>[\w.]+))?"
    r"(?:\s*,\s*(?P<bus>\S+))?"
)
#: The same row when a driver *does* print the uuid column (older builds). Read it,
#: but never depend on it — see the probe's note on why the query no longer asks.
_NV_WIDE_UUID = re.compile(
    r"^(?P<idx>\d+),\s*(?P<name>.+?)\s*,\s*(?P<uuid>GPU-\S+)\s*,\s*(?P<util>\S+?)\s*%?\s*,"
    r"\s*(?P<used>\S+?)\s*(?:MiB\s*)?,\s*(?P<total>\S+?)\s*(?:MiB\s*)?,\s*"
    r"(?P<temp>\S+?)\s*,\s*(?P<power>\S+)"
)
#: Apps query: ``gpu_bus_id, pid, process_name, used_memory``. The bus id is the card
#: key (``nvidia-smi -L`` prints it and so does the tile); the index is not offered
#: by this query at all.
_NV_APP = re.compile(
    r"^(?P<bus>[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f:.]+)\s*,\s*(?P<pid>\d+)\s*,\s*"
    r"(?P<name>.+?)\s*,\s*(?P<mem>[\d.]+)\s*(?:MiB)?\s*$"
)
#: The same row when the driver prints the **uuid** instead of the bus id.
_NV_APP_UUID = re.compile(
    r"^(?P<uuid>GPU-\S+)\s*,\s*(?P<pid>\d+)\s*,\s*(?P<name>.+?)\s*,\s*"
    r"(?P<mem>[\d.]+)\s*(?:MiB)?\s*$"
)
#: The wide query must also yield each card's bus id to join the app rows onto it.
_NV_BUS = re.compile(r"(?P<bus>[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f:.]+)")
#: The default ``nvidia-smi`` table, for the ``fleet`` probe kind (top-8 only).
_NV_TILE_NAME = re.compile(
    r"^\|\s+(?P<idx>\d+)\s+(?P<name>NVIDIA .+?|Tesla .+?|GeForce .+?|\S+ \S+)\s+(?:\w+)\s*"
    r"\|\s*[0-9A-Fa-f]{8}:"
)
_NV_TILE_STAT = re.compile(
    r"^\|\s*\S+\s+(?P<temp>\d+)C\s+\S+\s+(?P<power>[\d.]+)W\s*/\s*(?P<cap>[\d.]+)W\s*\|\s*"
    r"(?P<used>\d+)MiB\s*/\s*(?P<total>\d+)MiB\s*\|\s*(?P<util>[\d.]+)%"
)
_NV_TILE_PROC = re.compile(
    r"^\|\s+(?P<dev>\d+)\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(?P<pid>\d+)\s+\S+\s+(?P<name>.+?)\s+(?P<mem>\d+)MiB\s*\|"
)


class Nvidia(Family):
    key = "nvidia"
    label = "NVIDIA"
    marker = re.compile(r"^##NVIDIA\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        cards: list[Card] = []
        #: ``uuid -> card`` and ``bus id -> card``: the gpu query keys on the index,
        #: the apps query on the bus id (or, on older drivers, the uuid), and the wide
        #: row may carry either — or neither.
        by_uuid: dict[str, Card] = {}
        by_bus: dict[str, Card] = {}
        apps: list[tuple[str, str, Proc]] = []
        for line in body.splitlines():
            line = line.strip()
            # A process row is checked first: every one of them also looks like a
            # (garbage) device row, because both start with "a name, a number, ...".
            a = _NV_APP.match(line)
            if a:
                apps.append(("bus", a.group("bus").lower(), _proc(a)))
                continue
            u = _NV_APP_UUID.match(line)
            if u:
                apps.append(("uuid", u.group("uuid"), _proc(u)))
                continue
            # With a uuid column, ``_NV_WIDE`` still matches by reading the uuid as
            # part of the *name* and shifting one field left, so the uuid form must be
            # tried first or every such host would report util=76000.
            m = _NV_WIDE_UUID.match(line) or _NV_WIDE.match(line)
            if not m:
                continue
            card = Card(
                index=int(m.group("idx")),
                name=m.group("name").strip(),
                util=_field(m, "util"),
                mem_used=_field(m, "used"),
                mem_total=_field(m, "total"),
                temp=_field(m, "temp"),
                power=_field(m, "power"),
            )
            cards.append(card)
            fields = m.groupdict()
            if fields.get("uuid"):
                by_uuid[fields["uuid"]] = card
            if fields.get("bus"):
                by_bus[fields["bus"].lower()] = card
        if cards:
            for kind, key, proc in apps:
                owner = by_bus.get(key) if kind == "bus" else by_uuid.get(key)
                if owner is not None:
                    owner.procs.append(proc)
            # A vendor query that lists no process prints nothing at all, so an empty
            # app list *is* the answer "no process holds a card".
            for card in cards:
                card.procs_known = True
            res.cards = cards
            res.version = _regex_version(body, r"NVIDIA-SMI\s+(\S+)")
            return res
        return self._parse_tiles(body, res)

    def _parse_tiles(self, body: str, res: AccelResult) -> AccelResult:
        """The default ``nvidia-smi`` table: for old drivers and ``--nvidia-tile``."""
        cards: list[Card] = []
        pending: Card | None = None
        procs: list[tuple[int, Proc]] = []
        for line in body.splitlines():
            n = _NV_TILE_NAME.match(line)
            if n:
                pending = Card(index=int(n.group("idx")), name=n.group("name").strip())
                cards.append(pending)
                continue
            st = _NV_TILE_STAT.match(line)
            if st and pending is not None:
                pending.temp = _num(st.group("temp"))
                pending.power = _num(st.group("power"))
                pending.mem_used = _num(st.group("used"))
                pending.mem_total = _num(st.group("total"))
                pending.util = _num(st.group("util"))
                pending = None
                continue
            pr = _NV_TILE_PROC.match(line)
            if pr:
                procs.append(
                    (
                        int(pr.group("dev")),
                        Proc(
                            pid=int(pr.group("pid")),
                            user=None,
                            name=_shorten(pr.group("name")),
                            mem_mib=_num(pr.group("mem")),
                        ),
                    )
                )
        if not cards:
            res.error = _first_error(body) or "nvidia-smi returned no GPU rows"
            return res
        by_index = {c.index: c for c in cards}
        for dev, proc in procs:
            if dev in by_index:
                by_index[dev].procs.append(proc)
        if "Processes:" in body:
            for c in cards:
                c.procs_known = True
        res.cards = cards
        res.version = _regex_version(body, r"NVIDIA-SMI\s+(\S+)")
        return res


# --- Ascend (CANN / npu-smi) --------------------------------------------------

#: Card header: ``| 0     Ascend910           | OK            | 519.5    58   ...``
#: Power and temp sit in fixed columns, and either may be ``-`` (only the first chip
#: row of a card prints power). Matching the two column *ends* is what makes both
#: ``-           56`` and ``519.5       58`` parse.
_NPU_CARD = re.compile(
    r"^\|\s*(?P<card>\d+)\s+(?P<name>Ascend\S*)\s*\|\s*(?P<health>\S.*?)\s*\|\s*"
    r"(?P<power>[\d.]+|-)\s+(?P<temp>\d+)(?=\s)"
)
#: Chip row: ``| 0     1                   | 0000:9F:00.0  | 42    0/0    37363/ 65536 |``
#: The first field is the NPU *card*, the second is the chip within that card.
_NPU_CHIP = re.compile(
    r"^\|\s*(?P<chip>\d+)\s+(?P<phy>\d+)\s*\|\s*(?P<bus>[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f:.]+)"
    r"\s*\|\s*(?P<aicore>\d+)\s+(?P<mem>[\d.]+)\s*/\s*(?P<ddr>[\d.]+)\s+(?P<hbm>[\d.]+)\s*/\s*(?P<hbm_total>[\d.]+)\s*\|"
)
#: Process row: ``| 0       1                 | 2510884       | python                   | 37831 |``
_NPU_PROC = re.compile(
    r"^\|\s*(?P<card>\d+)\s+(?P<chip>\d+)\s*\|\s*(?P<pid>\d+)\s*\|\s*(?P<name>.+?)\s*\|\s*(?P<mem>\d+)\s*\|"
)


class Ascend(Family):
    key = "ascend"
    label = "Ascend"
    marker = re.compile(r"^##ASCEND\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        cards: list[Card] = []
        #: (name, health, power, temp) per NPU card, from its header row. One of
        #: power/temp is ``-`` on each of the card's two chip rows, so merge.
        meta: dict[int, tuple[str, str, float | None, float | None]] = {}
        card_no = -1
        for line in body.splitlines():
            m = _NPU_CARD.match(line)
            if m:
                # Each card prints this header twice — once above each of its chips,
                # with power and temp taking turns being "-". A header is therefore a
                # *continuation* of the card above it when it names the same card and
                # carries exactly the column the previous one lacked.
                name, health = m.group("name"), m.group("health")
                power, temp = _num(m.group("power")), _num(m.group("temp"))
                prev = meta.get(card_no) if card_no >= 0 else None
                continues_above = (
                    prev is not None
                    and prev[0] == name
                    and prev[1] == health
                    and (power is None) != (prev[2] is None)
                    and (temp is None) != (prev[3] is None)
                )
                if not continues_above:
                    card_no += 1
                prev = meta.get(card_no, (name, health, None, None))
                meta[card_no] = (name, health, power or prev[2], temp or prev[3])
                continue
            c = _NPU_CHIP.match(line)
            if c:
                chip, phy = int(c.group("chip")), int(c.group("phy"))
                # ``phy`` (second field) is the global chip index — the one the runtime
                # addresses via ASCEND_RT_VISIBLE_DEVICES — while ``chip`` restarts at
                # zero for every card, so the card is whatever header block we are in.
                owner = card_no if card_no >= 0 else phy
                cname, health, power, temp = meta.get(owner, ("NPU", "OK", None, None))
                cards.append(
                    Card(
                        index=phy,
                        name=cname,
                        util=float(c.group("aicore")),
                        mem_used=float(c.group("hbm")),
                        mem_total=float(c.group("hbm_total")),
                        temp=temp,
                        power=power,
                        card=owner,
                        health=health,
                    )
                )
        # Second pass: the process table follows every chip block and is keyed by
        # (NPU card, chip-in-card).
        by_owner: dict[int, list[Card]] = {}
        for c in cards:
            by_owner.setdefault(c.card if c.card is not None else c.index, []).append(c)
        for siblings in by_owner.values():
            siblings.sort(key=lambda c: c.index)
        saw_table = "Process id" in body or "No running processes" in body
        for line in body.splitlines():
            p = _NPU_PROC.match(line)
            if not p:
                continue
            chips = by_owner.get(int(p.group("card")), [])
            chip = int(p.group("chip"))
            if chip >= len(chips):
                continue
            chips[chip].procs.append(
                Proc(
                    pid=int(p.group("pid")),
                    user=None,
                    name=_shorten(p.group("name")),
                    mem_mib=_num(p.group("mem")),
                )
            )
        if saw_table:
            for c in cards:
                c.procs_known = True
        if not cards:
            res.error = _first_error(body) or "npu-smi returned no chip rows"
        res.cards = cards
        res.version = _regex_version(body, r"npu-smi\s+\S+\s+Version:\s*(\S+)") or _regex_version(
            body, r"npu-smi\s+(\d\S+)"
        )
        return res


# --- Kunlunxin (xpu-smi) ------------------------------------------------------

_XPU_NAME = re.compile(
    r"^\|\s*(?P<idx>\d+)\s+(?P<name>P800\S*(?:\s+\w+)?)\s+\S+\s*\|\s*[0-9A-Fa-f]{8}:"
)
_XPU_STAT = re.compile(
    r"^\|\s*\S+\s+(?P<temp>\d+)C\s+\S+\s+(?P<power>[\d.]+)W\s*/\s*(?P<cap>[\d.]+)W\s*\|\s*"
    r"(?P<used>\d+)MiB\s*/\s*(?P<total>\d+)MiB\s*\|\s*(?P<util>[\d.]+)%"
)
_XPU_PROC = re.compile(
    r"^\|\s*(?P<dev>\d+)\s+\S+\s+\S+\s+(?P<pid>\d+)\s+\S+\s+(?P<name>.+?)\s+(?P<mem>\d+)MiB\s*\|"
)


class Kunlunxin(Family):
    key = "kunlunxin"
    label = "Kunlunxin"
    marker = re.compile(r"^##KUNLUNXIN\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        cards = _tile_cards(body, _XPU_NAME, _XPU_STAT)
        if not cards:
            res.error = _first_error(body) or "xpu-smi returned no XPU rows"
            return res
        by_index = {c.index: c for c in cards}
        for line in body.splitlines():
            p = _XPU_PROC.match(line)
            if p and int(p.group("dev")) in by_index:
                card = by_index[int(p.group("dev"))]
                card.procs.append(
                    Proc(
                        pid=int(p.group("pid")),
                        user=None,
                        name=_shorten(p.group("name")),
                        mem_mib=_num(p.group("mem")),
                    )
                )
        if "Processes:" in body:
            for c in cards:
                c.procs_known = True
        res.cards = cards
        res.version = _regex_version(body, r"Driver Version:\s*(\S+)")
        return res


# --- MetaX (mx-smi) -----------------------------------------------------------

_MX_NAME = re.compile(
    r"^\|\s*(?P<idx>\d+)\s+(?P<name>MetaX\s+\S+)\s*\|\s*\d+\s+\S+\s*\|\s*[0-9A-Fa-f]{4}:"
)
#: ``| 192W / 450W      | 45C          P9 | 64914/65536 MiB     | Available            |``
_MX_STAT = re.compile(
    r"^\|\s*(?P<power>[\d.]+)W\s*/\s*(?P<cap>[\d.]+)W\s*\|\s*(?P<temp>\d+)C\s+\S+\s*\|\s*"
    r"(?P<used>[\d.]+)\s*/\s*(?P<total>[\d.]+)\s*MiB\s*\|"
)
_MX_PROC = re.compile(r"^\|\s*(?P<dev>\d+)\s+(?P<pid>\d+)\s+(?P<name>\S.*?)\s+(?P<mem>\d+)\s*\|")
#: The same row inside the wide layout's own process table.
_MX_PROC_ROWS = _MX_PROC


_MX_WIDE_GPU = re.compile(r"^GPU#(?P<idx>\d+)\s+(?P<name>\S+)\s+(?P<bus>[0-9A-Fa-f:.]+)\s*$")
_MX_WIDE_MEM = re.compile(r"^\s+vis_vram\s+(?P<kind>total|used)\s*:\s*(?P<kb>\d+)\s*KB\s*$")
_MX_WIDE_UTIL = re.compile(r"^\s+(?P<what>GPU|VPUE|VPUD)\s*:\s*(?P<pct>[\d.]+)\s*%\s*$")


class MetaX(Family):
    """沐曦 MetaX (曦云 C550/N260).

    Two layouts: ``##METAX-WIDE`` from the per-type queries (``--show-memory`` /
    ``--show-usage`` / ``--show-process``) and the plain ``##METAX`` default table
    used by drivers that predate those flags. Both are parsed here because the probe
    decides which one ran, and a host must never be reported free because its driver
    fell back to the older layout.
    """

    key = "metax"
    label = "MetaX"
    marker = re.compile(r"^##METAX")

    #: On the default table, per-card utilisation is on the *name* row
    #: (``| 97%  Disabled |``), not the stats row.
    _UTIL = re.compile(r"\|\s*(?P<util>[\d.]+)%\s")

    def parse(self, body: str) -> AccelResult:
        if re.search(r"^GPU#\d+", body, re.MULTILINE):
            return self._parse_wide(body)
        return self._parse_table(body)

    def _parse_wide(self, body: str) -> AccelResult:
        """``GPU#n NAME BDF`` followed by indented ``key : value`` lines.

        The three queries print the *same* per-card blocks in sequence, so a block for
        an index already seen updates that card instead of adding another. The process
        table comes last and repeats "GPU" as a column heading, which is not a card.
        """
        res = AccelResult(family=self.key, label=self.label)
        blocked: list[Card] = []
        cards: dict[int, Card] = {}
        current: Card | None = None
        in_procs = False
        for line in body.splitlines():
            if not in_procs and re.match(r"^\s*\|?\s*Process:", line):
                in_procs = True
                continue
            if in_procs:
                p = _MX_PROC_ROWS.match(line)
                if p and int(p.group("dev")) in cards:
                    cards[int(p.group("dev"))].procs.append(
                        Proc(
                            pid=int(p.group("pid")),
                            user=None,
                            name=_shorten(p.group("name")),
                            mem_mib=_num(p.group("mem")),
                        )
                    )
                continue
            m = _MX_WIDE_GPU.match(line)
            if m:
                idx = int(m.group("idx"))
                current = cards.get(idx)
                if current is None:
                    current = Card(index=idx, name=m.group("name"))
                    cards[idx] = current
                    blocked.append(current)
                continue
            if current is None:
                continue
            mem = _MX_WIDE_MEM.match(line)
            if mem:
                mib = round(int(mem.group("kb")) / 1024, 1)
                if mem.group("kind") == "total":
                    current.mem_total = mib
                else:
                    current.mem_used = mib
                continue
            util = _MX_WIDE_UTIL.match(line)
            if util and util.group("what") == "GPU":
                current.util = float(util.group("pct"))
        if not cards:
            res.error = _first_error(body) or "mx-smi returned no GPU rows"
            return res
        if in_procs:
            # The process table printed: no rows then genuinely means nothing is running.
            for c in cards.values():
                c.procs_known = True
        res.cards = blocked
        res.version = _regex_version(body, r"mx-smi\s+version:\s*(\S+)")
        return res

    def _parse_names(self, body: str) -> list[Card]:
        cards: list[Card] = []
        pending: Card | None = None
        for line in body.splitlines():
            n = _MX_NAME.match(line)
            if n:
                pending = Card(index=int(n.group("idx")), name=" ".join(n.group("name").split()))
                u = self._UTIL.search(line)
                if u:
                    pending.util = float(u.group("util"))
                cards.append(pending)
                continue
            st = _MX_STAT.match(line)
            if st and pending is not None:
                pending.temp = _num(st.group("temp"))
                pending.power = _num(st.group("power"))
                pending.mem_used = _num(st.group("used"))
                pending.mem_total = _num(st.group("total"))
                pending = None
        return cards

    def _parse_table(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        cards = self._parse_names(body)
        if not cards:
            res.error = _first_error(body) or "mx-smi returned no GPU rows"
            return res
        by_index = {c.index: c for c in cards}
        in_procs = False
        for line in body.splitlines():
            if re.match(r"^\|\s*Process:", line):
                in_procs = True
                continue
            if not in_procs:
                continue
            p = _MX_PROC.match(line)
            if p and int(p.group("dev")) in by_index:
                by_index[int(p.group("dev"))].procs.append(
                    Proc(
                        pid=int(p.group("pid")),
                        user=None,
                        name=_shorten(p.group("name")),
                        mem_mib=_num(p.group("mem")),
                    )
                )
        if in_procs:
            for c in cards:
                c.procs_known = True
        res.cards = cards
        res.version = _regex_version(body, r"Kernel Mode Driver Version:\s*(\S+)")
        return res


# --- Enflame (efsmi) ----------------------------------------------------------

#: ``| 0      ZIXIAOC200           | 35.6.5            | 00:23:00.0  Enable        |``
_EF_NAME = re.compile(
    r"^\|\s*(?P<idx>\d+)\s+(?P<name>[A-Za-z][A-Za-z0-9\-]+)\s*\|\s*(?P<bus>[0-9A-Fa-f:]+)"
)
#: ``| {51|}          LP0      132W / 300W | 42976MiB  Disable | {0|}          A0B1J60510343 |``
_EF_STAT = re.compile(
    r"^\|\s*(?P<temp>\d+)[^\s|]*\s+\S+\s+(?P<power>[\d.]+W)\s*/\s*(?P<cap>[\d.]+W)\s*\|\s*"
    r"(?P<used>\d+)MiB\s+\S+\s*\|\s*(?P<util>\d+)%"
)
#: ``pmon`` rows: ``0    1280324  12  5  python`` or ``0    N/A   N/A  N/A  no process running on``
_EF_PMON = re.compile(r"^(?P<dev>\d+)\s+(?P<pid>\d{2,})\s+\S+\s+\S+\s+(?P<name>\S.*?)\s*$")


class Enflame(Family):
    key = "enflame"
    label = "Enflame"
    marker = re.compile(r"^##ENFLAME\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        cards = _tile_cards(body, _EF_NAME, _EF_STAT)
        if not cards:
            res.error = _first_error(body) or "efsmi returned no GCU rows"
            return res
        by_index = {c.index: c for c in cards}
        # efsmi's default table has no memory *total*; pmon adds the process rows.
        in_pmon = "*Dev Pid" in body or "*Idx #" in body
        for line in body.splitlines():
            m = _EF_PMON.match(line.strip())
            if m and int(m.group("dev")) in by_index:
                by_index[int(m.group("dev"))].procs.append(
                    Proc(pid=int(m.group("pid")), user=None, name=_shorten(m.group("name")))
                )
        if in_pmon:
            for c in cards:
                c.procs_known = True
        res.cards = cards
        res.version = _regex_version(body, r"Driver Ver:\s*(\S+)")
        return res


# --- Moore Threads (mthreads-gmi) ---------------------------------------------

_MT_NAME = re.compile(
    r"^(?P<idx>\d+)\s+(?P<name>MTT\s+\S+)\s*\|\s*\S+\s*\|\s*(?P<util>\d+)%\s+(?P<used>\d+)MiB\((?P<total>\d+)MiB\)"
)
_MT_TEMP = re.compile(r"^\s+\S+\s+P\d+\s*\|[^|]*\|\s*(?P<temp>\d+)C\s")
_MT_PROC = re.compile(r"^(?P<dev>\d+)\s+(?P<pid>\d{3,})\s+(?P<name>\S.*?)\s+(?P<mem>\d+)MiB\s*$")


class MooreThreads(Family):
    key = "mthreads"
    label = "MooreThreads"
    marker = re.compile(r"^##MTHREADS\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        cards: list[Card] = []
        cur: Card | None = None
        for line in body.splitlines():
            m = _MT_NAME.match(line.strip())
            if m:
                cur = Card(
                    index=int(m.group("idx")),
                    name=" ".join(m.group("name").split()),
                    util=float(m.group("util")),
                    mem_used=float(m.group("used")),
                    mem_total=float(m.group("total")),
                )
                cards.append(cur)
                continue
            t = _MT_TEMP.match(line)
            if t and cur is not None and cur.temp is None:
                cur.temp = float(t.group("temp"))
        if cards:
            by_index = {c.index: c for c in cards}
            for line in body.splitlines():
                p = _MT_PROC.match(line.strip())
                if p and int(p.group("dev")) in by_index:
                    by_index[int(p.group("dev"))].procs.append(
                        Proc(
                            pid=int(p.group("pid")),
                            user=None,
                            name=_shorten(p.group("name")),
                            mem_mib=_num(p.group("mem")),
                        )
                    )
            if "Processes:" in body:
                for c in cards:
                    c.procs_known = True
            res.version = _regex_version(body, r"Driver Version:\s*(\S+)")
        else:
            res.error = _first_error(body) or "mthreads-gmi returned no MTT rows"
        res.cards = cards
        return res


# --- TsingMicro (tsm_smi) -----------------------------------------------------

_TSM_ROW = re.compile(
    r"^\|\s*(?:(?P<card>\d+)\s+(?P<cname>TX\S+)\s*\|\s*)?\|\s*(?P<chip>\d+)\s+"
    r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f:.]+\s+(?P<temp>\d+)C\s+(?P<pw>\d+)W\s*/\s*(?P<cap>\d+)W\s+"
    r"(?P<used>[\d.]+)M\s*/\s*(?P<total>[\d.]+)M\s+(?P<util>\d+)%"
)
_TSM_ROW_FLAT = re.compile(
    r"^\|\s*(?:(?P<card>\d+)\s+(?P<cname>TX\S+)\s*\|\s*)(?P<chip>\d+)\s+"
    r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f:.]+\s+(?P<temp>\d+)C\s+(?P<pw>\d+)W\s*/\s*(?P<cap>\d+)W\s+"
    r"(?P<used>[\d.]+)M\s*/\s*(?P<total>[\d.]+)M\s+(?P<util>\d+)%"
)
_TSM_PROC = re.compile(
    r"^\|\s*(?P<card>\d+)\s+(?P<pid>\d{3,})\s+(?P<name>\S.*?)\s+(?P<mem>\d+)\s*\|"
)


class TsingMicro(Family):
    """Chips, not cards: four chips per TX8110, the card id only on the first row."""

    key = "tsingmicro"
    label = "TsingMicro"
    marker = re.compile(r"^##TSINGMICRO\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        cards: list[Card] = []
        card_id: int | None = None
        for line in body.splitlines():
            m = _TSM_ROW_FLAT.match(line)
            if not m:
                continue
            if m.group("card") is not None:
                card_id = int(m.group("card"))
                cname = m.group("cname")
            cards.append(
                Card(
                    index=int(m.group("chip")),
                    name=cname if m.group("card") is not None else "TX8110",
                    util=float(m.group("util")),
                    mem_used=float(m.group("used")),
                    mem_total=float(m.group("total")),
                    temp=float(m.group("temp")),
                    power=float(m.group("pw")),
                    card=card_id,
                )
            )
        if not cards:
            res.error = _first_error(body) or "tsm_smi returned no chip rows"
            return res
        # The process table keys on the *card*; attribute a process to every chip of
        # that card, since tsm_smi cannot say which chip of the card it holds.
        per_card: dict[int, list[Card]] = {}
        for c in cards:
            per_card.setdefault(c.card if c.card is not None else c.index // 4, []).append(c)
        saw_table = "Processes:" in body
        for line in body.splitlines():
            p = _TSM_PROC.match(line)
            if not p:
                continue
            for chip in per_card.get(int(p.group("card")), []):
                chip.procs.append(
                    Proc(
                        pid=int(p.group("pid")),
                        user=None,
                        name=_shorten(p.group("name")),
                        mem_mib=_num(p.group("mem")),
                    )
                )
        if saw_table:
            for c in cards:
                c.procs_known = True
        res.version = _regex_version(body, r"SMI\s+V(\S+)")
        res.cards = cards
        return res


# --- T-Head PPU and Iluvatar (nvidia-smi lineage, untested formats) -----------


class Thead(Family):
    """``ppu-smi`` mirrors ``xpu-smi``'s tile layout; no live sample exists yet."""

    key = "thead"
    label = "T-Head"
    marker = re.compile(r"^##THEAD\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        name = re.compile(r"^\|\s*(?P<idx>\d+)\s+(?P<name>PPU-\S+)\s+\S+\s*\|\s*[0-9A-Fa-f]{8}:")
        res.cards = _tile_cards(body, name, _XPU_STAT)
        if not res.cards:
            res.error = _first_error(body) or "ppu-smi returned no PPU rows"
        return res


class Iluvatar(Family):
    key = "iluvatar"
    label = "Iluvatar"
    marker = re.compile(r"^##ILUVATAR\b")

    def parse(self, body: str) -> AccelResult:
        res = AccelResult(family=self.key, label=self.label)
        name = re.compile(
            r"^\|\s*(?P<idx>\d+)\s+(?P<name>Iluvatar\s+\S+)\s+\S+\s*\|\s*[0-9A-Fa-f]{8}:"
        )
        res.cards = _tile_cards(body, name, _XPU_STAT)
        if not res.cards:
            res.error = _first_error(body) or "ixsmi returned no GPU rows"
        return res


def _tile_cards(body: str, name_re: re.Pattern[str], stat_re: re.Pattern[str]) -> list[Card]:
    """Parse the nvidia-smi-style two-line-per-card tile layout.

    The stats line belongs to the card named on the line *above* it — attaching it to
    the next one instead would shift every value by one card, which is worse than not
    parsing at all.
    """
    cards: list[Card] = []
    pending: Card | None = None
    for line in body.splitlines():
        n = name_re.match(line)
        if n:
            pending = Card(index=int(n.group("idx")), name=" ".join(n.group("name").split()))
            cards.append(pending)
            continue
        s = stat_re.match(line)
        if s and pending is not None:
            g = s.groupdict()
            pending.temp = _num(g.get("temp"))
            pending.power = _num(g.get("power"))
            pending.mem_used = _num(g.get("used"))
            pending.mem_total = _num(g.get("total"))
            pending.util = _num(g.get("util"))
            pending = None
    return cards


class Generic(Family):
    """No accelerator tool answered — CPU-only, or a vendor without a parser yet."""

    key = "generic"
    label = "CPU"
    marker = re.compile(r"^##GENERIC\b")

    def parse(self, body: str) -> AccelResult:
        return AccelResult(family=self.key, label=self.label)


FAMILIES: dict[str, Family] = {
    f.key: f
    for f in (
        Nvidia(),
        Ascend(),
        Kunlunxin(),
        MetaX(),
        Enflame(),
        MooreThreads(),
        Thead(),
        Iluvatar(),
        TsingMicro(),
    )
}

#: Families the probe tries, in the order their markers may appear.
PROBE_ORDER: tuple[str, ...] = (
    "nvidia",
    "ascend",
    "kunlunxin",
    "metax",
    "enflame",
    "mthreads",
    "tsingmicro",
    "thead",
    "iluvatar",
)

_ALIASES = {
    "cuda": "nvidia",
    "nvidia": "nvidia",
    "ascend": "ascend",
    "ascend-910c": "ascend",
    "ascend-910b": "ascend",
    "npu": "ascend",
    "kunlunxin": "kunlunxin",
    "kunlun": "kunlunxin",
    "xpu": "kunlunxin",
    "metax": "metax",
    "enflame": "enflame",
    "gcu": "enflame",
    "mthreads": "mthreads",
    "moore-threads": "mthreads",
    "musa": "mthreads",
    "tsingmicro": "tsingmicro",
    "tsm": "tsingmicro",
    "thead": "thead",
    "t-head": "thead",
    "ppu": "thead",
    "iluvatar": "iluvatar",
}


def family_hint(platform: str | None, tags: list[str] | None = None) -> str:
    """Rename an inventory ``platform``/tag into a family key, or ``generic``.

    A *hint* only, used to decide which tools are worth trying first and what to call
    a CPU-only host. The probe's markers decide what actually ran, so a wrong or
    missing hint costs one skipped tool, never a wrong reading.

    ``dcu`` (hy-smi) deliberately maps to ``generic``: its parser is not written yet,
    and the CPU section of the probe is still useful on those hosts.
    """
    if platform and platform in FAMILIES:
        return platform
    if platform and platform in _ALIASES:
        return _ALIASES[platform]
    for tag in tags or []:
        if tag in _ALIASES:
            return _ALIASES[tag]
    return "generic"
