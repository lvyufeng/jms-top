"""Find every asset this account can reach, straight from KoKo.

``jms-top hosts --discover`` covers the REST path, but it needs a session token and an
API that may not be reachable. KoKo itself knows the same answer and needs nothing but
the SSH access already used for polling: open an interactive session, press ``p``, and
it prints the assets you are permitted to log in to — one page at a time, with the name
and address in fixed-width columns.

That is what this module drives. It is a screen scraper over a stable, documented menu
(``p`` = 资产, ``n``/``b`` = next/previous page), which is a weaker contract than an API
— so the parser is written to fail loudly rather than to guess: a page whose table
cannot be read raises, and a discovery that returns fewer assets than the header
announced says so.

Nothing here connects to the assets themselves. It only enumerates them, so it is as
safe to run as ``jms-top --ping`` is to run.
"""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import time
from dataclasses import dataclass, field

from .config import Bastion
from .ssh import _BASE_OPTS, _BATCH_MODE

#: One row of KoKo's asset table: an index, then pipe-separated columns. The trailing
#: pipe is optional — the last column has no `|` after it on some builds/wrappings.
_ROW = re.compile(r"^\s*\d+\s+\|(?P<name>[^|]*)\|(?P<addr>[^|]*)\|(?P<platform>[^|]*)(?:\||$)")

#: ``页码：1，每页行数：1，总页数：77，总数量：77``
_TOTAL = re.compile(r"总数量[：:]\s*(?P<n>\d+)")

#: Terminal control sequences KoKo emits for colour and line clearing.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

#: Asset-name suffixes that describe hardware rather than identity; used to turn
#: ``bm-baai-dx-zone1-d-a100-40g-2-42`` into a short, stable inventory key.
_NOISE = re.compile(r"^(bm|p|hp|vm)-", re.IGNORECASE)

#: The model token inside a name: ``a100``, ``h800``, ``910c``, ``p800``, ``mc550`` …
_MODEL = re.compile(
    r"^(a100|a800|h100|h800|4090|910c|910b|810e|tx8110|s5000|p800|mc550|bw1[01]00)$",
    re.IGNORECASE,
)


class DiscoverError(RuntimeError):
    """KoKo's list could not be read."""


@dataclass
class Asset:
    name: str
    address: str
    platform: str = ""
    asset_id: str | None = None
    #: The name exactly as KoKo printed it, before :func:`_tidy` — needed to look the
    #: asset up in the API, which keeps the original spelling.
    raw_name: str = ""
    #: Inventory key derived from the name; unique within the result.
    key: str = ""
    tags: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return _ANSI.sub("", text).replace("\r", "")


class KoKoSession:
    """A pty-backed interactive KoKo session, used as a context manager.

    A pty is required: KoKo prints its menu only to a terminal, and without one it
    silently runs a non-interactive shell instead.
    """

    def __init__(self, bastion: Bastion, *, connect_timeout: float = 20.0):
        self.bastion = bastion
        self.connect_timeout = connect_timeout
        self._master: int | None = None
        self._proc: subprocess.Popen | None = None
        self._buffer = ""

    def __enter__(self) -> KoKoSession:
        if not self.bastion.login:
            raise DiscoverError("no login configured; set [bastion] login in config.toml")
        argv = ["ssh", "-tt", *_BASE_OPTS, *_BATCH_MODE]
        if self.bastion.ssh_key:
            argv += ["-i", str(self.bastion.ssh_key), "-o", "IdentitiesOnly=yes"]
        argv += ["-p", str(self.bastion.ssh_port), f"{self.bastion.login}@{self.bastion.host}"]
        master, slave = pty.openpty()
        env = {**os.environ, "TERM": "xterm"}
        self._proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave, close_fds=True, env=env
        )
        os.close(slave)
        self._master = master
        self._read_until(lambda t: "Opt>" in t, self.connect_timeout)
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait(timeout=5)
        if self._master is not None:
            os.close(self._master)
        self._master, self._proc = None, None

    # --- terminal plumbing ------------------------------------------------------

    def _read_until(self, predicate, timeout: float) -> str:  # noqa: ANN001
        """Pump the pty until ``predicate(buffer)`` — or raise on timeout.

        Timeouts raise rather than returning what arrived: a partial page would parse
        into a partial inventory, and "you have 30 assets" when you have 77 is exactly
        the silent under-report this tool is supposed to stop.
        """
        if self._master is None:
            raise DiscoverError("session is not open")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._master], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(self._master, 1 << 16)
            except OSError as exc:  # the pty closed: ssh died
                raise DiscoverError(f"session closed while reading: {exc}") from exc
            if not chunk:
                break
            self._buffer += _clean(chunk.decode("utf-8", "replace"))
            if predicate(self._buffer):
                return self._buffer
        raise DiscoverError(
            f"timed out after {timeout:g}s waiting for KoKo; got: {self._buffer[-300:]!r}"
        )

    def send(self, keys: bytes, *, wait: float = 1.0) -> str:
        """Send keystrokes and collect whatever comes back for ``wait`` seconds."""
        if self._master is None:
            raise DiscoverError("session is not open")
        self._buffer = ""
        os.write(self._master, keys)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._master], [], [], 0.15)
            if ready:
                try:
                    chunk = os.read(self._master, 1 << 16)
                except OSError:
                    break
                if not chunk:
                    break
                self._buffer += _clean(chunk.decode("utf-8", "replace"))
        return self._buffer


def parse_page(text: str) -> list[Asset]:
    """Read the asset rows out of one rendered page.

    The columns are fixed-width *and* pipe-separated, so the pipes are the contract and
    the padding is not. Trailing decoration KoKo adds to some names (a ``（ID：）``
    suffix, a ``维修中`` marker) is kept in the name — it is real information about the
    asset — but never in the derived key.
    """
    assets: list[Asset] = []
    for line in text.splitlines():
        match = _ROW.match(line)
        if not match:
            continue
        name = _tidy(match.group("name"))
        address = match.group("addr").strip()
        if not name or name.startswith("名称"):
            continue
        assets.append(
            Asset(
                name=name,
                address=address,
                platform=match.group("platform").strip(),
                raw_name=re.sub(r"\s+", " ", match.group("name")).strip(),
            )
        )
    return assets


def _tidy(name: str) -> str:
    """Strip the padding and the ``（ID：...）`` suffix KoKo appends to some rows."""
    return re.sub(r"\s*（ID[：:][^）]*）\s*$", "", name.strip())


def announce_total(text: str) -> int | None:
    """The ``总数量`` KoKo announced, if this page carries it."""
    match = _TOTAL.search(text)
    return int(match.group("n")) if match else None


def discover(bastion: Bastion, *, page_wait: float = 1.6, max_pages: int = 400) -> list[Asset]:
    """Every asset the account may log in to, with a unique inventory key each.

    Pages forward until KoKo stops producing new rows or the announced count is reached.
    The count is checked at the end: a short answer is reported rather than returned, so
    "some assets are missing" can never look like "that is all your assets".
    """
    assets: list[Asset] = []
    seen: set[str] = set()
    total: int | None = None
    with KoKoSession(bastion) as session:
        page = session.send(b"p\r\n", wait=max(page_wait, 3.0))
        total = announce_total(page) or total
        rows = parse_page(page)
        for asset in rows:
            if asset.name not in seen:
                seen.add(asset.name)
                assets.append(asset)
        for _ in range(max_pages):
            if total is not None and len(assets) >= total:
                break
            page = session.send(b"n\r\n", wait=page_wait)
            fresh = [a for a in parse_page(page) if a.name not in seen]
            if not fresh:
                break  # last page: KoKo repeats the final row rather than blanking
            for asset in fresh:
                seen.add(asset.name)
                assets.append(asset)
    if total is not None and len(assets) < total:
        raise DiscoverError(
            f"KoKo advertised {total} assets but only {len(assets)} were readable; "
            "the list may have changed mid-scan — re-run"
        )
    assign_keys(assets)
    return assets


def attach_asset_ids(assets: list[Asset], known: dict[str, str]) -> int:
    """Fill in ``asset_id`` from a ``name -> id`` map; returns how many were matched.

    KoKo's list shows names and addresses but not the asset UUID, and the UUID is what
    a login string wants (`login#account#asset`). The names match exactly in practice,
    so the API is used purely as a lookup table — **membership stays KoKo's decision**,
    which is the whole point: an asset you cannot log in to is not something this tool
    should poll.
    """
    matched = 0
    for asset in assets:
        # `_tidy` removes a trailing `（ID：…）`; the API still lists the full name, so
        # both spellings are tried before giving up.
        for candidate in (asset.name.strip(), asset.raw_name.strip() if asset.raw_name else ""):
            asset_id = known.get(candidate) if candidate else None
            if asset_id:
                asset.asset_id = asset_id
                matched += 1
                break
    return matched


def assign_keys(assets: list[Asset]) -> None:
    """Give each asset a short, unique, shell-friendly key plus tags.

    The full KoKo name encodes datacenter, zone, rack and serial
    (``bm-baai-dx-zone1-d-a100-40g-2-42``), which is unusable as a selector. The key
    keeps the identifying tail and the accelerator family, and tags carry the rest so
    `jms-top a100` / `jms-top zone1` both work.
    """
    taken: set[str] = set()
    for asset in assets:
        bare = _NOISE.sub("", asset.name).lower()
        parts = [p for p in bare.split("-") if p]
        model = _model_of(bare)
        # The last part is the machine's own ordinal, which is what distinguishes
        # siblings; the model is what makes it readable. `a100-42`, not `nvidia-2-42`.
        # Names without a model token (the 810e boxes) fall back to the family.
        serial = parts[-1] if parts else ""
        stem = model or family_of(asset.name)
        if stem == "unknown":
            stem = parts[0] if parts else asset.name
        # A name like `u101-u101` would otherwise become the stuttering key `u101-u101`.
        base = stem if stem == serial else "-".join(p for p in [stem, serial] if p)
        base = base or asset.name
        key, n = base, 2
        while key in taken:
            key, n = f"{base}-{n}", n + 1
        taken.add(key)
        asset.key = key
        asset.tags = sorted({*parts, family_of(asset.name), model or ""} - {"", "d", "lc"})


#: Substring -> family. Ordered, because ``bw1100`` must not be caught by ``bw1`` twice
#: and ``gpu`` is a fallback that would otherwise swallow the specific names.
_FAMILY_HINTS: list[tuple[str, str]] = [
    ("910c", "ascend"),
    ("910b", "ascend"),
    ("a100", "nvidia"),
    ("a800", "nvidia"),
    ("h800", "nvidia"),
    ("h100", "nvidia"),
    ("4090", "nvidia"),
    ("810e", "thead"),
    ("tx8110", "tsingmicro"),
    ("s5000", "mthreads"),
    ("p800", "kunlunxin"),
    ("mc550", "metax"),
    ("bw1100", "hygon"),
    ("bw1000", "hygon"),
]


def _model_of(bare: str) -> str:
    """The accelerator model in a name, if it names one (``a100``, ``910c``)."""
    for part in bare.split("-"):
        if _MODEL.match(part):
            return part.lower()
    return ""


def family_of(name: str) -> str:
    lowered = name.lower()
    for needle, family in _FAMILY_HINTS:
        if needle in lowered:
            return family
    if "gpu" in lowered:
        return "nvidia"
    return "unknown"
