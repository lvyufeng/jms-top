"""Parsing a probe reply into a :class:`jms_top.model.HostView`.

The reply is a flat document of ``##SECTION`` markers. Every parser here is total:
a section that is missing, truncated or unparseable yields empty/``None`` values and
*never* an exception — a monitor that dies on a host with an unexpected kernel is
useless precisely when the fleet is unhealthy.

The one rule that is load-bearing: **a missing value is not a zero**. ``None`` means
"not reported"; the model renders it as ``—`` and never as ``0`` or ``free``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .cpu import CpuSample
from .families import FAMILIES
from .model import HostView, NodeStats
from .probe import M_CPU, M_DISK, M_STALLED, M_SYSTEM, M_TOP, M_WHO, SENTINEL

#: A section marker line. ``##STALLED:nvidia:15s`` carries a payload, so an optional
#: ``:detail`` suffix is allowed — every other marker is bare.
#: Family markers may be hyphenated (``##METAX-WIDE``); the payload form
#: (``##STALLED:nvidia:15s``) is a colon-suffixed variant of the same name.
_MARKER = re.compile(r"^##(?P<name>[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*)(?::(?P<detail>[^\s]*))?$")


def split_sections(text: str) -> dict[str, list[str]]:
    """Split a probe reply into ``{marker_without_hash: [lines]}``.

    Marker lines are ``##SYS``-style by themselves (the vendor snippets echo them
    alone), so a marker is only taken when the whole line matches — a process whose
    command line happens to contain ``##`` does not start a new section.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        m = _MARKER.match(stripped)
        if m:
            name = m.group("name")
            current = sections.setdefault(name, [])
            # A marker may carry a payload (``##STALLED:nvidia:15s``), which is the
            # section's first line; a bare marker starts empty.
            detail = m.group("detail")
            if detail:
                current.append(detail)
            continue
        if current is not None:
            current.append(line)
    return sections


def parse_system(lines: list[str], *, stamp: float) -> NodeStats:
    """Read the ``##SYS`` block: load, cores, memory, boot, CPU counters, hostname.

    The first five fields are positional (load / cores / cores-online / memory), and
    each is checked by *shape* before being consumed, so a reply that lost a line
    degrades to "unknown" instead of shifting every later field by one. The tail
    (epochs, uptimes, CPU counters, hostname) is not positional between probe kinds —
    the full probe prints ``btime`` and the cheap one does not — so the counters are
    found by shape and everything before them is read as ``[boot,] uptime``.
    """
    st = NodeStats()
    vals = [ln.strip() for ln in lines if ln.strip()]
    idx = 0
    if idx < len(vals) and re.match(r"^[\d.]+\s+[\d.]+\s+[\d.]+", vals[idx]):
        parts = vals[idx].split()
        st.load1, st.load5, st.load15 = (float(p) for p in parts[:3])
        idx += 1
    if idx < len(vals) and vals[idx].isdigit():
        st.cpus = int(vals[idx])
        idx += 1
    if idx < len(vals) and vals[idx].isdigit():
        st.cpus_online = int(vals[idx])
        idx += 1
    if idx < len(vals) and re.match(r"^\d+\s+\d+", vals[idx]):
        mem = vals[idx].split()
        st.mem_total_mib = round(int(mem[0]) / 1024, 1)
        st.mem_avail_mib = round(int(mem[1]) / 1024, 1)
        if len(mem) > 2 and mem[2].isdigit():
            st.swap_total_mib = round(int(mem[2]) / 1024, 1)
        if st.mem_total_mib:
            st.mem_pct = round(100.0 * (1 - st.mem_avail_mib / st.mem_total_mib), 1)
        idx += 1

    # Two bare integers are the CPU jiffy counters; everything else is the hostname.
    # Location is not enough to tell them apart: `btime` is an epoch, which is
    # indistinguishable from an uptime in seconds, and the cheap probe omits it.
    rest = vals[idx:]
    cpu_index = next((i for i, v in enumerate(rest) if re.match(r"^\d+\s+\d+$", v)), None)
    leading = rest if cpu_index is None else rest[:cpu_index]
    if len(leading) >= 2 and leading[0].isdigit() and leading[1].isdigit():
        st.boot_time = float(leading[0])  # epoch seconds, full probe only
        st.uptime_s = float(leading[1])  # seconds since boot
    elif leading and leading[0].isdigit():
        st.uptime_s = float(leading[0])  # seconds since boot, cheap probe only
    cpu_line = rest[cpu_index] if cpu_index is not None else None
    if cpu_line:
        total, idle = (int(x) for x in cpu_line.split())
        st.cpu_sample = CpuSample(total=total, idle=idle, stamp=stamp)
    # The hostname is the last line *after* the counters. Restricting it to the tail
    # matters: `btime` is an epoch and a short uptime is a small integer, so a line
    # before the counters can be either — but nothing after them is ever numeric.
    tail = rest[cpu_index + 1 :] if cpu_index is not None else []
    host_lines = [v for v in tail if v]
    if host_lines:
        st.hostname = host_lines[-1]
    elif rest and rest[-1] and not rest[-1].isdigit() and rest[-1] != cpu_line:
        st.hostname = rest[-1]  # counters missing: fall back to the last text line
    return st


def parse_who(lines: list[str]) -> list[str]:
    """``who | awk '{print $1}' | sort | uniq -c`` -> ``["root", ...]``."""
    out = []
    for line in lines:
        m = re.match(r"^\s*(\d+)\s+(\S+)\s*$", line)
        if m:
            count, user = int(m.group(1)), m.group(2)
            out.extend([user] * count)
    return out


def parse_top(lines: list[str]) -> list[dict]:
    """``ps axo pid=,user=,etime=,pcpu=,pmem=,cmd=`` rows."""
    out = []
    for line in lines:
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$", line)
        if not m:
            continue
        out.append(
            {
                "pid": int(m.group(1)),
                "user": m.group(2),
                "elapsed": m.group(3),
                "cpu": float(m.group(4)),
                "mem": float(m.group(5)),
                "cmd": m.group(6).strip(),
            }
        )
    return out


def parse_disk(line: str) -> tuple[float | None, float | None]:
    """``df -P -k /`` -> ``(used_pct, free_gib)``."""
    parts = line.split()
    if len(parts) < 4:
        return None, None
    # "Filesystem 1024-blocks Used Available Capacity Mounted" — the mountpoint may
    # contain spaces, so size columns are read from the left and the mount from the right.
    pct = re.search(r"(\d+)%", parts[-2])
    try:
        free_gib = round(int(parts[3]) / (1024 * 1024), 1)
    except (ValueError, IndexError):
        free_gib = None
    return (float(pct.group(1)) if pct else None), free_gib


def _belongs(marker: str, family: str) -> bool:
    """Whether a section marker belongs to ``family``.

    ``##HYGON``, ``##HYGON-VRAM`` and ``##HYGON-PROCS`` are all the family's; ``##SYS``
    is not. Matching on ``FAMILY`` or ``FAMILY-`` avoids the trap where ``HYGON`` is a
    prefix of ``HYGON-VRAM`` and a looser test silently pulls in other families'
    sections (`##THEAD` vs nothing, but `##NVIDIA` vs `##NVIDIA-WIDE` both belong).
    """
    up = marker.upper()
    fam = family.upper()
    return up == fam or up.startswith(f"{fam}-")


def parse_reply(view: HostView, text: str, *, stamp: float | None = None) -> HostView:
    """Fill ``view`` from one probe reply. Mutates and returns it.

    ``stamp`` is the wall-clock second the reply arrived; it is what makes the CPU
    delta between two replies meaningful, so a caller replaying captured text passes
    a synthetic clock.
    """
    stamp = stamp if stamp is not None else datetime.now(timezone.utc).timestamp()
    # Parse into a fresh NodeStats: the caller may hand us an existing view (the fast
    # tick reuses one) and a field carried over from an older reply would then read as
    # current — `uptime_s` going missing is invisible until the node reboots.
    view.node = NodeStats()
    sections = split_sections(text)
    # ``##CPU`` is the cheap section; ``##SYS`` the full one. They carry the same
    # layout, so whichever arrived is read the same way, and the keys are normalised
    # (marker names arrive without their ``##`` prefix).
    sys_key = M_SYSTEM[2:] if M_SYSTEM[2:] in sections else M_CPU[2:]

    # Keys are marker names *without* the ``##`` prefix (`split_sections` strips it),
    # so strip it here too rather than comparing against the wire format.
    if sys_key in sections:
        view.node = parse_system(sections[sys_key], stamp=stamp)
    if M_WHO[2:] in sections:
        view.node.logins = parse_who(sections[M_WHO[2:]])
    if M_TOP[2:] in sections:
        view.node.top = parse_top(sections[M_TOP[2:]])
    if sections.get(M_DISK[2:]):
        pct, free = parse_disk(sections[M_DISK[2:]][0])
        view.node.disk_pct, view.node.disk_free_gib = pct, free

    # ``##STALLED`` is empty unless it carries a payload, so "present" is what marks a
    # stall, not "non-empty".
    stalled = sections.get(M_STALLED[2:])
    if stalled is not None:
        # The vendor tool was time-boxed out, so its silence means "no answer", not
        # "no cards". Recorded here so the host reads as UNKNOWN with the reason on
        # screen, instead of quietly looking like a machine with nothing to report.
        detail = " ".join(line.strip() for line in stalled if line.strip()) or "timeout"
        from .families import AccelResult

        family = view.platform if view.platform in FAMILIES else "generic"
        label = FAMILIES[family].label if family in FAMILIES else family
        vendor, _, how_long = detail.partition(":")
        if vendor and vendor in FAMILIES:
            family, label = vendor, FAMILIES[vendor].label
        view.accel = AccelResult(
            family=family,
            label=label,
            error=f"accelerator tool did not answer within {how_long or detail} (driver stalled?)",
        )

    # Accelerator sections: at most one per host in practice. If a host somehow
    # answers for two families (a node with both an NPU and a GPU), the first in
    # probe order wins and the rest are ignored rather than merged into nonsense.
    stall_error = view.accel.error if (stalled and view.accel) else None
    # The first family that actually *found cards* owns the host. A family whose tool
    # merely exists does not: `nvidia-smi` is installed on hosts that have no NVIDIA
    # driver, where it prints "NVIDIA-SMI has failed because it couldn't communicate
    # with the NVIDIA driver" and no rows at all. Taking that as the answer would show
    # a 沐曦 (MetaX) box as an NVIDIA host with no cards — which is exactly what used
    # to happen on metax-mc550-121, because NVIDIA is probed first.
    failed: dict[str, object] = {}
    for fam, parser in FAMILIES.items():
        keys = [fam.upper(), f"{fam.upper()}-WIDE"]  # e.g. ##METAX / ##METAX-WIDE
        present = [k for k in keys if k in sections]
        if not present:
            continue
        # Every section belonging to this family, not just the first: a vendor that
        # needs more than one tool builds several (`##HYGON`, `##HYGON-VRAM`,
        # `##HYGON-PROCS` — memory from sysfs, processes from KFD). Their markers keep
        # the family name as a prefix, and the parser reads them by name.
        parts = [f"##{k}\n" + "\n".join(sections[k]) for k in sections if _belongs(k, fam)]
        body = "\n".join(parts)
        # A stalled tool prints no marker of its own, so the section may be empty and
        # the reason recorded above is still the answer.
        if not body.strip() and not stalled:
            continue
        parsed = parser.parse(body)
        if parsed.total:
            view.accel = parsed
            break
        if parsed.total == 0 and stall_error:
            # Whatever did parse is a partial table; "no rows" from a killed tool is
            # not the same answer as "no rows" from a tool that ran.
            parsed.error = stall_error
        # Kept, not chosen: if nothing else found cards this is the reason to show.
        failed[fam] = parsed
    else:
        # No family reported a single card. Prefer a genuine tool error over a silent
        # empty section, and say which tool it came from.
        if failed:
            view.accel = next(iter(failed.values()))  # type: ignore[assignment]
        elif not view.accel:
            view.accel = None
    view.updated_at = datetime.now(timezone.utc)
    view.raw = text
    return view


def reached(text: str) -> bool:
    """Whether a reply proves the script ran on the far side.

    KoKo rejects a connection with ssh exit **0** and the error on stdout, so a
    sentinel in the output is the only trustworthy liveness signal. Only the first few
    lines are searched: the sentinel
    is the first thing either script prints, and a process command line further down
    must not be able to fake it.
    """
    for line in text.splitlines()[:3]:
        if line.strip() == SENTINEL:
            return True
    return False


def parse_proc_detail(lines: list[str]) -> dict[int, dict[str, str | None]]:
    """``##PROCDETAIL`` -> ``{pid: {user, elapsed, cmdline}}``.

    A follow-up section (see :func:`jms_top.probe.proc_detail_script`) that runs only
    when an answer needs to name a process rather than point at a pid. Absent on a
    normal tick, so this is not on the hot path.
    """
    out: dict[int, dict[str, str | None]] = {}
    for line in lines:
        parts = line.split("|", 3)
        if len(parts) != 4 or not parts[0].strip().isdigit():
            continue
        pid = int(parts[0])
        out[pid] = {
            "user": parts[1].strip() or None,
            "elapsed": parts[2].strip() or None,
            "cmdline": parts[3].strip() or None,
        }
    return out


def apply_proc_detail(view, detail: dict[int, dict[str, str | None]]) -> None:  # noqa: ANN001
    """Fill ``Proc.user``/``elapsed``/``cmdline`` from a detail reply, in place."""
    if not detail or view.accel is None:
        return
    for card in view.accel.cards:
        for proc in card.procs:
            info = detail.get(proc.pid) if proc.pid is not None else None
            if not info:
                continue
            if info.get("user"):
                proc.user = info["user"]
            if info.get("elapsed"):
                proc.elapsed = info["elapsed"]
            if info.get("cmdline"):
                proc.cmdline = info["cmdline"]
