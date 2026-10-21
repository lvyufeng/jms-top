"""The remote half: one script that answers "what is free on this host".

Both shapes of answer run through jms-bridge's control-master SSH runner, so the
bastion costs one handshake per host rather than one per poll:

``cpu``
    The **node** half: load, cores, memory, CPU counters, uptime, logins, top
    processes, disk. All of it is file reads and one ``ps`` — ~1 s per host even when
    the accelerator driver is having a bad day.

``fleet``
    ``cpu`` plus **every accelerator family's** tool. Each vendor's tool is guarded by
    ``command -v`` and wrapped in a ``##FAMILY`` marker, so a host emits exactly one
    section — the vendor it has — and nothing for the ones it lacks. That is what lets
    a single command serve nine accelerator families without the inventory having to
    be right about which one each host runs.

``gpu``
    The accelerator section alone, for a fast capture.

Design notes:

* **No ``env_setup``.** Chip tools live on the default ``PATH`` (or in the vendor's
  install dir, which each snippet tries); the inventory's ``env_setup`` exists to
  activate a *dev* environment for running jobs. Sourcing it here would make the
  monitor fail on exactly the hosts whose env script is broken — when you most want to
  see that the host is fine and its *env* is not.
* **Never a multi-second sleep on the far side.** CPU utilisation comes from the
  delta between successive polls (see :mod:`jms_top.cpu`); the script only reads
  counters.
* **The output is bounded.** ``ps | head`` and a per-vendor ``head -N``, because the
  bastion buffers a reply (jms-bridge measured ~200 s TTFB on large buffered
  responses) and a 30 000-process node must not push megabytes through it every 3 s.
* **Every vendor call is time-boxed.** Chip tools buffered through a pipe (all of them
  but the GPU query, which pipes itself) cannot exceed the local script budget anyway;
  the explicit numbers matter for ``nvidia-smi``, whose ``--query-gpu`` runs *without*
  a pipe: it was measured at 9-11 s on an H800 host while the ASCII tile on the same
  host took 18 s, so a "hung" driver does not exist here and a 6 s box would have
  reported healthy cards as unknown. The budget is that latency plus headroom, and a
  query that *is* cut off reports ``##STALLED`` — the host then reads *unknown* with
  "driver stalled?" on screen, because a stalled tool has reported nothing, which is
  not the same answer as "no cards".
* **Two forms of NVIDIA output.** ``--query-gpu`` CSV by default: seven values per
  card that parse exactly, and it survives an 8-card host where the ASCII tile does
  not (the tile is ~45 lines per card and ``head`` cut it at card 8 — that is the
  ``nvidia_tile`` fallback, kept for hosts whose driver lacks ``--query-gpu``).
  The CSV deliberately asks for the **index** and bus id, not the uuid: some driver
  builds accept ``uuid`` in the query and then print an empty column, which shifts
  every later field by one and makes *every* card unreadable on those hosts.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Section markers. Parsed by :mod:`jms_top.scan`; the family ones are also the
#: prefixes :mod:`jms_top.families` matches on.
M_CPU = "##CPU"
M_SYSTEM = "##SYS"
M_WHO = "##WHO"
M_TOP = "##TOP"
M_DISK = "##DISK"
M_STALLED = "##STALLED"

#: The node section. ``##CPU`` and ``##SYS`` carry the *same* layout on purpose: the
#: cheap tick replaces the node stats, so a field missing from it would blink between a
#: value and "—" every interval.
_NODE = r"""set +e +u +o pipefail
echo '{marker}'
cat /proc/loadavg 2>/dev/null
nproc --all 2>/dev/null || nproc 2>/dev/null
nproc 2>/dev/null
awk '/MemTotal/{{t=$2}} /MemAvailable/{{a=$2}} /SwapTotal/{{sw=$2}} END{{print t, a, sw}}' /proc/meminfo 2>/dev/null
# total jiffies (user nice system idle iowait irq softirq steal), then idle+iowait:
# both are "CPU not doing work", the convention top/htop use.
awk '/^cpu /{{print $2+$3+$4+$5+$6+$7+$8+$9, $4+$5}}' /proc/stat 2>/dev/null
cut -d. -f1 /proc/uptime 2>/dev/null
hostname 2>/dev/null
echo '##WHO'
who 2>/dev/null | awk '{{print $1}}' | sort | uniq -c | sort -rn | head -8
echo '##TOP'
ps axo pid=,user=,etime=,pcpu=,pmem=,cmd= --sort=-pcpu 2>/dev/null | head -{top_n} | cut -c1-200
echo '##DISK'
df -P -k / 2>/dev/null | tail -1
"""


def node_script(marker: str = M_SYSTEM, *, top_n: int = 6) -> str:
    """The node section, under ``marker`` (``##SYS`` for full, ``##CPU`` for cheap)."""
    return _NODE.format(marker=marker, top_n=top_n)


# --- per-family snippets -------------------------------------------------------
# Each starts with its own marker, so a host that lacks the tool emits nothing at all.
# ``T`` is the per-command time box (dash has no `timeout` default; busybox does).

_NVIDIA_WIDE = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "if command -v nvidia-smi >/dev/null 2>&1; then "
    "out=$($T nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null); rc=$?; "
    "echo '##NVIDIA'; "
    "if [ $rc -eq 124 ]; then echo '##STALLED:nvidia:12s'; else echo \"$out\"; "
    "$T nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory "
    "--format=csv,noheader,nounits 2>/dev/null; fi; fi"
)
_NVIDIA_TILE = (
    "T=timeout 10; command -v timeout >/dev/null 2>&1 || T=; "
    "if command -v nvidia-smi >/dev/null 2>&1; then "
    "out=$($T nvidia-smi 2>/dev/null); rc=$?; echo '##NVIDIA'; "
    "if [ $rc -eq 124 ]; then echo '##STALLED:nvidia:10s'; else "
    "printf '%s\\n' \"$out\" | head -{tile_n}; fi; fi"
)
_ASCEND = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "if command -v npu-smi >/dev/null 2>&1; then echo '##ASCEND'; "
    "out=$($T npu-smi info 2>/dev/null); [ $? -eq 124 ] && echo '##STALLED:ascend:6s' || "
    "printf '%s\\n' \"$out\" | head -{tile_n}; fi"
)
_KUNLUNXIN = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "if command -v xpu-smi >/dev/null 2>&1; then echo '##KUNLUNXIN'; "
    "out=$($T xpu-smi 2>/dev/null); [ $? -eq 124 ] && echo '##STALLED:kunlunxin:6s' || "
    "printf '%s\\n' \"$out\" | head -{tile_n}; fi"
)
#: MetaX（沐曦）曦云 C550/N260: ``/opt/mxdriver/bin/mx-smi`` with ``/usr/bin/mx-smi``
#: symlinked by the driver. Per-type queries are used, not the default table, because
#: they are both faster (~0.9 s vs ~2 s) and less ambiguous: the default table prints
#: utilisation in one column, memory in another, and the process table only on certain
#: driver versions. The BDF printed on every ``GPU#n`` header joins the process rows
#: (which are keyed by GPU index) back to the holder in one pass.
#:
#: Falls back to the default table when the per-type queries come back empty — a
#: driver or firmware build where those flags do not exist yet. The fallback is
#: line-aligned so the parser knows which layout it is reading; a host that answers
#: neither prints nothing and is reported as unknown, never as free.
_METAX = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "M=$(command -v mx-smi 2>/dev/null); "
    '[ -n "$M" ] || [ -x /opt/mxdriver/bin/mx-smi ] && M=${M:-/opt/mxdriver/bin/mx-smi}; '
    'if [ -n "$M" ]; then '
    'u=$($T "$M" --show-usage 2>/dev/null); '
    'm=$($T "$M" --show-memory 2>/dev/null); '
    "p=$($T \"$M\" --show-process 2>/dev/null | sed -n '/Process:/,/End of Log/p'); "
    # Three queries, so "did it answer" cannot be one command's exit status: if the
    # memory query produced nothing the driver did not answer at all, and saying so is
    # what keeps the host UNKNOWN instead of quietly empty.
    "[ -n \"$m\" ] || { echo '##STALLED:metax:6s'; exit 0; }; "
    "if printf '%s' \"$m\" \"$u\" | grep -q 'GPU#'; then echo '##METAX-WIDE'; "
    "printf '%s\\n' \"$m\" | head -{half}; "
    "printf '%s\\n' \"$u\" | head -{half}; "
    "printf '%s\\n' \"$p\" | head -{half}; "
    "else echo '##METAX'; printf '%s\\n' \"$($T \"$M\" 2>/dev/null)\" | head -{tile_n}; fi; fi"
)
_ENFLAME = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "if command -v efsmi >/dev/null 2>&1; then echo '##ENFLAME'; "
    "out=$($T efsmi 2>/dev/null); [ $? -eq 124 ] && echo '##STALLED:enflame:6s' || "
    "printf '%s\\n' \"$out\" | head -{tile_n}; "
    "$T efsmi pmon -c 1 2>/dev/null | head -{half}; fi"
)
_MTHREADS = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "if command -v mthreads-gmi >/dev/null 2>&1; then echo '##MTHREADS'; "
    "out=$($T mthreads-gmi 2>/dev/null); [ $? -eq 124 ] && echo '##STALLED:mthreads:6s' || "
    "printf '%s\\n' \"$out\" | head -{tile_n}; fi"
)
_TSINGMICRO = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "if command -v tsm_smi >/dev/null 2>&1; then echo '##TSINGMICRO'; "
    "out=$($T tsm_smi 2>/dev/null); [ $? -eq 124 ] && echo '##STALLED:tsingmicro:6s' || "
    "printf '%s\\n' \"$out\" | grep -v 'exceeds max' | head -{tile_n}; fi"
)
#: 平头哥 PPU (T-Head ZW810E). The SDK installs ``ppu-smi`` under
#: ``/usr/local/PPU_SDK/ppu-smi/bin/`` and does **not** put it on ``PATH`` — measured on
#: the live fleet, where 16 PPUs were present and ``command -v ppu-smi`` found nothing.
#: Both locations are tried.
#:
#: The bare summary table is used rather than ``-q``: ``-q`` wants an explicit device
#: list, and combines -d ... -c into "Invalid combination of input arguments" on driver
#: 1.3.2. The summary prints name, bus id, temperature, power, memory and utilisation
#: for every device in one fixed-width table, which is exactly what the parser reads.
_THEAD = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "P=$(command -v ppu-smi 2>/dev/null); "
    '[ -n "$P" ] || { [ -x /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi ] '
    "&& P=/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi; }; "
    "if [ -n \"$P\" ]; then echo '##THEAD'; "
    "out=$($T \"$P\" 2>/dev/null); [ $? -eq 124 ] && echo '##STALLED:thead:6s' || "
    "printf '%s\\n' \"$out\" | head -{tile_n}; fi"
)
#: 天数智芯 (Iluvatar) BI-V150. ``ixsmi`` ships under ``/usr/local/corex/bin`` and is
#: **not** on ``PATH`` — same trap as T-Head. Worse, it links against the corex
#: runtime in ``/usr/local/corex/lib`` and fails with a loader error without it, so the
#: library path is set for the SDK location rather than relying on the environment a
#: login shell happens to have.
#: 海光 (Hygon) DCU, C-3000 family. `hy-smi` ships under `/opt/hyhal/bin` — not on
#: PATH — and prints everything *rate*-like as a percentage: `VRAM%` is 73 on every
#: card of a machine where the real figures are 104.9 GiB of 144 GiB. So `hy-smi` is
#: asked only for what it is the only source of (temperature, power, HCU%), and the
#: memory comes from sysfs, which reports exact bytes per card.
#:
#: `hy-smi` has no process table at all (no --showapps, no --showprocess). The kernel
#: does: every PID with a queue is listed under /sys/class/kfd/kfd/proc, which is the
#: same information rocm-smi reads.
_HYGON = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "H=$(command -v hy-smi 2>/dev/null); "
    '[ -n "$H" ] || { [ -x /opt/hyhal/bin/hy-smi ] && H=/opt/hyhal/bin/hy-smi; }; '
    "if [ -n \"$H\" ]; then echo '##HYGON'; "
    "out=$($T \"$H\" 2>/dev/null); [ $? -eq 124 ] && echo '##STALLED:hygon:6s' || "
    "printf '%s\\n' \"$out\" | head -{tile_n}; "
    # sysfs: one line per card, "<card> <used_bytes> <total_bytes>"; ordered by card
    # number so it can be zipped against the hy-smi rows.
    "echo '##HYGON-VRAM'; "
    "for c in /sys/class/drm/card*/device; do "
    '[ -r "$c/mem_info_vram_total" ] || continue; '
    'echo "$(basename $(dirname $c)) $(cat $c/mem_info_vram_used) $(cat $c/mem_info_vram_total)"; '
    "done 2>/dev/null | sort -V; "
    # KFD: the processes holding a GPU queue, with their command line.
    "echo '##HYGON-PROCS'; "
    "for q in /sys/class/kfd/kfd/proc/*/; do "
    '[ -d "$q" ] || continue; p=${q%/}; p=${p##*/}; '
    'u=$(stat -c %U "/proc/$p" 2>/dev/null || ps -o user= -p $p 2>/dev/null); '
    "c=$(tr '\\0' ' ' < \"/proc/$p/cmdline\" 2>/dev/null | cut -c1-64); "
    '[ -n "$c" ] && echo "$p $u $c"; '
    "done 2>/dev/null | head -40; "
    # The SKU string lives only here: `Card Series: BW1101`, which is the model people
    # ask for. The summary table above has no name column at all.
    "echo '##HYGON-MODEL'; $T \"$H\" --showproductname 2>/dev/null | head -4; fi"
)
#: 天数智芯 (Iluvatar) BI-V150. Two traps, both seen on the live fleet:
#:
#: * **``ixsmi`` is not on ``PATH``** and the install directory is *versioned* on some
#:   hosts and not on others: u101 has ``/usr/local/corex/bin/ixsmi``, u180 has
#:   ``/usr/local/corex-4.5.0.20260509/bin/ixsmi`` and no unversioned symlink at all.
#:   Hard-coding one path meant u180's 16 cards were invisible while its neighbours
#:   worked — so every ``corex*`` directory is tried, newest version last-sorted first.
#: * **The library path must match the binary.** ``ixsmi`` links against ``lib`` in the
#:   *same* tree; pointing ``LD_LIBRARY_PATH`` at a different version fails to load.
#:   The root is derived from whichever ``ixsmi`` was found rather than hard-coded.
_ILUVATAR = (
    "T=timeout 6; command -v timeout >/dev/null 2>&1 || T=; "
    "X=$(command -v ixsmi 2>/dev/null); "
    'if [ -z "$X" ]; then '
    'for c in /usr/local/corex*/bin/ixsmi; do [ -x "$c" ] && X=$c; done; '
    "fi; "
    "if [ -n \"$X\" ]; then echo '##ILUVATAR'; "
    # root = the directory holding bin/ — lib/ sits beside it
    "R=${X%/bin/*}; "
    'out=$(LD_LIBRARY_PATH=$R/lib:$R/lib64:${LD_LIBRARY_PATH:-} $T "$X" 2>/dev/null); '
    "[ $? -eq 124 ] && echo '##STALLED:iluvatar:6s' || "
    "printf '%s\\n' \"$out\" | head -{tile_n}; fi"
)

#: Families whose snippet prints ``##<FAMILY>-WIDE`` (a per-type query layout rather
#: than the vendor's default table). The parser reads the marker: a host on an older
#: driver that fell back to the table must not be parsed as if it were wide.
WIDE_FAMILIES = frozenset({"metax"})

#: Family key -> snippet template. Insertion order is the order they are tried.
ACCEL_SNIPPETS: dict[str, str] = {
    "nvidia": _NVIDIA_WIDE,
    "ascend": _ASCEND,
    "kunlunxin": _KUNLUNXIN,
    "metax": _METAX,
    "enflame": _ENFLAME,
    "mthreads": _MTHREADS,
    "tsingmicro": _TSINGMICRO,
    "thead": _THEAD,
    "iluvatar": _ILUVATAR,
    "hygon": _HYGON,
}


@dataclass(frozen=True)
class ProbeLimits:
    """How much of each section to pull back.

    What keeps one node from costing the poll loop more than the others: ``ps`` is
    capped at ``top_n`` rows and each vendor table at ``tile_n`` lines, which on an
    8-card host still leaves room for the card rows *and* their process rows.
    """

    top_n: int = 6
    tile_n: int = 260
    half: int = 90


def _fill(template: str, lim: ProbeLimits) -> str:
    """Substitute the budget placeholders.

    Not ``str.format``: the shell templates are full of awk field references like
    ``{t=$2}``, which ``format`` would read as placeholders. Only these three names
    are ever substituted.
    """
    return (
        template.replace("{top_n}", str(lim.top_n))
        .replace("{tile_n}", str(lim.tile_n))
        .replace("{half}", str(lim.half))
    )


def build_script(
    kind: str = "fleet",
    *,
    limits: ProbeLimits | None = None,
    nvidia_tile: bool = False,
) -> str:
    """Return the remote bash script for ``kind`` (``cpu`` / ``fleet`` / ``gpu``).

    The script never exits non-zero on a missing tool: every command is guarded, so
    the reply is always a complete report and "this host has no accelerators" is a
    normal answer rather than an error.
    """
    lim = limits or ProbeLimits()
    if kind == "cpu":
        return node_script(M_CPU, top_n=lim.top_n)
    if kind == "fleet":
        parts = [node_script(M_SYSTEM, top_n=lim.top_n)]
    elif kind == "gpu":
        parts = []
    else:
        raise ValueError(f"unknown probe kind {kind!r}")

    for fam, template in ACCEL_SNIPPETS.items():
        if fam == "nvidia" and nvidia_tile:
            template = _NVIDIA_TILE
        parts.append(_fill(template, lim))
    return "\n".join(parts) + "\n"


#: Printed by ``ping`` and required back in the reply: a KoKo rejection (no permission,
#: asset offline, bad address) exits **0** with the error on stdout, so the exit status
#: alone reads as success. Same sentinel pattern as jms-bridge, for the same reason.
SENTINEL = "jms-top-ok"
PING_SCRIPT = f"echo {SENTINEL}\n"


def with_sentinel(script: str) -> str:
    """Put the reachability sentinel on the first line of a probe script."""
    return f"echo {SENTINEL}\n{script}"
