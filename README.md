# jms-top

A live terminal view of a JumpServer fleet — nvitop's feel, for the one question that
matters before you start a job: **which of my machines can I actually take right now?**

![jms-top on the recorded fleet](docs/fleet.svg)

```
  jms-top    59/152 cards free  (39%)   5 free hosts   12 residual  │ 18/19 up │ next tick in 2.1s · full 4s ago
 Ascend 48   NVIDIA 32   Enflame 16   MetaX 16   MooreThreads 16   Kunlunxin 16   TsingMicro 8
 HOST                ADDRESS       KIND           CARD #             CARDS  STATE      LOAD   MEM   CPU   UP      USERS               AGE
 a800-171            10.0.0.12   NVIDIA         FFFFFFFF            8/8    FREE        2%    8%  39%   126d7h  root                0s
 gcu-s60             Enflame        FFFFFFFF            8/8    FREE        9%    3%   0%   120d8h  root                0s
 h800-65             NVIDIA         FFFFFFFF            8/8    FREE        1%   17%  12%   135d2h  root                0s
 mthreads-16         MooreThreads   FFFFFFFF            8/8    FREE        0%    2%   1%   115d20h root                0s
 ascend-910c-108     Ascend         BBBBFFBBFFFFFFFF   10/16   PARTIAL     5%   17%   2%   79d1h   pid 2510883,pid 2   0s
 p800-13             Kunlunxin      BFFBFFFF            6/8    PARTIAL     1%    5%   0%   72d3h   pid 2598031,pid 2   0s
 metax-mc550-121     NVIDIA         —                   —      UNKNOWN     5%    9%   0%   134d0h  root                0s
 h100-17             NVIDIA         BBBBBBBB            0/8    BUSY        6%   25%   4%   76d21h  pid 1062720,pid     0s
 tsingmicro-137      TsingMicro     ~~~~~~~~            0/8    RESIDUAL    1%    8%   0%   5d5h    root                0s
 dcu-bw1000-3        —              —                   —      DOWN        —     —    —    —       -                   —
```

`CARD #` is one character per accelerator, so "can I get four cards on one host?" is a
glance rather than eight numbers read one at a time.

| symbol | means |
|---|---|
| `F` | free — nothing running, memory clear |
| `B` | busy — a process holds it, or its memory is committed |
| `~` | **residual** — no process listed but memory is held (a leaked or paused job) |
| `?` | unknown — the vendor tool did not answer; **never** reported as free |
| `—` | not reported at all |

## Install

```bash
git clone https://github.com/lvyufeng/jms-top && cd jms-top
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
ln -s "$PWD/.venv/bin/jms-top" ~/.local/bin/jms-top   # so `jms-top` works anywhere
```

Or with [pipx](https://pipx.pypa.io) if you do not want a dev checkout:

```bash
pipx install .
```

**Self-contained.** No other package is needed: config, login, the KoKo SSH data
plane and every vendor parser live in this repo. Its own files are:

* `~/.config/jms-top/config.toml` — bastion URL, KoKo port, your login, cached token;
* `~/.config/jms-top/hosts.yaml` — which machines exist (name → asset id, account,
  platform, tags).

Both are mode 600 in a 700 directory, and both can be relocated with `JMS_TOP_CONFIG` /
`JMS_TOP_INVENTORY`. Nothing else on the machine has to be configured.

## First run

Three steps, once.

```bash
jms-top init          # write ~/.config/jms-top/{config.toml,hosts.yaml}
```

Then edit `config.toml` — it needs two lines:

```toml
[bastion]
url = "https://jumpserver.example.com"   # the web UI's address
ssh_port = 2222                          # KoKo's SSH port (some use 2224)
login = "your-login"                     # your JumpServer *login*, not the OS account
```

### Logging in

The two planes authenticate differently, and it is worth knowing which is which:

```bash
jms-top login            # JumpServer username + password -> cached session token
```

`login` prompts for the password (never echoed, never in your shell history), exchanges
it at `POST /api/v1/authentication/auth/`, and caches the returned **session token** in
`config.toml`. Every later run reuses it, so you log in once per token lifetime.
`jms-top logout` drops it; `--save-password` keeps the password in a 0600 file if you
would rather not type it again.

If your KoKo also accepts password auth, put the password under
`[bastion] ssh_password` (or `JMS_TOP_SSH_PASSWORD`) and install `sshpass`. **An SSH key
is better**: register a key with JumpServer and skip that field entirely.

```bash
jms-top hosts --discover          # assets you can log in to, from the gateway itself
jms-top hosts --discover --write  # ...and write hosts.yaml from them
```

`--discover` asks **KoKo** (the same SSH gateway the poller uses) for the assets your
account may log in to, so it needs no token and lists exactly the machines that will
actually answer. `--discover-api` asks the REST API instead, which returns everything
you are *permitted*, including assets that are under repair or not yet activated — a
superset that shows up as `DOWN`. The KoKo list is the one to use.

Asset names like `bm-baai-dx-zone1-d-a100-40g-2-42` become short keys (`a100-42`) with
the rest kept as tags and groups (`jms-top a100`, `jms-top zone1`).

Check it with `jms-top --ping` (liveness only, seconds) before opening the UI.

If `--ping` says hosts are DOWN, the message on each row is KoKo's own words — "no
permission", "asset offline", or a refused password — and `jms-top hosts` shows what was
actually configured.

### Migrating from a jms-bridge inventory

`jms-top` reads the same `hosts.yaml` *shape* (name → `asset_id`/`account`/`platform`/
`tags`, plus groups), but its own copy, under `~/.config/jms-top/hosts.yaml`. Copy the
hosts and groups across; jms-top ignores the fields it does not use (`env_setup`,
`workdir`, `workspaces`). Or regenerate it from JumpServer itself with
`jms-top hosts --discover --write`.

## Quick start

```bash
jms-top
```

Type `q` to quit. The screen starts with `starting…` in the title bar: it pings every
host, then runs a first full round. On the 19-machine fleet here that is **5–7 s**;
afterwards the CPU side refreshes every 3 s and each host's card table on its own
schedule.

### Choosing what to watch

```bash
jms-top                     # everything in the inventory
jms-top ascend              # one group
jms-top metax,ascend        # several selectors, comma-separated
jms-top h100-17 gpu         # a host name and a tag together
jms-top -t ascend -t metax  # same thing, spelled the long way
jms-top 'metax, p800'       # spaces are fine too — quote them for the shell
```

A selector is a **host name**, a **group name**, or a **tag** — whatever appears in
`hosts.yaml`. An unknown selector is an error that lists the known names, rather than a
silently empty screen.

### Not opening the UI

```bash
jms-top --once                 # one full round, printed as a table
jms-top --ping                 # is everything reachable? (seconds, not a full round)
jms-top -t h100-17 --once      # one host, scriptable
jms-top --once --json | jq .summary
```

`--json` emits the whole fleet: per host its verdict, cards, processes, users, load,
memory, and the timestamp the numbers are from. That is what to pipe into a scheduler
or an alert rather than parsing the table.

### Without a bastion

```bash
jms-top --demo                 # the real UI on a recorded fleet — no SSH, no account
jms-top --demo --once
```

Useful for trying the keys, for screenshots, and for committing to the repo: the demo
replays `tests/fixtures/demo/fleet.json` through the same parsers and the same app, so
what you see is what the real thing does with that data.

### Speed

```bash
jms-top -i 2                   # a 2 s tick (also: press `i` / `o` while running)
jms-top -f 10                  # re-read card tables every 10 s
jms-top --timeout 20           # lower the per-SSH-command ceiling
```

`-i` is the CPU tick and `-f` the card tick; `-f` defaults to twice `-i`. While running,
`i` multiplies the interval by 2 and `o` divides it by 2 (clamped to 1–60 s). Both are
per-fleet, not per-host. The defaults (3 s / 6 s) are what the measured latency supports
on a 19-host fleet — see *What the probe costs* below before going much below `-i 2`:
the CPU tick alone measured 1.2–3.4 s wall on this fleet, so a 1 s interval would just
queue rounds back to back.

## Keys

| key | does |
|---|---|
| `↑` `↓` / `j` `k` | move the selection (the side panel and detail follow immediately) |
| `f` | cycle the filter: all → free → partial → residual → busy → down/unknown |
| `/` | search across host, tag, group, platform, user, and process name — filters as you type, `enter` keeps it, `esc` closes the box |
| `enter` | open the selected host's accelerator process list |
| `esc` | close the process list, or the search box |
| `r` | run a full round now |
| `space` | pause / resume polling (the screen keeps showing what it has) |
| `i` / `o` | poll **slower** / **faster** (interval ×2 / ÷2, clamped to 1–60 s) |
| click `ADDRESS` / `KIND` | sort by address / group by accelerator (click again → back to availability) |
| `ctrl+r` | reload the UI from source now (normally automatic — see below) |
| `q` | quit |

### Hot reload

Edit anything in `panels.py`, `render.py`, `tui.py` or `app.tcss` while the app is
running and the screen picks it up within a second, **without losing the connection or
the data**: the monitor keeps its multiplexed SSH channels and its last round, and only
the widgets are re-composed around them.

```
  hot reload: reloaded panels
```

`ctrl+r` forces a reload (useful after a `git checkout` or an edit the watcher missed).
Changes to `monitor.py`, `probe.py`, `families.py` or `scan.py` are **not** hot-swapped
— those hold state that in-flight SSH commands are using — and the footer says
`restart to apply monitor/parser changes` when it sees one. `--no-watch` turns the
watcher off entirely.

## Reading the verdicts

| verdict | when |
|---|---|
| `FREE` | every card free, and the CPU side is idle (< 0.75 load per core) |
| `PARTIAL` | some cards free; or all free but the node is CPU-busy |
| `BUSY` | at least one card is in use |
| `RESIDUAL` | every card is held by memory alone — no process, no compute |
| `UNKNOWN` | the vendor tool errored or was cut off; the reason is in the detail panel |
| `DOWN` | the bastion refused the connection (no permission, asset offline) |
| `CPU` | reachable, no accelerator tool answered |

Per card: busy if a process holds it, or its health is not OK, or ≥ 50 % of its memory
is committed, or utilisation ≥ 5 %. Free only when a signal says so — no signal at all
means unknown.

Three rules shape all of this:

1. **A missing value is never a zero.** Anything not reported renders as `—`, and a
   card whose numbers did not parse is `?` (magenta), never `F` (green).
2. **A card is busy the moment its memory is committed.** Process table first, then
   memory, then utilisation — because on a shared cluster memory, not utilisation,
   decides who can start next. A vLLM stack at `0%` utilisation holding 87 % of the HBM
   is busy, and `RESIDUAL` exists to call that state out by name.
3. **Count the unit a job actually asks for.** Ascend 910C reports 2 chips per card and
   TsingMicro 4 per card; those are shown per chip (what you ask for) with the card id
   alongside.

## Accelerator support

| family | tool | reported per | processes |
|---|---|---|---|
| NVIDIA | `nvidia-smi` | GPU | yes (bus id) |
| Ascend | `npu-smi info` | chip (2/card on 910C) | yes |
| Kunlunxin | `xpu-smi` | XPU | yes |
| MetaX (沐曦) | `mx-smi` | GPU | yes — `--show-memory/usage/process`, falls back to the default table |
| Enflame | `efsmi` + `pmon` | GCU | yes |
| Moore Threads | `mthreads-gmi` | GPU | yes |
| TsingMicro | `tsm_smi` | chip (4/card on TX8110) | card-level; process table per card |
| T-Head PPU | `ppu-smi` | PPU | yes — installed under `/usr/local/PPU_SDK/ppu-smi/bin`, not on `PATH` |
| Iluvatar (天数) | `ixsmi` | GPU | yes — `/usr/local/corex/bin`, needs `LD_LIBRARY_PATH` |
| Hygon DCU | `hy-smi` | — | not parsed yet; the CPU side still works |

Which tool to run is decided by the host, not the inventory: every family is probed
behind a `command -v` guard and reports under its own `##FAMILY` marker, so a machine
that is mislabelled in `hosts.yaml` still lands on the right parser.

Every parser is pinned by a test against output captured from the real fleet
(`tests/fixtures/smi`, 2026-10-21), and the tests assert the *judgement* — which cards
are free, which are held by memory alone, which host must be `unknown` — not just the
numbers.

## What the probe costs

Measured on the live fleet, per host, for a full round:

| | |
|---|---|
| CPU consumed by the probe (incl. the vendor tool) | 0.2 – 1.4 CPU-seconds |
| share of a 256-core host, once per 6–18 s | well under 1 % |
| accelerator memory / SM time taken | none |
| bytes returned | 1–11 KB |

The probe reads counters — `/proc`, `ps`, and the vendor's management interface. It
takes no accelerator memory, creates no context, and runs no kernel, so a training job's
throughput is unaffected. What it does touch is the card's own utilisation sampling,
which is the same small perturbation `nvidia-smi` always has.

Two things keep it bounded:

* **Every vendor call is time-boxed** (6 s, 12 s for the NVIDIA query whose host was
  measured at 9–11 s). A cut-off tool reports `##STALLED`, and that host reads
  `UNKNOWN` with the reason on screen — a hung driver costs one host one round, never
  the fleet.
* **Hosts with nothing free are re-read less often** (3× the interval), because a busy
  card stays busy. The `AGE` column always says how old a card table is.

## Limits worth knowing

* **A card's holder is a pid, not a user.** The vendor tools do not report ownership and
  these hosts run everything as `root`. Use the bastion's audit log to learn who it was.
* **The CPU tick cannot see a card go idle.** A host's card verdict is at most one full
  interval old; `AGE` says exactly how old.
* **`nvidia-smi` is slow on some hosts** (9–11 s on one H800 here, against 0.3 s
  elsewhere), so the GPU query gets 12 s; a slower answer is reported as `UNKNOWN`
  rather than guessed.
* **Some process tables lie.** TsingMicro declines to list processes while 32 chips
  report "taskNum exceeds max", so those cards are judged on memory alone.

## Development

```bash
pytest                 # 59 tests, no cluster and no bastion required
ruff check src tests
jms-top --demo         # the recorded fleet, in the real UI
```

`docs/architecture.md` explains the layering and why there are two poll speeds.
`tests/fixtures/demo/fleet.json` is a captured fleet (both probe shapes, all reachable
hosts, addresses replaced with placeholders); `demo.py` replays it through the same
parsers and app as live data, so neither the tests nor the demo can drift from it.