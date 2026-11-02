# Architecture

```
 hosts.yaml (~/.config/jms-top)   JumpServer KoKo           vendor SMI on the host
        │                               │                          │
        ▼                               ▼                          ▼
  inventory.py ──► monitor.FleetMonitor ──► ssh.SSHRunner ──► probe.py (bash)
                          │                                        │
                          │        reply: "jms-top-ok\n##SYS … ##NVIDIA …"
                          ▼                                        ▼
                     scan.parse_reply ─────────────────► families.py (one parser per vendor)
                          │
                          ▼
                     model.HostView ──► tui.JmsTopApp ──► panels.py (Rich)
                          │
                          └──► render.py (--once text)  cli.snapshot (--json)
```

Each layer has one job, and the expensive one (SSH) is confined to a single class.

| module | responsibility | never does |
|---|---|---|
| `config.py` | read `config.toml`, cache the token at 0600, write starters | talk to the network |
| `auth.py` | REST login, asset discovery | store a password unless asked |
| `inventory.py` | `hosts.yaml` → `Host`, selectors, KoKo login strings | dial out |
| `ssh.py` | ControlMaster data plane, the reachability sentinel | parse anything |
| `probe.py` | the remote bash: what to ask each vendor tool for | parse anything |
| `families.py` | per-vendor parsing → `Card` / `Proc` | touch the network |
| `scan.py` | split a reply into sections and fill a `HostView` | judge a host |
| `model.py` | verdicts, roll-ups, sorting (`HostView.verdict`) | do I/O |
| `monitor.py` | scheduling: two-speed ticks, backoff, pause | render |
| `panels.py` / `render.py` | text and Rich rendering | mutate models |
| `tui.py` | keys, filters, cursor, repaint policy | wait on the network |

## Why two speeds

A tick is a round trip through the bastion per host. The full probe (vendor SMI +
`ps` + `df`) costs 1-20 s per host depending on the vendor: `npu-smi info` on an 8-card
Ascend host alone takes ~6 s, because it queries every chip's sensors one by one and
cannot be batched (`-t <type>` demands a card id, which would be eight more round
trips). The cheap probe (a handful of `/proc` reads) costs ~100 ms.

Measured on the 19-host fleet, fan-out 8 (2026-10-21):

| probe | wall time |
|---|---|
| `ping` (one `echo`, warm ControlMaster) | 3-8 s |
| cheap tick (all hosts) | 1.2-3.4 s |
| full tick (all hosts) | 20 s |

So the screen refreshes CPU on every tick and card tables on a per-host timer. A host
with nothing free is re-read three times less often: a busy card stays busy, and the
only transition worth catching immediately is "free → busy", which the cheap tick
notices and promotes to a full read on the next tick.

## Why a missing value is `None` and not `0`

Every layer is written so that "not reported" cannot become "zero":

* `families.Card.mem_frac` is `None` when either figure is missing, and `Card.busy`
  returns `None` when *no* signal is available — rendered as `unknown` (magenta), and
  never counted in `free_cards`.
* `scan.parse_reply` builds a **fresh** `NodeStats` per reply, so a field the reply did
  not carry is absent rather than stale.
* `render.DASH` is the only way a missing value is printed.

This is not pedantry. On the live fleet, `p800-15` reports 0% utilisation on cards
holding 87% of memory, and `tsingmicro-137` reports 0% on all 32 chips while 90% of
each chip's HBM is taken. Utilisation alone would call both fleets idle.

## Extension points

* **A new vendor**: add a snippet in `probe.ACCEL_SNIPPETS` (marker + `command -v`
  guard + `head`), a parser class in `families.FAMILIES`, and a verbatim sample in
  `tests/fixtures/smi`. Nothing else changes — the probe decides which family ran by
  the marker it printed, not by the inventory's guess.
* **A new view**: `panels.py` takes a `HostView` and returns Rich markup; `tui.py`
  only decides *when* to call it.
* **Another transport**: `FleetMonitor` takes any object with
  `async exec_on(host, script, *, timeout, workdir, env_setup)`; `tests/test_monitor.py`
  ships the fake one the tests use.
## Why the terminal is guarded outside Textual

`terminal.py` exists because Textual restores the terminal only when it gets to run its
own shutdown, and the failure modes a long-running monitor meets are exactly the ones
where it does not:

* **A fatal signal.** `SIGHUP` (closed laptop, closed window, lost session) has default
  action *terminate*; Python does not intercept it and Textual does not either. The
  process dies with the tty in raw mode, on the alternate screen, with echoes off.
* **A terminal that is replaced, not closed.** When a tmux server exits or a session is
  reset, the app keeps descriptors to a pty with no reader: reads return EOF, writes
  `EIO`. Textual's input thread ends quietly and the app stays up with no input path.

So `JmsTopApp.run` installs three things before Textual takes the terminal:

1. **The guard** — `SIGHUP`/`SIGTERM` restore the terminal (leave the alternate screen,
   show the cursor, mouse/kitty modes off, echo and line discipline back on) and then
   re-raise with the default action, so the exit status still reports the signal. It
   restores again at `atexit`, through a fresh non-blocking `dup` of the tty, so a
   restore can never be stuck behind Textual's own (possibly dead) writer thread.
2. **The supervisor** — one thread, one check a second, outside the event loop: the
   app's writer thread is dead, or a probe write to the tty raises. Either means the
   terminal is gone, so the app is asked to exit (closing its SSH ControlMasters) and
   the process is ended 4 s later regardless, because a stuck thread in the writer must
   not keep 77 connections alive with no screen to show for them. A resize to zero is
   deliberately *not* a trigger: macOS reports it while the app is suspended, and
   quitting there would fight the shell.
3. **The resilient selector** — the object Textual's input thread calls `select` on.
   Once the output has failed it stops asking the kernel and reports nothing, so a
   closed pty (permanently readable, permanently EOF) cannot spin the thread or raise
   into the app.

`tests/test_terminal.py` drives the real app on a real pty for the two cases that can
be reproduced deterministically — a fatal signal, and the pty going away — and asserts
both the exit path and the tty's settings afterwards.
