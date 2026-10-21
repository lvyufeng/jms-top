"""``jms-top`` entry point: the TUI, plus the non-interactive modes that make it scriptable.

The sub-commands exist so the monitor is usable where a terminal is not:

* ``jms-top --once`` / ``--json``: one round, for cron, CI, or ``jq``. This is the
  "which hosts are free right now" a shell script or an agent wants.
* ``jms-top --ping``: liveness only, ~1s on a warm bastion connection.
* ``jms-top --hosts a,b --json``: snapshot a few machines by name / group / tag.
* ``jms-top --demo``: the TUI against the recorded fleet in ``tests/fixtures``, so the
  interface can be explored (and demoed) with no bastion at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone

from . import __version__
from .model import FleetView


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jms-top",
        description="nvitop-style live view of a JumpServer fleet: which machines "
        "are free, right now.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  jms-top                       # every host in the inventory
  jms-top -t ascend,gpu         # a group or tag, as jms-bridge selectors
  jms-top -i 3                  # poll every 3s instead of 5s
  jms-top --once --json | jq    # one round, machine-readable
  jms-top --ping                # liveness only
  jms-top --demo                # the TUI on recorded data, no bastion needed
""",
    )
    p.add_argument("hosts", nargs="*", help="host names / groups / tags (default: all)")
    p.add_argument("-t", "--target", action="append", default=[], help="selector (repeatable)")
    p.add_argument(
        "-i", "--interval", type=float, default=3.0, help="CPU poll interval, seconds (default 3)"
    )
    p.add_argument(
        "--timeout", type=float, default=45.0, help="per-host probe timeout (default 45)"
    )
    p.add_argument(
        "-f",
        "--full-interval",
        type=float,
        default=None,
        help="card-table refresh, seconds (default 2x interval)",
    )
    p.add_argument("--once", action="store_true", help="one round, then exit")
    p.add_argument("--json", action="store_true", help="with --once/--ping: JSON on stdout")
    p.add_argument("--ping", action="store_true", help="liveness check only")
    p.add_argument("--demo", action="store_true", help="replay the recorded fleet (no SSH)")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colour in text output")
    p.add_argument(
        "--screenshot",
        metavar="PATH.svg",
        help="render one TUI frame to an SVG and exit (for docs; needs --demo or a bastion)",
    )
    p.add_argument(
        "--nvidia-tile",
        action="store_true",
        help="parse nvidia-smi's ASCII tile instead of --query-gpu",
    )
    p.add_argument("--version", action="version", version=f"jms-top {__version__}")
    return p


def _exit_on_broken_pipe() -> None:
    """`jms-top --json | head` is a normal invocation, not a crash.

    Python turns the closed pipe into ``BrokenPipeError`` on the final flush, which
    normally surfaces *after* main() returns as a traceback on stderr. Pointing
    SIGPIPE back at its default (what every POSIX CLI does) makes the shell see the
    process die on the signal, silently — and the exit status stays non-zero, so a
    script still notices.
    """
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main(argv: list[str] | None = None) -> int:
    _exit_on_broken_pipe()
    args = _build_parser().parse_args(argv)
    # `jms-top metax,ascend` reads as two selectors to anyone who has used a shell, so
    # commas split — the same rule an inventory selector already follows.
    targets = [
        part
        for value in list(args.target) + list(args.hosts)
        for part in (t.strip() for t in value.split(","))
        if part
    ]

    if args.demo:
        from .demo import demo_monitor

        if args.screenshot:
            from .tui import screenshot

            asyncio.run(
                screenshot(demo_monitor(interval=args.interval), args.screenshot, demo=True)
            )
            return 0
        from .demo import run_demo

        return run_demo(interval=args.interval, once=args.once, as_json=args.json)

    from .monitor import FleetMonitor, MonitorError

    if args.once or args.json or args.ping:
        return asyncio.run(_one_shot(args, targets))

    from .tui import JmsTopApp, screenshot

    if args.screenshot:
        try:
            monitor = FleetMonitor(
                targets=targets or None,
                interval=args.interval,
                full_interval=args.full_interval,
                timeout=args.timeout,
                nvidia_tile=args.nvidia_tile,
            )
        except MonitorError as exc:
            print(f"jms-top: {exc}", file=sys.stderr)
            return 2
        asyncio.run(screenshot(monitor, args.screenshot))
        return 0

    try:
        monitor = FleetMonitor(
            targets=targets or None,
            interval=args.interval,
            full_interval=args.full_interval,
            timeout=args.timeout,
            nvidia_tile=args.nvidia_tile,
        )
    except MonitorError as exc:
        print(f"jms-top: {exc}", file=sys.stderr)
        return 2
    JmsTopApp(monitor).run()
    return 0


async def _one_shot(args, targets: list[str]) -> int:
    from .monitor import FleetMonitor, MonitorError

    try:
        monitor = FleetMonitor(
            targets=targets or None,
            interval=args.interval,
            full_interval=args.full_interval,
            timeout=args.timeout,
            nvidia_tile=args.nvidia_tile,
        )
    except MonitorError as exc:
        print(f"jms-top: {exc}", file=sys.stderr)
        return 2
    try:
        if args.ping:
            await monitor.ping_all()
            out = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "hosts": [
                    {"host": h.name, "reachable": h.reachable, "error": h.error}
                    for h in monitor.view.hosts
                ],
            }
            if args.json:
                print(json.dumps(out, indent=2))
            else:
                for h in monitor.view.hosts:
                    mark = "up  " if h.reachable else "DOWN"
                    print(f"{mark}  {h.name:<28}{h.error or ''}")
            return 0 if all(h["reachable"] for h in out["hosts"]) else 1

        return await run_monitor_once(monitor, as_json=args.json)
    finally:
        await monitor.close()


async def run_monitor_once(monitor, *, as_json: bool = False, color: bool | None = None) -> int:
    """One full round printed either as the table or as the ``--json`` document."""
    from .render import render_table

    await monitor.poll()
    view = monitor.view
    if as_json:
        print(json.dumps(snapshot(view), indent=2))
        return 0
    use_colour = sys.stdout.isatty() if color is None else color
    print(render_table(view, color=use_colour))
    return 0


def snapshot(view: FleetView) -> dict:
    """The ``--json`` document: a stable shape that scripts can rely on."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interval": view.interval,
        "summary": {
            "hosts": len(view.hosts),
            "reachable": view.reachable,
            "free_hosts": [h.name for h in view.free_hosts],
            "free_cards": view.free_cards,
            "busy_cards": view.busy_cards,
            "unknown_cards": view.unknown_cards,
            "total_cards": view.total_cards,
            "families": view.families,
        },
        "hosts": [
            {
                **h.to_dict(),
                "group": h.group,
                "users": h.users,
                "top": h.node.top,
            }
            for h in sorted(view.hosts, key=lambda h: h.sort_key)
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
