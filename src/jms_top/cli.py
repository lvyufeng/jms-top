"""``jms-top`` entry point: the TUI, plus the non-interactive modes that make it scriptable.

The sub-commands exist so the monitor is usable where a terminal is not:

* ``jms-top --once`` / ``--json``: one round, for cron, CI, or ``jq``. This is the
  "which hosts are free right now" a shell script or an agent wants.
* ``jms-top --ping``: liveness only, ~1s on a warm bastion connection.
* ``jms-top --hosts a,b --json``: snapshot a few machines by name / group / tag.
* ``jms-top --demo``: the TUI against the recorded fleet in ``tests/fixtures``, so the
  interface can be explored (and demoed) with no bastion at all.

And four verbs for the first run, so nothing has to be set up by hand::

    jms-top init      # write ~/.config/jms-top/{config.toml,hosts.yaml}
    jms-top login     # username + password -> cached session token
    jms-top hosts     # list what is configured; --discover drafts it from JumpServer
    jms-top logout    # drop the cached token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
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
  jms-top init                  # first run: write a starter config + inventory
  jms-top login                 # log in to JumpServer (password -> token)
  jms-top hosts --discover      # draft the inventory from the assets you can reach
  jms-top                       # every host in the inventory
  jms-top -t ascend,gpu         # a group or tag
  jms-top --once --json | jq    # one round, machine-readable
  jms-top --ping                # liveness only
  jms-top --demo                # the TUI on recorded data, no bastion needed
""",
    )
    p.add_argument(
        "command",
        nargs="?",
        choices=["init", "login", "logout", "hosts"],
        help="set-up verbs: init / login / logout / hosts (omit to open the TUI)",
    )
    p.add_argument("hosts", nargs="*", help="host names / groups / tags (default: all)")

    # --- `login` / `logout` / `hosts` options ---
    p.add_argument("-u", "--user", help="login: JumpServer username")
    p.add_argument("-p", "--password", help="login: password (else prompt / env / saved)")
    p.add_argument(
        "--save-password",
        action="store_true",
        help="login: store the password in a 0600 file for next time",
    )
    p.add_argument(
        "--forget-password", action="store_true", help="logout: also delete the saved password"
    )
    p.add_argument(
        "--discover",
        action="store_true",
        help="hosts: list the assets this account can reach (and draft an inventory)",
    )
    p.add_argument(
        "--write",
        metavar="PATH",
        help="hosts --discover: write the drafted inventory here (default: show it)",
    )
    p.add_argument("--force", action="store_true", help="init: overwrite existing files")
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
        "--watch",
        action="store_true",
        help="hot-reload the UI when its sources change (default outside --demo)",
    )
    p.add_argument(
        "--no-watch", action="store_true", help="disable hot reload even in a dev checkout"
    )
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

    if args.command == "init":
        return _cmd_init(args)
    if args.command in {"login", "logout", "hosts"}:
        return asyncio.run(_cmd_auth(args))
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

        return run_demo(
            interval=args.interval, once=args.once, as_json=args.json, watch=_watch_enabled(args)
        )

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
    JmsTopApp(monitor, watch=_watch_enabled(args)).run()
    return 0


def _watch_enabled(args) -> bool:  # noqa: ANN001 - argparse.Namespace
    """Hot reload is on by default unless told otherwise.

    The cost is one ``stat`` per source file per second and nothing else, and the
    benefit — an edit landing while you watch — is only useful if you do not have to
    know about a flag to get it. ``--no-watch`` exists for the case where the running
    code must not move under an operator's feet.
    """
    return not args.no_watch


# --- set-up verbs ---------------------------------------------------------------


def _cmd_init(args) -> int:  # noqa: ANN001 - argparse.Namespace
    """Write the config + inventory templates, never clobbering what is there."""
    from .config import config_path, inventory_path, write_starter_files

    written = write_starter_files(force=args.force)
    config, inv = config_path(), inventory_path()
    if not written:
        print(f"already configured:\n  {config}\n  {inv}\n(nothing to do; --force to rewrite)")
        return 0
    for path in written:
        print(f"wrote {path}")
    print(
        "\nnext:\n"
        f"  1. set bastion.url and bastion.login in {config}\n"
        "  2. jms-top login        (or ssh keys, if JumpServer already knows yours)\n"
        "  3. jms-top hosts --discover --write   (build the inventory from your assets)"
    )
    return 0


async def _cmd_auth(args) -> int:  # noqa: ANN001 - argparse.Namespace
    from .config import ConfigError, load_settings

    if args.command == "init":  # pragma: no cover - init is synchronous
        return _cmd_init(args)

    try:
        settings = load_settings(require_url=args.command != "logout")
    except ConfigError as exc:
        print(f"jms-top: {exc}", file=sys.stderr)
        return 2

    if args.command == "logout":
        return _cmd_logout(args)
    if args.command == "hosts":
        return await _cmd_hosts(args, settings)
    return await _cmd_login(args, settings)


def _cmd_logout(args) -> int:  # noqa: ANN001
    from .config import clear_token, credentials_path, forget_password

    cleared = clear_token()
    print("session token cleared" if cleared else "no cached session token")
    if args.forget_password:
        print("saved password deleted" if forget_password() else "no saved password")
    elif credentials_path().exists():
        print(f"a saved password is still at {credentials_path()} (--forget-password to delete)")
    return 0


async def _cmd_login(args, settings) -> int:  # noqa: ANN001
    import getpass
    import os

    from .auth import AuthError, login
    from .config import ConfigError, _saved_password

    username = args.user or settings.bastion.login or os.environ.get("JMS_TOP_USER")
    password = args.password or os.environ.get("JMS_TOP_PASSWORD") or _saved_password()
    if not username:
        username = input("JumpServer username: ").strip()
    if not password:
        if not sys.stdin.isatty():
            print(
                "jms-top: no password — pass -p, set JMS_TOP_PASSWORD, "
                "or log in once with --save-password",
                file=sys.stderr,
            )
            return 2
        # getpass keeps it off the screen and out of the shell history, which is what
        # a -p argument cannot promise.
        password = getpass.getpass(f"JumpServer password for {username}: ")
    if not username or not password:
        print("jms-top: need a username and a password", file=sys.stderr)
        return 2

    try:
        result = await login(settings, username, password, save=args.save_password)
    except (AuthError, ConfigError) as exc:
        print(f"jms-top: login failed: {exc}", file=sys.stderr)
        return 1
    print(f"logged in as {result.username}")
    print(f"session token cached in {result.config_file} (mode 600)")
    if result.password_saved:
        print(f"password saved to {result.password_saved}")
    print(
        "\nthe SSH data plane still needs a way in: a key JumpServer knows (ssh-key) "
        "or [bastion] ssh_password"
    )
    return 0


async def _cmd_hosts(args, settings) -> int:  # noqa: ANN001
    from .auth import AuthError, discover_assets
    from .config import ConfigError, write_private
    from .inventory import Inventory, InventoryError, draft_inventory

    if args.discover:
        try:
            assets = await discover_assets(settings)
        except (AuthError, ConfigError) as exc:
            print(f"jms-top: cannot list assets: {exc}", file=sys.stderr)
            return 1
        if not assets:
            print("no assets visible to this account", file=sys.stderr)
            return 1
        for asset in assets:
            accounts = ",".join(asset.accounts) or "-"
            print(f"{asset.name:<28} {accounts:<12} {asset.id}  {asset.address or ''}")
        print(f"\n{len(assets)} assets")
        text = draft_inventory(
            [
                {
                    "id": a.id,
                    "name": a.name,
                    "address": a.address,
                    "platform": a.platform,
                    "tags": a.tags,
                    "account": a.accounts[0] if a.accounts else None,
                }
                for a in assets
            ]
        )
        if args.write:
            write_private(pathlib.Path(args.write).expanduser(), text)
            print(f"drafted inventory written to {args.write}")
        else:
            print("\n--- draft inventory (pass --write PATH to save) ---")
            print(text)
        return 0

    try:
        inventory = Inventory.load()
    except InventoryError as exc:
        print(f"jms-top: {exc}", file=sys.stderr)
        return 2
    for host in inventory.hosts.values():
        group = inventory.group_of(host.name) or "-"
        print(f"{host.name:<28} {group:<12} {host.platform:<14} {host.asset_id or '-'}")
    print(f"\n{len(inventory.hosts)} hosts from {inventory.path}")
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
