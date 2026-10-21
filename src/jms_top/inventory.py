"""The inventory: which machines exist, and how to address each one through KoKo.

One entry per machine::

    hosts:
      ascend-910c-108:
        asset_id: fce4b403-72d5-4b82-a1a7-22efe74dfcd0
        account: root
        platform: ascend-910c
        tags: [npu, ascend]

    groups:
      ascend: [ascend-910c-108]

Only ``asset_id`` and ``account`` are needed to address a machine; everything else is
display metadata. ``platform`` is a *hint* for the UI and for which vendor tool to
prefer — it is never trusted to decide what a host runs, because the probe asks the
host itself (see :mod:`jms_top.probe`).

A host can also be reached by plain SSH (``ssh_host``), which is how the same view can
cover machines that are not behind JumpServer at all. That path shares the rest of the
code and is useful for local development.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import ConfigError, inventory_path


class InventoryError(RuntimeError):
    """The inventory is missing or malformed."""


@dataclass
class Host:
    """One addressable machine."""

    name: str
    #: JumpServer asset UUID. May be omitted when ``ssh_host`` is set.
    asset_id: str | None = None
    #: The OS account to log in as on the far side (the ``#account`` in the login).
    account: str | None = None
    #: Per-host override of the bastion login user (defaults to ``bastion.login``).
    login: str | None = None
    #: The machine's own address, as JumpServer knows it. Shown in the UI and used as
    #: the rendezvous address for anything that talks host-to-host.
    address: str | None = None
    #: Direct SSH instead of the bastion: a name, IP, or ``~/.ssh/config`` alias.
    ssh_host: str | None = None
    ssh_user: str | None = None
    ssh_port: int | None = None
    #: Display hint only — the probe decides the real family.
    platform: str = "generic"
    tags: list[str] = field(default_factory=list)

    @property
    def is_direct(self) -> bool:
        return bool(self.ssh_host)

    def login_string(self, default_login: str | None) -> str:
        """The KoKo per-asset login: ``login#account#asset_id``.

        KoKo parses this as exactly three fields (``user#account#asset``) or four with
        a protocol (``user#protocol#account#asset``); two fields do not parse, so an
        account is required. The asset may be a UUID or a name — a name is allowed to
        match several assets, which then needs an extra prompt, so the UUID is what the
        inventory stores.
        """
        user = self.login or default_login
        if not user:
            raise InventoryError(
                f"host {self.name!r} has no 'login' and no bastion default login is set "
                "(add `login = ...` under [bastion] in config.toml)"
            )
        if self.asset_id is None:
            raise InventoryError(f"host {self.name!r} has no 'asset_id'")
        if not self.account:
            raise InventoryError(
                f"host {self.name!r} has no 'account' — KoKo needs user#account#asset, "
                "and two fields do not parse"
            )
        return f"{user}#{self.account}#{self.asset_id}"


@dataclass
class Inventory:
    hosts: dict[str, Host]
    groups: dict[str, list[str]] = field(default_factory=dict)
    #: Where it was loaded from, for messages.
    path: Path | None = None

    # --- selection --------------------------------------------------------------

    def resolve(self, selectors: list[str]) -> list[Host]:
        """Expand names / groups / tags into hosts, preserving inventory order.

        ``all`` (or ``*``) means everything. An unknown selector is an error rather than
        an empty list: a typo here would otherwise show as a blank screen, which reads
        like "no machines are free" — the most expensive wrong answer this tool can give.
        """
        if not selectors:
            return list(self.hosts.values())
        picked: dict[str, Host] = {}
        for selector in selectors:
            if selector in ("all", "*"):
                picked.update(self.hosts)
            elif selector in self.groups:
                for name in self.groups[selector]:
                    if name not in self.hosts:
                        raise InventoryError(
                            f"group {selector!r} lists unknown host {name!r} "
                            f"(known hosts: {sorted(self.hosts)})"
                        )
                    picked[name] = self.hosts[name]
            elif selector in self.hosts:
                picked[selector] = self.hosts[selector]
            else:
                tagged = [h for h in self.hosts.values() if selector in h.tags]
                if not tagged:
                    raise InventoryError(
                        f"selector {selector!r} matched no host, group, or tag "
                        f"(known hosts: {sorted(self.hosts)})"
                    )
                for host in tagged:
                    picked[host.name] = host
        # Inventory order, not selection order: two runs with the same selectors must
        # produce the same rows in the same places.
        return [h for name, h in self.hosts.items() if name in picked]

    def group_of(self, name: str) -> str | None:
        for group, members in self.groups.items():
            if name in members:
                return group
        return None

    # --- loading ----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> Inventory:
        target = Path(path) if path is not None else inventory_path()
        target = target.expanduser()
        if not target.exists():
            raise InventoryError(
                f"no inventory at {target}.\n"
                "Run `jms-top init` to write a starter one, or `jms-top hosts --discover` "
                "to build it from the assets your account can reach."
            )
        try:
            data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            # A hand-edited YAML file with a bad indent is the likeliest failure here,
            # and yaml's own error names the line — pass it through rather than
            # replacing it with "invalid inventory".
            raise InventoryError(f"{target} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise InventoryError(
                f"{target} must be a mapping with a 'hosts:' key, got {type(data).__name__}"
            )
        raw = data.get("hosts") or {}
        if not isinstance(raw, dict) or not raw:
            raise InventoryError(f"{target} has no 'hosts:' section")
        hosts: dict[str, Host] = {}
        for name, spec in raw.items():
            hosts[name] = _host_from(name, spec or {}, target)
        groups = {
            str(k): [str(n) for n in (v or [])] for k, v in (data.get("groups") or {}).items()
        }
        return cls(hosts=hosts, groups=groups, path=target)


def _host_from(name: str, spec: dict, path: Path) -> Host:
    if not isinstance(spec, dict):
        raise InventoryError(f"{path}: host {name!r} must be a mapping, got {type(spec).__name__}")
    asset = spec.get("asset_id")
    return Host(
        name=str(name),
        asset_id=str(asset) if asset is not None else None,
        account=spec.get("account"),
        login=spec.get("login"),
        address=spec.get("address"),
        ssh_host=spec.get("ssh_host"),
        ssh_user=spec.get("ssh_user"),
        ssh_port=int(spec["ssh_port"]) if spec.get("ssh_port") else None,
        platform=str(spec.get("platform", "generic")),
        tags=[str(t) for t in (spec.get("tags") or [])],
    )


def draft_inventory(assets: list[dict]) -> str:
    """Render discovered assets as a starter ``hosts.yaml``.

    Grouped by the tag vocabulary the fleet already uses (family first, then anything
    else the asset carries), so a generated file is usable without editing every entry.
    """
    lines = [
        "# Draft inventory, generated by `jms-top hosts --discover`.",
        "# Review the account on each entry: KoKo needs the OS user, not your login.",
        "",
        "hosts:",
    ]
    groups: dict[str, list[str]] = {}
    for asset in assets:
        name = _slug(asset["name"], taken=set(groups) | set(locals().get("_names", set())))
        _names = locals().setdefault("_names", set())
        _names.add(name)
        lines.append(f"  {name}:")
        lines.append(f"    asset_id: {asset['id']}")
        lines.append(f"    account: {asset.get('account') or 'root'}")
        if asset.get("address"):
            lines.append(f"    address: {asset['address']}")
        tags = list(asset.get("tags") or [])
        if asset.get("platform"):
            lines.append(f"    platform: {asset['platform']}")
        if tags:
            lines.append(f"    tags: [{', '.join(_yaml_scalar(t) for t in tags)}]")
        for tag in tags or ["all"]:
            groups.setdefault(tag, []).append(name)
    if groups:
        lines += ["", "groups:"]
        for tag, members in sorted(groups.items()):
            lines.append(f"  {tag}: [{', '.join(members)}]")
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: str) -> str:
    """Quote a YAML scalar only when it would otherwise be misread."""
    if value and all(c.isalnum() or c in "-_." for c in value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _slug(name: str, *, taken: set[str]) -> str:
    """A host key derived from an asset name, unique within the file."""
    base = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name.strip()).strip("-")
    base = base.lower() or "host"
    candidate, n = base, 2
    while candidate in taken:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def require_dashboard_url(settings) -> str:  # noqa: ANN001 - Settings, avoids a cycle
    """The bastion URL, or a config error naming where to set it."""
    if not settings.bastion.url:
        raise ConfigError(f"no bastion URL in {settings.config_file}")
    return settings.bastion.url


def write_inventory(assets, path: Path) -> None:  # noqa: ANN001 - discover.Asset
    """Write a complete ``hosts.yaml`` from discovered assets.

    Grouped by model and by family, so `jms-top a100` and `jms-top nvidia` both select
    something useful without hand-editing 77 entries. ``account`` is guessed as ``root``
    because that is what KoKo needs and what these images use; it is the one field worth
    reviewing by eye.
    """
    from .config import write_private

    lines = [
        "# Inventory generated by `jms-top hosts --discover`.",
        "#",
        "# One entry per asset this account may log in to, taken from the JumpServer",
        "# gateway itself. Re-run the command to refresh it; review `account` — KoKo",
        "# needs the OS user, not your login.",
        "",
        "hosts:",
    ]
    groups: dict[str, list[str]] = {}
    for asset in assets:
        lines.append(f"  {asset.key}:")
        # The asset id is what KoKo's login string carries; without it nothing on this
        # host can be dialled and every probe reports DOWN.
        if getattr(asset, "asset_id", None):
            lines.append(f"    asset_id: {asset.asset_id}")
        if asset.address:
            lines.append(f"    address: {asset.address}")
        if asset.platform:
            lines.append(f"    platform: {asset.platform}")
        lines.append(f"    account: {getattr(asset, 'account', None) or 'root'}")
        if asset.tags:
            lines.append("    tags: [" + ", ".join(_yaml_scalar(t) for t in asset.tags) + "]")
        for tag in _useful_tags(asset):
            groups.setdefault(tag, []).append(asset.key)
    if groups:
        lines += ["", "groups:"]
        for tag in sorted(groups):
            lines.append(f"  {tag}: [{', '.join(groups[tag])}]")
    write_private(path, "\n".join(lines) + "\n")


#: Name fragments that are noise in a selector: datacenter codes, zone numbers, sizes.
_NOISE_TAGS = {"bm", "vm", "p", "d", "lc", "gpu", "root", "unknown", "default", ""}


def _useful_tags(asset) -> list[str]:  # noqa: ANN001 - discover.Asset
    """The tags worth turning into a group.

    A group is only worth writing if it selects something narrower than "everything", so
    per-machine ordinals and hardware noise are dropped: ``zone1`` is a useful selector,
    ``42`` and ``96g`` are not. The family is always kept — it is the selector people
    actually use.
    """
    family = family_of_name(asset.name)
    keep = {family} if family and family != "unknown" else set()
    for tag in asset.tags:
        if tag in _NOISE_TAGS or tag.isdigit() or re.fullmatch(r"\d+g", tag):
            continue
        if re.fullmatch(r"zone\d+|dx|yz|jn|bj|sh|nj|wx", tag):
            keep.add(tag)
        elif re.fullmatch(
            r"(a100|a800|h100|h800|910c|910b|810e|tx8110|s5000|p800|mc550|bw1[01]00)", tag
        ):
            keep.add(tag)  # the model, which is more specific than the family
    return sorted(keep)


def family_of_name(name: str) -> str:
    from .discover import family_of

    return family_of(name)
