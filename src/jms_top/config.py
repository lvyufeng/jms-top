"""This machine's own configuration: which bastion, which login, which hosts.

Everything jms-top reads lives under one directory:

``~/.config/jms-top/config.toml``
    The bastion URL, the KoKo SSH port, the default login, and the cached *session
    token* that ``jms-top login`` writes. Created by ``jms-top init`` if missing.
``~/.config/jms-top/hosts.yaml``
    The inventory: host name -> asset id, account, platform, tags.
``~/.config/jms-top/credentials.toml``
    Optional saved password, mode 600, written only when asked for.

Both files can be relocated with ``JMS_TOP_CONFIG`` / ``JMS_TOP_INVENTORY`` /
``JMS_TOP_CREDENTIALS``.

A note on why the token lives in ``config.toml`` and the password does not: the token
is what makes every later run password-free, and it expires server-side; the password
never expires and is the one secret worth keeping out of a file people copy around. So
the token is written (with mode 600 enforced on the whole directory, see
:func:`write_private`) and the password is only stored when ``login --save-password``
is asked for explicitly.

Nothing here imports anything heavier than the standard library, and nothing here is
required for ``--demo``: the demo path builds a config object in memory and never
touches ``~``.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib  # type: ignore

CONFIG_DIR_ENV = "JMS_TOP_CONFIG"
INVENTORY_ENV = "JMS_TOP_INVENTORY"
CREDENTIALS_ENV = "JMS_TOP_CREDENTIALS"

DEFAULT_CONFIG_DIR = "~/.config/jms-top"
DEFAULT_SSH_PORT = 2222
DEFAULT_API_BASE = "/api/v1"


class ConfigError(RuntimeError):
    """Raised when configuration is missing, unreadable, or unusable."""


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def config_dir() -> Path:
    """Where jms-top keeps its files. ``JMS_TOP_CONFIG`` may be a dir *or* a file."""
    override = _env(CONFIG_DIR_ENV)
    if override:
        path = Path(override).expanduser()
        # A path with a suffix is an explicit config file; use its parent for siblings.
        return path.parent if path.suffix == ".toml" else path
    return Path(DEFAULT_CONFIG_DIR).expanduser()


def config_path() -> Path:
    override = _env(CONFIG_DIR_ENV)
    if override and Path(override).expanduser().suffix == ".toml":
        return Path(override).expanduser()
    return config_dir() / "config.toml"


def inventory_path() -> Path:
    override = _env(INVENTORY_ENV)
    return Path(override).expanduser() if override else config_dir() / "hosts.yaml"


def credentials_path() -> Path:
    override = _env(CREDENTIALS_ENV)
    return Path(override).expanduser() if override else config_dir() / "credentials.toml"


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` readable by the owner only.

    ``mkdir(mode=0o700)`` is not enough on its own: ``mode`` is filtered through the
    umask, so on a 022 umask the directory silently ends up 755 while believing it is
    700. Both the mkdir and the file are therefore followed by an explicit ``chmod`` —
    a token file that is world-readable is the failure this guards.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:  # pragma: no cover - a shared dir we do not own
        pass
    # `open` with the mode and then chmod: an existing file keeps its old mode through
    # the open, and the token must not spend a moment at 644.
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


@dataclass
class Bastion:
    """Where the bastion is and how to log in to it.

    ``login`` is the *KoKo* login — the JumpServer user in ``login#account#asset`` — not
    a REST username; the REST side is authenticated by the cached session token.
    """

    url: str
    #: KoKo's SSH gateway port (JumpServer default 2222; some deployments use 2224).
    ssh_port: int = DEFAULT_SSH_PORT
    #: The ``login`` field of a ``login#account#asset`` login string — the JumpServer
    #: user, not the OS account on the far side.
    login: str | None = None
    #: Private key for the data plane. ``None`` -> ssh-agent and ``~/.ssh`` defaults.
    ssh_key: Path | None = None
    #: Set when KoKo only accepts password auth. Needs ``sshpass``.
    ssh_password: str | None = None
    #: Cache of ``jms-top login``. Sent as ``Bearer`` to the REST API.
    session_token: str | None = None
    #: Fan-out cap: bounded so a 20-host fleet does not trip KoKo's per-account limit.
    max_parallel: int = 8
    verify_tls: bool = True
    api_base: str = DEFAULT_API_BASE
    api_timeout: float = 30.0

    @property
    def host(self) -> str:
        """The bare ``host[:port]`` of the bastion, without scheme or path."""
        return re.sub(r"^[a-z]+://", "", self.url).split("/")[0]

    def api_url(self, path: str) -> str:
        return f"{self.url.rstrip('/')}/{self.api_base.strip('/')}/{path.lstrip('/')}"


@dataclass
class Settings:
    """Everything loaded from ``config.toml`` plus the credentials file."""

    bastion: Bastion
    #: Where each file was actually read from, so messages can point at it.
    config_file: Path
    inventory_file: Path = field(default_factory=inventory_path)


def load_settings(*, require_url: bool = True) -> Settings:
    """Read ``config.toml`` (+ saved credentials), with environment overrides.

    Environment beats file, file beats default — the usual order, and the one that lets
    a CI job run without a config file at all::

        JMS_TOP_URL, JMS_TOP_SSH_PORT, JMS_TOP_LOGIN, JMS_TOP_SSH_KEY,
        JMS_TOP_SSH_PASSWORD, JMS_TOP_SESSION_TOKEN, JMS_TOP_MAX_PARALLEL
    """
    path = config_path()
    data = _read_toml(path)
    section = data.get("bastion", {}) if isinstance(data, dict) else {}
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: [bastion] must be a table")

    url = _env("JMS_TOP_URL") or section.get("url")
    if not url:
        if require_url:
            raise ConfigError(
                f"no bastion URL configured.\n"
                f"Run `jms-top init` to write a starter {path}, then set bastion.url — "
                "or export JMS_TOP_URL."
            )
        url = ""

    password = _env("JMS_TOP_SSH_PASSWORD") or section.get("ssh_password")
    if not password:
        password = _saved_password()

    key = _env("JMS_TOP_SSH_KEY") or section.get("ssh_key")
    return Settings(
        bastion=Bastion(
            url=str(url).rstrip("/"),
            ssh_port=int(_env("JMS_TOP_SSH_PORT") or section.get("ssh_port", DEFAULT_SSH_PORT)),
            login=_env("JMS_TOP_LOGIN") or section.get("login"),
            ssh_key=Path(str(key)).expanduser() if key else None,
            ssh_password=password,
            session_token=_env("JMS_TOP_SESSION_TOKEN") or section.get("session_token"),
            max_parallel=_parallel(section.get("max_parallel", 8)),
            verify_tls=bool(section.get("verify_tls", True)),
            api_base=str(section.get("api_base", DEFAULT_API_BASE)),
            api_timeout=float(section.get("api_timeout", 30.0)),
        ),
        config_file=path,
    )


def _parallel(value: object) -> int:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"max_parallel must be an integer, got {value!r}") from exc
    if n < 1:
        # 0 would build a Semaphore no task can acquire — a silent hang.
        raise ConfigError(f"max_parallel must be at least 1, got {n}")
    return n


# --- saved password -------------------------------------------------------------


def _saved_password() -> str | None:
    data = _read_toml(credentials_path())
    section = data.get("credentials", {})
    value = section.get("password") if isinstance(section, dict) else None
    return str(value) if value else None


def save_password(password: str) -> Path:
    path = credentials_path()
    write_private(
        path,
        "# Written by `jms-top login --save-password`; mode 600.\n"
        "# Delete this file (or run `jms-top logout --forget-password`) to revoke it.\n"
        "[credentials]\n"
        f"password = {_toml_string(password)}\n",
    )
    return path


def forget_password() -> bool:
    path = credentials_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _toml_string(value: str) -> str:
    """A TOML basic string, which is JSON's escaping for every realistic password."""
    import json

    return json.dumps(value)


# --- token cache ----------------------------------------------------------------


_TOKEN_LINE = re.compile(r"(?m)^\s*session_token\s*=.*$")


def store_token(token: str) -> Path:
    """Cache a session token in ``config.toml``, preserving everything else in it.

    Rewriting only the one line (rather than re-serialising the TOML) keeps comments,
    ordering and any hand-added fields intact — a config file people edit by hand is
    the one file a tool must not silently reformat.
    """
    path = config_path()
    line = f"session_token = {_toml_string(token)}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if _TOKEN_LINE.search(text):
            # Callable replacement: a token containing a backslash would otherwise be
            # read as an escape sequence by `re.sub`.
            text = _TOKEN_LINE.sub(lambda _m: line, text)
        elif "[bastion]" in text:
            text = text.replace("[bastion]", f"[bastion]\n{line}", 1)
        else:
            text = text.rstrip("\n") + f"\n\n[bastion]\n{line}\n"
    else:
        text = f"[bastion]\n{line}\n"
    write_private(path, text)
    return path


def clear_token() -> bool:
    path = config_path()
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if not _TOKEN_LINE.search(text):
        return False
    write_private(path, _TOKEN_LINE.sub("", text))
    return True


# --- first run ------------------------------------------------------------------

STARTER_CONFIG = """\
# jms-top configuration. Written by `jms-top init`; edit freely.
#
# Everything here can also come from the environment (JMS_TOP_URL,
# JMS_TOP_SSH_PORT, JMS_TOP_LOGIN, JMS_TOP_SSH_KEY, ...), which is how to run
# without this file at all.

[bastion]
# The JumpServer web address — the KoKo SSH gateway is assumed to be the same host.
url = "https://jumpserver.example.com"
# KoKo's SSH port. JumpServer ships 2222; some deployments expose 2224.
ssh_port = 2222
# Your JumpServer *login* name (the user part of `login#account#asset`).
login = "your-login"
# Optional: a specific private key for the data plane (default: ssh-agent / ~/.ssh).
# ssh_key = "~/.ssh/id_ed25519"
# Only if this deployment insists on password auth (needs `sshpass` installed):
# ssh_password = "..."
"""

STARTER_INVENTORY = """\
# jms-top inventory. Written by `jms-top init`; edit freely.
#
# One entry per machine you want in the view. `asset_id` is the JumpServer UUID of the
# asset (资产列表 -> copy the id), and `account` is the OS user to log in as.
#
# Too many to type? `jms-top hosts --discover` lists every asset you can reach and
# writes a draft of this file for you.

hosts:
  example-gpu-1:
    asset_id: 00000000-0000-0000-0000-000000000000
    account: root
    platform: cuda
    tags: [gpu]

groups:
  gpu: [example-gpu-1]
"""


def write_starter_files(*, force: bool = False) -> list[Path]:
    """Create ``config.toml`` and ``hosts.yaml`` if they are missing."""
    written = []
    for path, text in ((config_path(), STARTER_CONFIG), (inventory_path(), STARTER_INVENTORY)):
        if path.exists() and not force:
            continue
        write_private(path, text)
        written.append(path)
    return written


def is_private(path: Path) -> bool:
    """Whether ``path`` is readable by its owner only (used by ``logout`` reporting)."""
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    return not (mode & (stat.S_IRWXG | stat.S_IRWXO))
