"""Logging in to JumpServer: REST token minting, and asset discovery for the inventory.

Two planes, two credentials, and it is worth being explicit about which is which:

**Control plane (REST, HTTPS).** ``POST /api/v1/authentication/auth/`` takes a username
and password and returns a session token. That token is what lets every later run be
password-free, so :func:`login` caches it in ``config.toml`` (mode 600). The token is a
*short-lived* credential — the server expires it — and ``jms-top logout`` drops it.

**Data plane (SSH through KoKo).** Writing ``hosts.yaml`` by hand means copying UUIDs
out of a web UI, which is the step people get wrong. :func:`discover_assets` reads the
API's own resource list with the cached token and produces a draft inventory: name,
id, address, and the accounts bound to each asset.

Nothing here is needed to run ``jms-top`` once the token is cached and the inventory
exists: it is the first-run path, not a per-poll dependency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .config import ConfigError, Settings, clear_token, save_password, store_token


class AuthError(RuntimeError):
    """Login failed, or the bastion refused a request."""


# --- control plane --------------------------------------------------------------


async def _request(
    settings: Settings,
    method: str,
    path: str,
    *,
    token: str | None = None,
    json_body: dict | None = None,
    params: dict | None = None,
):
    """One REST call, translating transport failures into :class:`AuthError`."""
    try:
        import httpx
    except ModuleNotFoundError as exc:  # pragma: no cover - httpx is a dependency
        raise ConfigError("httpx is required for login/discovery: pip install httpx") from exc

    bastion = settings.bastion
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(
            verify=bastion.verify_tls, timeout=bastion.api_timeout
        ) as client:
            return await client.request(
                method,
                bastion.api_url(path),
                headers=headers,
                json=json_body,
                params=params,
            )
    except Exception as exc:  # httpx.HTTPError and TLS/DNS failures alike
        raise AuthError(
            f"{method} {bastion.api_url(path)} failed: {type(exc).__name__}: {exc}"
        ) from exc


@dataclass
class LoginResult:
    username: str
    token: str
    config_file: str
    password_saved: str | None = None


async def login(
    settings: Settings,
    username: str,
    password: str,
    *,
    save: bool = False,
) -> LoginResult:
    """Exchange username/password for a session token and cache it.

    The token is *not* printed: it is a credential, and a terminal scrollback (or a log
    file) is a bad place for one. Only the fact that it worked, and where it was saved,
    are reported.
    """
    if not settings.bastion.url:
        raise ConfigError("no bastion URL configured; set [bastion] url in config.toml")
    response = await _request(
        settings,
        "POST",
        "authentication/auth/",
        json_body={"username": username, "password": password},
    )
    if response.status_code >= 400:
        raise AuthError(_explain(response))
    try:
        payload = response.json()
    except ValueError as exc:
        raise AuthError(f"login returned {response.text[:200]!r}, not JSON") from exc
    token = payload.get("token") if isinstance(payload, dict) else None
    if not token:
        raise AuthError(f"login succeeded but returned no token: {payload!r}")

    path = store_token(token)
    saved = None
    if save:
        saved = str(save_password(password))
    return LoginResult(username=username, token=token, config_file=str(path), password_saved=saved)


def _explain(response) -> str:  # noqa: ANN001 - httpx.Response, kept import-free
    """Turn an API error response into the sentence a user can act on."""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(payload.get("detail") or payload.get("error") or payload)
        else:
            detail = str(payload)
    except ValueError:
        detail = response.text[:300]
    hints = {
        401: " — check the username and password",
        403: " — the account may not be allowed to log in via the API",
        404: " — is [bastion] api_base in config.toml correct?",
    }
    return f"HTTP {response.status_code}: {detail}{hints.get(response.status_code, '')}"


def logout() -> dict:
    """Forget the cached session token (and optionally the saved password)."""
    return {"token_cleared": clear_token()}


# --- asset discovery ------------------------------------------------------------


@dataclass
class Asset:
    id: str
    name: str
    address: str | None = None
    platform: str | None = None
    #: OS accounts bound to the asset; the first is the default for an inventory entry.
    accounts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


async def _token(settings: Settings) -> str:
    token = settings.bastion.session_token
    if not token:
        raise AuthError(
            "not logged in — run `jms-top login` (assets can only be listed with a token)"
        )
    return token


async def discover_assets(settings: Settings, *, limit: int = 500) -> list[Asset]:
    """Every asset this account can reach, with the accounts bound to each.

    Two calls, not N+1: the resource list carries name, address and platform, and the
    bound accounts come from one account endpoint per asset — but that endpoint is
    fetched only for assets, and its failure is tolerated (an asset with no visible
    accounts is still worth listing; the user can fill ``account`` in by hand).
    """
    token = await _token(settings)
    found: list[Asset] = []
    offset = 0
    while len(found) < limit:
        response = await _request(
            settings,
            "GET",
            "perms/users/permissions/assets/",
            token=token,
            params={"limit": min(100, limit - len(found)), "offset": offset},
        )
        if response.status_code >= 400:
            raise AuthError(_explain(response))
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("results", [])
        if not rows:
            break
        renamed = [_asset_from(row) for row in rows]
        found.extend(a for a in renamed if a is not None)
        offset += len(rows)
        if len(rows) < 100:
            break
    await _fill_accounts(settings, token, found)
    return found


def _asset_from(row: dict) -> Asset | None:
    """One permission row -> :class:`Asset`, tolerating the field-name drift between
    JumpServer releases (``id``/``asset.id``, ``name``/``asset.name``)."""
    asset = row.get("asset") if isinstance(row.get("asset"), dict) else row
    asset_id = asset.get("id") or row.get("asset_id") or row.get("id")
    name = asset.get("name") or row.get("asset_name") or row.get("name")
    if not asset_id or not name:
        return None
    platform = asset.get("platform")
    if isinstance(platform, dict):
        platform = platform.get("name")
    tags = []
    for tag in asset.get("tags") or []:
        tags.append(tag if isinstance(tag, str) else str(tag.get("name", tag)))
    return Asset(
        id=str(asset_id),
        name=str(name),
        address=asset.get("address") or None,
        platform=str(platform) if platform else None,
        tags=tags,
    )


async def _fill_accounts(settings: Settings, token: str, assets: list[Asset]) -> None:
    """Attach bound accounts, tolerating a per-asset failure (some are 403 by design)."""

    async def one(asset: Asset) -> None:
        try:
            response = await _request(
                settings, "GET", f"perms/users/assets/{asset.id}/accounts/", token=token
            )
        except AuthError:
            return
        if response.status_code >= 400:
            return
        try:
            rows = response.json()
        except ValueError:
            return
        rows = rows if isinstance(rows, list) else rows.get("results", [])
        for row in rows:
            if isinstance(row, dict):
                username = row.get("username") or (row.get("alias") if row.get("alias") else None)
                if username:
                    asset.accounts.append(str(username))

    # Bounded fan-out: a fleet of 500 assets must not open 500 sockets.
    gate = asyncio.Semaphore(8)

    async def bounded(asset: Asset) -> None:
        async with gate:
            await one(asset)

    await asyncio.gather(*(bounded(a) for a in assets))
