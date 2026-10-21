"""Tests for the standalone stack: config, inventory, KoKo addressing, and login.

These cover the part that used to live in jms-bridge. The bar is the same as the rest
of the suite: no network, and every assertion about something a user would notice —
the login string KoKo actually receives, the file mode a token is written with, the
error text when an inventory is wrong.
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

import pytest

from jms_top import auth, config
from jms_top.inventory import Host, Inventory, InventoryError, draft_inventory
from jms_top.ssh import ExecResult, SSHRunner, failure_reason, reached


@pytest.fixture()
def cfgdir(tmp_path, monkeypatch):
    """Point every jms-top file at a temp dir, so tests never touch ~/.config."""
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path))
    for var in (config.INVENTORY_ENV, config.CREDENTIALS_ENV):
        monkeypatch.delenv(var, raising=False)
    for var in (
        "JMS_TOP_URL",
        "JMS_TOP_LOGIN",
        "JMS_TOP_SSH_KEY",
        "JMS_TOP_SSH_PASSWORD",
        "JMS_TOP_SESSION_TOKEN",
        "JMS_TOP_SSH_PORT",
        "JMS_TOP_MAX_PARALLEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# --- config --------------------------------------------------------------------


def test_init_writes_private_starter_files(cfgdir):
    written = config.write_starter_files()
    assert {p.name for p in written} == {"config.toml", "hosts.yaml"}
    for path in written:
        # The token file is the one that must never be group/world readable; the
        # template is written the same way so the habit never depends on which file.
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfgdir.stat().st_mode) == 0o700
    assert config.is_private(cfgdir / "config.toml")


def test_init_does_not_clobber_without_force(cfgdir):
    config.write_private(config.config_path(), "url = 'mine'\n")
    written = config.write_starter_files()
    # Only the file that did not exist is written; the edited one is left alone.
    assert [p.name for p in written] == ["hosts.yaml"]
    assert "mine" in config.config_path().read_text()
    assert config.write_starter_files() == []


def test_load_settings_reads_file_then_env(cfgdir, monkeypatch):
    config.write_private(
        config.config_path(),
        '[bastion]\nurl = "https://one.example"\nssh_port = 2224\nlogin = "me"\n',
    )
    settings = config.load_settings()
    assert settings.bastion.url == "https://one.example"
    assert settings.bastion.ssh_port == 2224
    assert settings.bastion.host == "one.example"

    monkeypatch.setenv("JMS_TOP_URL", "https://two.example/")
    monkeypatch.setenv("JMS_TOP_LOGIN", "someone-else")
    settings = config.load_settings()
    assert settings.bastion.url == "https://two.example"  # trailing slash trimmed
    assert settings.bastion.login == "someone-else"


def test_a_missing_url_says_what_to_run(cfgdir):
    with pytest.raises(config.ConfigError) as exc:
        config.load_settings()
    assert "jms-top init" in str(exc.value)


def test_store_token_keeps_the_rest_of_the_file(cfgdir):
    """Rewriting a hand-edited config must not reorder or drop anything."""
    original = '# my notes\n[bastion]\nurl = "https://x.example"\nlogin = "me"  # keep\n'
    config.write_private(config.config_path(), original)
    config.store_token("abc.def")
    text = config.config_path().read_text()
    assert "# my notes" in text and 'login = "me"  # keep' in text
    assert 'session_token = "abc.def"' in text
    # Replacing an existing token edits the line rather than appending a second one.
    config.store_token("second")
    text = config.config_path().read_text()
    assert text.count("session_token") == 1 and "second" in text
    assert stat.S_IMODE(config.config_path().stat().st_mode) == 0o600


def test_clear_token_removes_only_the_line(cfgdir):
    config.write_private(config.config_path(), '[bastion]\nurl = "u"\nsession_token = "t"\n')
    assert config.clear_token() is True
    assert "session_token" not in config.config_path().read_text()
    assert 'url = "u"' in config.config_path().read_text()
    assert config.clear_token() is False


def test_saved_password_round_trips_and_is_private(cfgdir):
    path = config.save_password('p@ss "quoted"')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert config._saved_password() == 'p@ss "quoted"'
    assert config.forget_password() is True
    assert config._saved_password() is None


def test_max_parallel_of_zero_is_rejected_not_clamped(cfgdir):
    """0 builds a Semaphore nothing can acquire — a hang, not a slow run."""
    config.write_private(config.config_path(), '[bastion]\nurl = "u"\nmax_parallel = 0\n')
    with pytest.raises(config.ConfigError):
        config.load_settings()


# --- inventory -----------------------------------------------------------------


def write_inventory(cfgdir: Path, text: str) -> Path:
    path = cfgdir / "hosts.yaml"
    config.write_private(path, text)
    return path


BASIC = """\
hosts:
  gpu-1:
    asset_id: aaaa-1111
    account: root
    platform: cuda
    tags: [gpu, cuda]
  npu-1:
    asset_id: bbbb-2222
    account: ubuntu
    tags: [npu]
  direct-1:
    ssh_host: 127.0.0.1
    ssh_user: me
groups:
  cluster: [gpu-1, npu-1]
"""


def test_inventory_parses_and_resolves(cfgdir):
    write_inventory(cfgdir, BASIC)
    inv = Inventory.load()
    assert list(inv.hosts) == ["gpu-1", "npu-1", "direct-1"]
    assert [h.name for h in inv.resolve(["npu"])] == ["npu-1"]
    assert [h.name for h in inv.resolve(["cluster"])] == ["gpu-1", "npu-1"]
    # Inventory order, not selector order: two runs must produce the same rows.
    assert [h.name for h in inv.resolve(["direct-1", "gpu-1"])] == ["gpu-1", "direct-1"]
    assert len(inv.resolve([])) == 3
    assert inv.group_of("npu-1") == "cluster"


def test_unknown_selector_is_an_error_not_an_empty_screen(cfgdir):
    write_inventory(cfgdir, BASIC)
    with pytest.raises(InventoryError) as exc:
        Inventory.load().resolve(["nope"])
    assert "known hosts" in str(exc.value)


def test_a_broken_group_names_the_bad_member(cfgdir):
    write_inventory(
        cfgdir, "hosts:\n  a:\n    asset_id: x\n    account: root\ngroups:\n  g: [a, ghost]\n"
    )
    with pytest.raises(InventoryError) as exc:
        Inventory.load().resolve(["g"])
    assert "ghost" in str(exc.value)


def test_bad_yaml_reports_the_line(cfgdir):
    write_inventory(cfgdir, "hosts:\n  a:\n   asset_id: x\n    account: root\n")
    with pytest.raises(InventoryError) as exc:
        Inventory.load()
    assert "not valid YAML" in str(exc.value)


def test_missing_inventory_suggests_discover(cfgdir):
    with pytest.raises(InventoryError) as exc:
        Inventory.load()
    assert "--discover" in str(exc.value)


def test_draft_inventory_groups_and_escapes():
    text = draft_inventory(
        [
            {"id": "i-1", "name": "Ascend 910C 108", "account": "root", "tags": ["npu", "site-a"]},
            {"id": "i-2", "name": "ascend-910c-108", "account": "root", "tags": ["npu"]},
        ]
    )
    # Two assets that slug to the same key must not collide.
    assert "ascend-910c-108:" in text and "ascend-910c-108-2:" in text
    assert "groups:" in text and "npu: [" in text


# --- KoKo addressing -----------------------------------------------------------


def make_runner(**kwargs) -> SSHRunner:
    from jms_top.config import Bastion

    return SSHRunner(
        Bastion(url="https://bastion.example.com", ssh_port=2224, login="me", **kwargs)
    )


def test_login_string_is_three_fields():
    host = Host(name="a", asset_id="uuid-1", account="root")
    assert host.login_string("me") == "me#root#uuid-1"


def test_login_string_needs_an_account():
    """KoKo parses user#account#asset; two fields do not parse, so say so locally."""
    host = Host(name="a", asset_id="uuid-1")
    with pytest.raises(InventoryError) as exc:
        host.login_string("me")
    assert "account" in str(exc.value)


def test_target_and_port_for_bastion_and_direct():
    runner = make_runner()
    bastion_host = Host(name="a", asset_id="uuid-1", account="root")
    assert runner.target(bastion_host) == "me#root#uuid-1@bastion.example.com"
    assert runner.port(bastion_host) == 2224
    direct = Host(name="b", ssh_host="10.0.0.5", ssh_user="me", ssh_port=22)
    assert runner.target(direct) == "10.0.0.5"
    assert runner.port(direct) == 22


def test_argv_multiplexes_and_batches():
    runner = make_runner()
    argv = runner.argv(Host(name="a", asset_id="uuid-1", account="root"))
    assert "ControlMaster=auto" in argv and "BatchMode=yes" in argv
    assert any(a.startswith("ControlPath=") for a in argv)
    assert argv[-2].startswith("me#root#uuid-1@") and argv[-1] == "bash -s"


def test_password_auth_drops_batch_mode(monkeypatch):
    """BatchMode=yes disables the password prompt sshpass exists to answer."""
    runner = make_runner(ssh_password="s3cret")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/sshpass")
    argv = runner.argv(Host(name="a", asset_id="u", account="root"))
    assert argv[:3] == ["sshpass", "-e", "ssh"]
    assert "BatchMode=yes" not in argv
    assert "PreferredAuthentications=password" in argv


def test_rejection_reads_as_down_not_up():
    """KoKo answers a rejection with exit 0; the exit status must not mean "up"."""
    rejected = ExecResult("a", 0, "没有资产或资产未激活", "")
    assert reached(rejected) is False
    assert "没有资产" in failure_reason(rejected)
    assert reached(ExecResult("a", 0, "jms-top-ok\n##CPU\n", "")) is True


def test_failure_reason_prefers_the_real_message():
    assert "denied" in failure_reason(ExecResult("a", 255, "", "Permission denied"))
    assert "timed out" in failure_reason(ExecResult("a", 124, "", "", timed_out=True))


# --- login ---------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (repr(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture()
def fake_httpx(monkeypatch):
    """Replace auth._request so login logic can be tested without a network."""
    calls = []

    async def fake_request(settings, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return fake_httpx.reply

    monkeypatch.setattr(auth, "_request", fake_request)
    fake_httpx.calls = calls
    fake_httpx.reply = FakeResponse(200, {"token": "tok-123"})
    return fake_httpx


def settings_for(cfgdir, url="https://bastion.example.com"):
    config.write_private(config.config_path(), f'[bastion]\nurl = "{url}"\nlogin = "me"\n')
    return config.load_settings()


def test_login_caches_the_token_and_not_the_password(cfgdir, fake_httpx):
    settings = settings_for(cfgdir)

    async def run():
        return await auth.login(settings, "me", "pw")

    result = asyncio.run(run())
    assert result.token == "tok-123"
    text = config.config_path().read_text()
    assert 'session_token = "tok-123"' in text
    assert "pw" not in text  # the password is never written unless asked
    assert config._saved_password() is None
    method, path, kwargs = fake_httpx.calls[0]
    assert (method, path) == ("POST", "authentication/auth/")
    assert kwargs["json_body"] == {"username": "me", "password": "pw"}


def test_login_can_save_the_password_on_request(cfgdir, fake_httpx):
    settings = settings_for(cfgdir)

    async def run():
        return await auth.login(settings, "me", "pw", save=True)

    asyncio.run(run())
    assert config._saved_password() == "pw"


def test_login_error_carries_the_status_and_a_hint(cfgdir, fake_httpx):
    fake_httpx.reply = FakeResponse(401, {"detail": "invalid user or password"})
    settings = settings_for(cfgdir)

    async def run():
        await auth.login(settings, "me", "bad")

    with pytest.raises(auth.AuthError) as exc:
        asyncio.run(run())
    assert "401" in str(exc.value) and "check the username" in str(exc.value)
    assert (
        not config.config_path().exists() or "session_token" not in config.config_path().read_text()
    )


def test_login_without_a_token_is_an_error(cfgdir, fake_httpx):
    fake_httpx.reply = FakeResponse(200, {"messages": "ok but no token"})
    settings = settings_for(cfgdir)

    async def run():
        await auth.login(settings, "me", "pw")

    with pytest.raises(auth.AuthError) as exc:
        asyncio.run(run())
    assert "no token" in str(exc.value)


def test_discover_needs_a_token(cfgdir):
    settings = settings_for(cfgdir)

    async def run():
        await auth.discover_assets(settings)

    with pytest.raises(auth.AuthError) as exc:
        asyncio.run(run())
    assert "jms-top login" in str(exc.value)


def test_discover_maps_permission_rows_to_assets(cfgdir, fake_httpx):
    settings = settings_for(cfgdir)
    settings.bastion.session_token = "tok"

    rows = [
        {"asset": {"id": "1", "name": "gpu-1", "address": "10.0.0.1", "platform": "cuda"}},
        {"id": "2", "name": "npu-1", "platform": {"name": "ascend"}},
        {"name": "no-id-and-no-asset"},
    ]

    async def fake_request(settings, method, path, **kwargs):
        if path.endswith("accounts/"):
            return FakeResponse(200, [{"username": "root"}])
        return FakeResponse(200, {"results": rows})

    import jms_top.auth as mod

    original = mod._request
    mod._request = fake_request
    try:

        async def run():
            return await auth.discover_assets(settings)

        assets = asyncio.run(run())
    finally:
        mod._request = original

    assert [a.name for a in assets] == ["gpu-1", "npu-1"]
    assert assets[0].platform == "cuda"
    assert assets[1].platform == "ascend"  # nested platform dict
    assert assets[0].accounts == ["root"]


# --- discovery ----------------------------------------------------------------


def test_koko_rows_parse_into_assets():
    """KoKo prints fixed-width columns separated by pipes; the pipes are the contract."""
    from jms_top.discover import announce_total, parse_page

    page = """
  ID | 名称                                            | 地址              | 平台
-----+-------------------------------------------------+-------------------+--------
  1  | bm-baai-dx-zone1-d-a100-40g-2-42                | 10.1.2.42         | Linux
  1  | 229-pm-aliyun-wlcb-zonea4-d-810e-96G（ID：）    | 8.130.132.221:229 | Linux
页码：1，每页行数：1，总页数：77，总数量：77
"""
    assets = parse_page(page)
    assert [a.address for a in assets] == ["10.1.2.42", "8.130.132.221:229"]
    # The ``（ID：）`` suffix is decoration; the raw spelling is kept for API lookups.
    assert assets[1].name == "229-pm-aliyun-wlcb-zonea4-d-810e-96G"
    assert "（ID：" in assets[1].raw_name
    assert announce_total(page) == 77


def test_asset_keys_are_short_and_unique():
    """77 names like `bm-baai-dx-zone1-d-a100-40g-2-42` need keys a human can type."""
    from jms_top.discover import Asset, assign_keys

    assets = [
        Asset(name="bm-baai-dx-zone1-d-a100-40g-2-42", address="10.1.2.42"),
        Asset(name="bm-baai-dx-zone1-d-a100-40g-2-8", address="10.1.2.8"),
        Asset(name="bm-jn-zs-zone1-910C-64G-10-108", address="10.55.0.174"),
        Asset(name="u101-u101", address="10.31.28.101"),
    ]
    assign_keys(assets)
    keys = [a.key for a in assets]
    assert keys == ["a100-42", "a100-8", "910c-108", "u101"]
    assert len(set(keys)) == len(keys)
    assert "a100" in assets[0].tags and "nvidia" in assets[0].tags


def test_duplicate_keys_get_suffixed_not_dropped():
    from jms_top.discover import Asset, assign_keys

    assets = [Asset(name="bm-x-a100-1", address="a"), Asset(name="bm-y-a100-1", address="b")]
    assign_keys(assets)
    assert assets[0].key != assets[1].key
    assert assets[1].key == "a100-1-2"


def test_asset_ids_are_matched_through_the_untidied_name():
    """KoKo hides the UUID and tidies the name; the API keeps both. Match on either."""
    from jms_top.discover import Asset, attach_asset_ids

    assets = [
        Asset(name="clean-name", address="a", raw_name="clean-name"),
        Asset(name="tidied", address="b", raw_name="tidied（ID：）"),
        Asset(name="orphan", address="c", raw_name="orphan"),
    ]
    matched = attach_asset_ids(assets, {"clean-name": "id-1", "tidied（ID：）": "id-2"})
    assert matched == 2
    assert [a.asset_id for a in assets] == ["id-1", "id-2", None]


def test_written_inventory_carries_asset_ids(tmp_path):
    """Without `asset_id` no login string can be built and every host reports DOWN."""
    from jms_top.discover import Asset
    from jms_top.inventory import Inventory, write_inventory

    assets = [Asset(name="bm-x-a100-1", address="10.0.0.1", asset_id="uuid-1", key="a100-1")]
    assets[0].tags = ["a100", "nvidia"]
    path = tmp_path / "hosts.yaml"
    write_inventory(assets, path)
    text = path.read_text()
    assert "asset_id: uuid-1" in text and "account: root" in text
    inv = Inventory.load(path)
    assert inv.hosts["a100-1"].login_string("me") == "me#root#uuid-1"
