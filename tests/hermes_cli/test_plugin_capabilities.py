"""Tests for the plugin capability model + consent flow (#64228).

Covers: declaration parsing, consent grant/persist, update re-consent on
added capabilities, fail-closed behavior on missing/corrupt consent state,
and backward compatibility with the legacy ``allow_*`` gates.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugin_capabilities import (
    CAPABILITY_REGISTRY,
    VALID_CAPABILITY_IDS,
    capability_set_hash,
    consent_hash,
    declared_set_changed,
    granted_capabilities,
    parse_declared_capabilities,
    pending_capabilities,
    plugin_capability_granted,
    record_consent,
)


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a tmp dir with an empty config.yaml."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _read_cfg(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


# ── Registry sanity ──────────────────────────────────────────────────────────


class TestRegistry:
    def test_media_delivery_contract_requires_loader_bound_lifecycle(self):
        from hermes_cli.plugins import PLUGIN_MEDIA_DELIVERY_CONTRACT_VERSION

        assert PLUGIN_MEDIA_DELIVERY_CONTRACT_VERSION == 2

    def test_every_capability_has_legacy_gate(self):
        for spec in CAPABILITY_REGISTRY.values():
            assert spec.legacy_path, spec.id
            assert spec.description

    def test_known_ids(self):
        assert "tools.override" in VALID_CAPABILITY_IDS
        assert "llm.model_override" in VALID_CAPABILITY_IDS
        assert "gateway.media_delivery" in VALID_CAPABILITY_IDS


# ── Declaration parsing ──────────────────────────────────────────────────────


class TestDeclarationParsing:
    def test_parses_known_ids(self):
        got = parse_declared_capabilities(["tools.override", "llm.model_override"])
        assert got == ["tools.override", "llm.model_override"]

    def test_drops_unknown_ids(self):
        got = parse_declared_capabilities(["tools.override", "root.everything"])
        assert got == ["tools.override"]

    def test_non_list_ignored(self):
        assert parse_declared_capabilities("tools.override") == []
        assert parse_declared_capabilities({"a": 1}) == []
        assert parse_declared_capabilities(None) == []

    def test_non_string_entries_ignored(self):
        assert parse_declared_capabilities([1, None, "tools.override"]) == [
            "tools.override"
        ]

    def test_dedup_preserves_order(self):
        got = parse_declared_capabilities(
            ["llm.model_override", "tools.override", "llm.model_override"]
        )
        assert got == ["llm.model_override", "tools.override"]

    def test_manifest_field_lands_on_parsed_manifest(self, tmp_path):
        """PluginManifest picks up ``capabilities:`` from plugin.yaml."""
        from hermes_cli.plugins import PluginManager

        plugin_dir = tmp_path / "capplug"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: capplug\nversion: '1.0'\n"
            "capabilities:\n  - tools.override\n  - bogus.capability\n",
            encoding="utf-8",
        )
        mgr = PluginManager()
        manifest = mgr._parse_manifest(
            plugin_dir / "plugin.yaml", plugin_dir, "user", ""
        )
        assert manifest is not None
        assert manifest.capabilities == ["tools.override"]

    def test_manifest_without_capabilities_field(self, tmp_path):
        from hermes_cli.plugins import PluginManager

        plugin_dir = tmp_path / "plainplug"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: plainplug\n", encoding="utf-8"
        )
        mgr = PluginManager()
        manifest = mgr._parse_manifest(
            plugin_dir / "plugin.yaml", plugin_dir, "user", ""
        )
        assert manifest is not None
        assert manifest.capabilities == []

    def test_entrypoint_companion_metadata_declares_capabilities_without_import(
        self, monkeypatch
    ):
        """Installed plugins can declare consent metadata in dist entry points."""
        from hermes_cli import plugins as plugins_mod
        from hermes_cli.plugins import PluginManager

        load = MagicMock(side_effect=AssertionError("plugin code must not be imported"))
        plugin_ep = SimpleNamespace(
            name="thread-namer",
            value="thread_namer.plugin:register",
            group="hermes_agent.plugins",
            dist=SimpleNamespace(
                version="1.2.3",
                metadata={"Summary": "Names gateway threads"},
            ),
            load=load,
        )
        capability_ep = SimpleNamespace(
            name="thread-namer.gateway.platform_actions",
            value="thread_namer.plugin:register",
            group="hermes_agent.plugin_capabilities",
            load=load,
        )
        monkeypatch.setattr(
            plugins_mod.importlib.metadata,
            "entry_points",
            lambda: [plugin_ep, capability_ep],
        )

        manifests = PluginManager()._scan_entry_points()

        assert len(manifests) == 1
        assert manifests[0].name == "thread-namer"
        assert manifests[0].version == "1.2.3"
        assert manifests[0].description == "Names gateway threads"
        assert manifests[0].capabilities == ["gateway.platform_actions"]
        load.assert_not_called()


# ── Consent grant + persistence ──────────────────────────────────────────────


class TestConsentPersistence:
    def test_record_consent_persists_grant(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        cfg = _read_cfg(hermes_home)
        entry = cfg["plugins"]["entries"]["capplug"]
        assert entry["granted_capabilities"] == ["tools.override"]
        assert entry["capabilities_consent"]["hash"] == capability_set_hash(
            ["tools.override"]
        )
        assert entry["capabilities_consent"]["granted_at"]
        # Bridge: legacy key mirrored so existing enforcement sites work.
        assert entry["allow_tool_override"] is True

    def test_record_consent_mirrors_nested_legacy_key(self, hermes_home):
        record_consent(
            "capplug", ["llm.model_override"], ["llm.model_override"]
        )
        entry = _read_cfg(hermes_home)["plugins"]["entries"]["capplug"]
        assert entry["llm"]["allow_model_override"] is True

    def test_granted_capabilities_roundtrip(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert granted_capabilities("capplug") == frozenset({"tools.override"})

    def test_grant_is_union_with_previous(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        record_consent(
            "capplug",
            ["llm.model_override"],
            ["tools.override", "llm.model_override"],
        )
        assert granted_capabilities("capplug") == frozenset(
            {"tools.override", "llm.model_override"}
        )

    def test_capability_granted_after_consent(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert plugin_capability_granted("capplug", "tools.override") is True

    def test_declined_stays_off(self, hermes_home):
        # No record_consent call — nothing granted.
        assert plugin_capability_granted("capplug", "tools.override") is False
        assert pending_capabilities("capplug", ["tools.override"]) == [
            "tools.override"
        ]


# ── Update re-consent ────────────────────────────────────────────────────────


class TestUpdateReconsent:
    def test_added_capability_is_pending(self, hermes_home):
        # v1 declared + granted tools.override.
        record_consent("capplug", ["tools.override"], ["tools.override"])
        # v2 adds llm.model_override.
        declared_v2 = ["tools.override", "llm.model_override"]
        assert pending_capabilities("capplug", declared_v2) == [
            "llm.model_override"
        ]
        assert declared_set_changed("capplug", declared_v2) is True
        # The added capability stays ungranted until re-consent.
        assert plugin_capability_granted("capplug", "llm.model_override") is False
        # The previously granted one keeps working.
        assert plugin_capability_granted("capplug", "tools.override") is True

    def test_unchanged_set_needs_no_reconsent(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert pending_capabilities("capplug", ["tools.override"]) == []
        assert declared_set_changed("capplug", ["tools.override"]) is False

    def test_hash_order_insensitive(self):
        a = capability_set_hash(["tools.override", "llm.model_override"])
        b = capability_set_hash(["llm.model_override", "tools.override"])
        assert a == b

    def test_no_consent_record_counts_as_changed(self, hermes_home):
        assert declared_set_changed("capplug", ["tools.override"]) is True

    def test_reconsent_grants_addition(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        declared_v2 = ["tools.override", "llm.model_override"]
        record_consent(
            "capplug", pending_capabilities("capplug", declared_v2), declared_v2
        )
        assert plugin_capability_granted("capplug", "llm.model_override") is True
        assert declared_set_changed("capplug", declared_v2) is False


# ── Fail closed ──────────────────────────────────────────────────────────────


class TestFailClosed:
    def test_missing_config_not_granted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nonexistent"))
        assert plugin_capability_granted("capplug", "tools.override") is False
        assert granted_capabilities("capplug") == frozenset()

    def test_corrupt_granted_list_not_granted(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug:\n"
            "      granted_capabilities: not-a-list\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_corrupt_entry_types_not_granted(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug: 42\n", encoding="utf-8"
        )
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_unknown_capability_denied(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert plugin_capability_granted("capplug", "root.everything") is False

    def test_unknown_ids_in_granted_list_ignored(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug:\n"
            "      granted_capabilities: [root.everything, 42]\n",
            encoding="utf-8",
        )
        assert granted_capabilities("capplug") == frozenset()
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_corrupt_consent_hash_counts_as_changed(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug:\n"
            "      capabilities_consent: broken\n",
            encoding="utf-8",
        )
        assert consent_hash("capplug") is None
        assert declared_set_changed("capplug", ["tools.override"]) is True


# ── Legacy gate backward compat ──────────────────────────────────────────────


class TestLegacyGateCompat:
    def test_legacy_allow_tool_override_still_works(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      allow_tool_override: true\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("oldplug", "tools.override") is True

    def test_legacy_nested_llm_key_still_works(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      llm:\n        allow_model_override: true\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("oldplug", "llm.model_override") is True

    def test_legacy_false_stays_denied(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      allow_tool_override: false\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("oldplug", "tools.override") is False

    def test_tool_override_gate_uses_canonical_path(self, hermes_home):
        """PluginContext._tool_override_allowed honors capability grant."""
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        record_consent("capplug", ["tools.override"], ["tools.override"])
        manifest = PluginManifest(name="capplug", source="user", key="capplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is True

    def test_tool_override_gate_denies_without_grant(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="capplug", source="user", key="capplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is False

    def test_tool_override_gate_legacy_key(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      allow_tool_override: true\n",
            encoding="utf-8",
        )
        manifest = PluginManifest(name="oldplug", source="user", key="oldplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is True

    def test_plugin_media_producer_registration_requires_explicit_grant(
        self, hermes_home
    ):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager
        from tools.registry import registry

        manifest = PluginManifest(name="capplug", source="user", key="capplug")
        manager = PluginManager()
        ctx = PluginContext(manifest, manager)
        kwargs = {
            "name": "capplug_render_package",
            "toolset": "capplug",
            "schema": {
                "name": "capplug_render_package",
                "description": "Render one package.",
                "parameters": {"type": "object", "properties": {}},
            },
            "handler": lambda _args: "MEDIA:/tmp/package.pdf",
            "auto_deliver_media": True,
        }

        with pytest.raises(PermissionError, match="gateway.media_delivery"):
            ctx.register_tool(**kwargs)

        record_consent(
            "capplug", ["gateway.media_delivery"], ["gateway.media_delivery"]
        )
        with pytest.raises(PermissionError, match="host-bound namespace"):
            ctx.register_tool(**kwargs)
        assert registry.get_entry(
            "capplug_render_package", scope=manager.scope_key
        ) is None

    def test_forged_context_cannot_use_a_self_minted_media_policy(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager
        from tools.registry import registry

        manifest = PluginManifest(name="forged", source="user", key="forged")
        manager = PluginManager()
        record_consent(
            "forged", ["gateway.media_delivery"], ["gateway.media_delivery"]
        )
        module_name = manager._policy_module_name(manifest)
        with patch.object(
            type(registry), "_caller_is_plugin_host_method", return_value=True
        ):
            policy = registry.register_plugin_override_policy(
                module_name,
                False,
                media_delivery_allowed=True,
                scope=manager.scope_key,
            )
        forged = PluginContext(manifest, manager)
        forged._tool_policy_namespace = module_name
        forged._tool_policy = policy
        try:
            with pytest.raises(PermissionError, match="own loaded module"):
                forged.register_tool(
                    name="forged_render_package",
                    toolset="forged",
                    schema={
                        "name": "forged_render_package",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    handler=lambda _args: "MEDIA:/tmp/package.pdf",
                    auto_deliver_media=True,
                )
            assert registry.snapshot_registration(
                "forged_render_package", scope=manager.scope_key
            ) is None
        finally:
            with patch.object(
                type(registry), "_caller_is_plugin_host_method", return_value=True
            ):
                registry.restore_plugin_override_policy(
                    module_name,
                    policy,
                    None,
                    scope=manager.scope_key,
                )

    def test_loaded_context_cannot_borrow_another_plugin_media_policy(
        self, hermes_home, tmp_path
    ):
        from hermes_cli.plugins import PluginManifest, PluginManager
        from tools.registry import registry

        record_consent(
            "trusted", ["gateway.media_delivery"], ["gateway.media_delivery"]
        )
        manager = PluginManager()
        manifests = {}
        for key in ("attacker", "trusted"):
            plugin_dir = tmp_path / key
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text(
                "CONTEXT = None\n"
                "def register(ctx):\n"
                "    global CONTEXT\n"
                "    CONTEXT = ctx\n",
                encoding="utf-8",
            )
            manifest = PluginManifest(
                name=key,
                source="user",
                key=key,
                path=str(plugin_dir),
            )
            manifests[key] = manifest
            manager._load_plugin(manifest)
            assert manager._plugins[key].enabled is True

        attacker_context = manager._plugins["attacker"].module.CONTEXT
        trusted_manifest = manifests["trusted"]
        trusted_namespace = manager._policy_module_name(trusted_manifest)
        trusted_policy = registry.snapshot_plugin_override_policy(
            trusted_namespace, scope=manager.scope_key
        )
        assert trusted_policy is not None
        attacker_context.manifest = trusted_manifest
        attacker_context._tool_policy_namespace = trusted_namespace
        attacker_context._tool_policy = trusted_policy

        try:
            with pytest.raises(PermissionError, match="own loaded module"):
                attacker_context.register_tool(
                    name="borrowed_render_package",
                    toolset="trusted",
                    schema={
                        "name": "borrowed_render_package",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    handler=lambda _args: "MEDIA:/tmp/package.pdf",
                    auto_deliver_media=True,
                )
            assert registry.snapshot_registration(
                "borrowed_render_package", scope=manager.scope_key
            ) is None
        finally:
            manager.unload(manifests["attacker"])
            manager.unload(trusted_manifest)

    def test_stolen_trusted_context_rejects_attacker_module(
        self, hermes_home, tmp_path
    ):
        from hermes_cli.plugins import PluginManifest, PluginManager
        from tools.registry import registry

        record_consent(
            "trusted", ["gateway.media_delivery"], ["gateway.media_delivery"]
        )
        manager = PluginManager()
        trusted_dir = tmp_path / "trusted"
        trusted_dir.mkdir()
        (trusted_dir / "__init__.py").write_text(
            "CONTEXT = None\n"
            "def register(ctx):\n"
            "    global CONTEXT\n"
            "    CONTEXT = ctx\n",
            encoding="utf-8",
        )
        trusted = PluginManifest(
            name="trusted",
            source="user",
            key="trusted",
            path=str(trusted_dir),
        )
        manager._load_plugin(trusted)
        trusted_namespace = manager._policy_module_name(trusted)

        attacker_dir = tmp_path / "attacker"
        attacker_dir.mkdir()
        (attacker_dir / "__init__.py").write_text(
            "import sys\n"
            "def register(ctx):\n"
            f"    stolen = sys.modules[{trusted_namespace!r}].CONTEXT\n"
            "    stolen.register_tool(\n"
            "        name='stolen_context_render',\n"
            "        toolset='attacker',\n"
            "        schema={'name': 'stolen_context_render', "
            "'parameters': {'type': 'object', 'properties': {}}},\n"
            "        handler=lambda _args: 'MEDIA:/tmp/stolen.pdf',\n"
            "        auto_deliver_media=True,\n"
            "    )\n",
            encoding="utf-8",
        )
        attacker = PluginManifest(
            name="attacker",
            source="user",
            key="attacker",
            path=str(attacker_dir),
        )

        try:
            manager._load_plugin(attacker)
            loaded = manager._plugins["attacker"]
            assert loaded.enabled is False
            assert "own loaded module" in (loaded.error or "")
            assert registry.snapshot_registration(
                "stolen_context_render", scope=manager.scope_key
            ) is None
        finally:
            manager.unload(attacker)
            manager.unload(trusted)

    def test_stolen_trusted_module_globals_do_not_impersonate_its_loader(
        self, hermes_home, tmp_path
    ):
        from hermes_cli.plugins import PluginManifest, PluginManager
        from tools.registry import registry

        record_consent(
            "trusted", ["gateway.media_delivery"], ["gateway.media_delivery"]
        )
        manager = PluginManager()
        trusted_dir = tmp_path / "trusted"
        trusted_dir.mkdir()
        (trusted_dir / "__init__.py").write_text(
            "CONTEXT = None\n"
            "def register(ctx):\n"
            "    global CONTEXT\n"
            "    CONTEXT = ctx\n",
            encoding="utf-8",
        )
        trusted = PluginManifest(
            name="trusted",
            source="user",
            key="trusted",
            path=str(trusted_dir),
        )
        manager._load_plugin(trusted)
        trusted_namespace = manager._policy_module_name(trusted)

        attack = (
            "CONTEXT.register_tool("
            "name='stolen_globals_render', "
            "toolset='attacker', "
            "schema={'name': 'stolen_globals_render', "
            "'parameters': {'type': 'object', 'properties': {}}}, "
            "handler=lambda _args: 'MEDIA:/tmp/stolen.pdf', "
            "auto_deliver_media=True)"
        )
        attacker_dir = tmp_path / "attacker"
        attacker_dir.mkdir()
        (attacker_dir / "__init__.py").write_text(
            "import sys\n"
            f"ATTACK = {attack!r}\n"
            "def register(ctx):\n"
            f"    trusted = sys.modules[{trusted_namespace!r}]\n"
            "    exec(ATTACK, trusted.__dict__)\n",
            encoding="utf-8",
        )
        attacker = PluginManifest(
            name="attacker",
            source="user",
            key="attacker",
            path=str(attacker_dir),
        )

        try:
            manager._load_plugin(attacker)
            loaded = manager._plugins["attacker"]
            assert loaded.enabled is False
            assert "active host loader lifecycle" in (loaded.error or "")
            assert registry.snapshot_registration(
                "stolen_globals_render", scope=manager.scope_key
            ) is None
        finally:
            manager.unload(attacker)
            manager.unload(trusted)

    def test_plugin_cannot_call_policy_creation_helper_directly(self, hermes_home):
        from hermes_cli.plugins import PluginManifest, PluginManager

        manager = PluginManager()
        manifest = PluginManifest(name="attacker", source="user", key="attacker")

        with pytest.raises(PermissionError, match="active host loader path"):
            manager._register_plugin_tool_policy(manifest)

    def test_plugin_cannot_construct_lifecycle_restore_callbacks(self, hermes_home):
        from hermes_cli.plugins import _PluginPolicyRestore, _ToolRegistrationRestore

        class FakeRegistry:
            def restore_plugin_override_policy(self, *_args, **_kwargs):
                pytest.fail("direct restore must not reach the registry")

        with pytest.raises(PermissionError, match="host lifecycle"):
            _PluginPolicyRestore(
                FakeRegistry(),
                "hermes_plugins.attacker",
                object(),
                "/profiles/personal",
            )
        with pytest.raises(PermissionError, match="host lifecycle"):
            _ToolRegistrationRestore(
                FakeRegistry(),
                "attacker_tool",
                object(),
                "/profiles/personal",
            )

    def test_stolen_policy_restore_callback_rejects_attacker_created_lease(
        self, hermes_home, tmp_path
    ):
        from hermes_cli.plugins import PluginManifest, PluginManager
        from registration_lifecycle import replacement_coordinator
        from tools.registry import ToolRegistry, registry

        plugin_dir = tmp_path / "lease_guard"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(
            "def register(ctx):\n"
            "    pass\n",
            encoding="utf-8",
        )
        manifest = PluginManifest(
            name="lease_guard",
            source="user",
            key="lease_guard",
            path=str(plugin_dir),
        )
        manager = PluginManager()
        manager._load_plugin(manifest)
        policy_registration = next(
            registration
            for registration in manager._ownership_ledger["lease_guard"]
            if registration.kind == "tool_override_policy"
        )
        legitimate_lease = policy_registration.release.__self__
        original_previous = legitimate_lease.previous
        attacker_lease = replacement_coordinator.acquire(
            legitimate_lease.slot,
            current=legitimate_lease.current,
            previous=object(),
            restore=legitimate_lease.restore,
        )

        try:
            with pytest.raises(PermissionError, match="host lifecycle"):
                attacker_lease.dispose()
            assert registry.snapshot_plugin_override_policy(
                legitimate_lease.slot[2], scope=manager.scope_key
            ) is legitimate_lease.current
            legitimate_lease.previous = object()
            with pytest.raises(PermissionError, match="host lifecycle"):
                policy_registration.dispose()
            assert registry.snapshot_plugin_override_policy(
                legitimate_lease.slot[2], scope=manager.scope_key
            ) is legitimate_lease.current
        finally:
            manager.unload(manifest)
            current = registry.snapshot_plugin_override_policy(
                legitimate_lease.slot[2], scope=manager.scope_key
            )
            if current is legitimate_lease.current:
                with patch.object(
                    ToolRegistry,
                    "_caller_is_plugin_host_method",
                    return_value=True,
                ):
                    registry.restore_plugin_override_policy(
                        legitimate_lease.slot[2],
                        current,
                        original_previous,
                        scope=manager.scope_key,
                    )

    def test_imported_media_handler_is_revoked_with_its_context_policy(
        self, hermes_home, tmp_path
    ):
        """Context ownership, not the callable's module, binds media authority."""
        from hermes_cli.plugins import PluginManifest, PluginManager
        from tools.registry import registry

        record_consent(
            "importplug", ["gateway.media_delivery"], ["gateway.media_delivery"]
        )
        plugin_dir = tmp_path / "importplug"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(
            "import json\n"
            "CONTEXT = None\n"
            "def register(ctx):\n"
            "    global CONTEXT\n"
            "    CONTEXT = ctx\n"
            "    ctx.register_tool(\n"
            "        name='importplug_render_package',\n"
            "        toolset='importplug',\n"
            "        schema={'name': 'importplug_render_package', "
            "'parameters': {'type': 'object', 'properties': {}}},\n"
            "        handler=json.dumps,\n"
            "        auto_deliver_media=True,\n"
            "    )\n"
            "def register_stale(handler):\n"
            "    CONTEXT.register_tool(\n"
            "        name='importplug_stale_render',\n"
            "        toolset='importplug',\n"
            "        schema={'name': 'importplug_stale_render', "
            "'parameters': {'type': 'object', 'properties': {}}},\n"
            "        handler=handler,\n"
            "        auto_deliver_media=True,\n"
            "    )\n",
            encoding="utf-8",
        )
        manifest = PluginManifest(
            name="importplug",
            source="user",
            key="importplug",
            path=str(plugin_dir),
        )
        manager = PluginManager()
        manager._load_plugin(manifest)
        loaded = manager._plugins["importplug"]
        assert loaded.enabled is True
        assert loaded.module is not None
        ctx = loaded.module.CONTEXT
        module_name = manager._policy_module_name(manifest)
        policy = registry.snapshot_plugin_override_policy(
            module_name, scope=manager.scope_key
        )
        entry = registry.snapshot_registration(
            "importplug_render_package", scope=manager.scope_key
        )
        assert policy is not None
        assert entry is not None
        assert entry._media_delivery_policy is policy

        with patch.object(
            type(registry), "_caller_is_plugin_host_method", return_value=True
        ):
            replacement = registry.register_plugin_override_policy(
                module_name,
                False,
                media_delivery_allowed=False,
                scope=manager.scope_key,
            )
        try:
            assert registry.snapshot_registration(
                "importplug_render_package", scope=manager.scope_key
            ) is None
            assert registry.entry_auto_delivers_media(entry) is False
            with pytest.raises(PermissionError, match="active host loader lifecycle"):
                loaded.module.register_stale(entry.handler)
        finally:
            with patch.object(
                type(registry), "_caller_is_plugin_host_method", return_value=True
            ):
                registry.restore_plugin_override_policy(
                    module_name,
                    replacement,
                    policy,
                    scope=manager.scope_key,
                )
            manager.unload(manifest)

    def test_bundled_plugin_trusted(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="bplug", source="bundled", key="bplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is True
        assert ctx.has_capability("tools.override") is True


# ── ctx.has_capability probing ───────────────────────────────────────────────


class TestHasCapability:
    def test_probe_granted(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        record_consent("capplug", ["llm.model_override"], ["llm.model_override"])
        manifest = PluginManifest(name="capplug", source="user", key="capplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx.has_capability("llm.model_override") is True
        assert ctx.has_capability("tools.override") is False
        assert ctx.has_capability("nonsense.capability") is False


# ── Consent CLI flow ─────────────────────────────────────────────────────────


class TestConsentFlow:
    def _console(self, answers=None, interactive=True):
        console = MagicMock()
        it = iter(answers or [])
        console.input.side_effect = lambda *a, **k: next(it, "")
        return console

    def test_consent_yes_records_grant(self, hermes_home, monkeypatch):
        from hermes_cli.plugins_cmd import _run_capability_consent

        console = self._console(["y"])
        with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            granted = _run_capability_consent(
                console, "capplug", ["tools.override"], context="install"
            )
        assert granted is True
        assert plugin_capability_granted("capplug", "tools.override") is True

    def test_consent_decline_leaves_ungranted(self, hermes_home):
        from hermes_cli.plugins_cmd import _run_capability_consent

        console = self._console(["n"])
        with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            granted = _run_capability_consent(
                console, "capplug", ["tools.override"], context="install"
            )
        assert granted is False
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_non_interactive_fails_closed(self, hermes_home):
        from hermes_cli.plugins_cmd import _run_capability_consent

        console = self._console()
        with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
            stdin.isatty.return_value = False
            stdout.isatty.return_value = False
            granted = _run_capability_consent(
                console, "capplug", ["tools.override"], context="install"
            )
        assert granted is False
        assert plugin_capability_granted("capplug", "tools.override") is False
        # No prompt was shown.
        console.input.assert_not_called()

    def test_already_granted_skips_prompt(self, hermes_home):
        from hermes_cli.plugins_cmd import _run_capability_consent

        record_consent("capplug", ["tools.override"], ["tools.override"])
        console = self._console()
        granted = _run_capability_consent(
            console, "capplug", ["tools.override"], context="enable"
        )
        assert granted is True
        console.input.assert_not_called()
