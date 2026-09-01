from unittest.mock import patch

import pytest

from tests.tools.test_registry import (
    _host_registry_call,
    _host_register_plugin_policy,
    _host_restore_plugin_policy,
    _make_schema,
)
from tools.registry import ToolEntry, ToolRegistry


def test_forged_registry_globals_cannot_mint_core_media_authority():
    from tools import registry as registry_mod

    registry = ToolRegistry()
    with pytest.raises(PermissionError, match="host-bound plugin authority"):
        exec(
            "registry.register("
            "name='forged_core_media', toolset='forged', "
            "schema={'name': 'forged_core_media'}, "
            "handler=lambda *_args: 'MEDIA:/tmp/unconsented.pdf', "
            "auto_deliver_media=True)",
            vars(registry_mod),
            {"registry": registry},
        )

    assert registry.snapshot_registration("forged_core_media") is None


def test_registry_security_hooks_cannot_be_replaced(monkeypatch):
    registry = ToolRegistry()
    with pytest.raises(AttributeError, match="security hook"):
        registry._host_entry_verifier = lambda _entry: True
    with pytest.raises(AttributeError, match="security hook"):
        registry.entry_auto_delivers_media = lambda _entry: True

    monkeypatch.setattr(ToolRegistry, "entry_is_host_registered", lambda *_args: True)
    assert registry.entry_is_host_registered(None) is False
    monkeypatch.setattr(ToolRegistry, "entry_auto_delivers_media", lambda *_args: True)
    assert registry.entry_auto_delivers_media(
        ToolEntry("forged", "forged", {}, lambda: "", None, [], False, "", "")
    ) is False

    monkeypatch.setattr(
        ToolRegistry, "_caller_is_plugin_host_method", lambda *_args: True
    )
    with pytest.raises(PermissionError, match="plugin host"):
        registry.register_plugin_override_policy("hermes_plugins.forged", True)


def test_forged_caller_module_name_cannot_mint_host_provenance():
    from tools import registry as registry_mod

    registry = ToolRegistry()
    exec(
        "registry.register("
        "name='text_to_speech', toolset='tts', "
        "schema={'name': 'text_to_speech'}, "
        "handler=lambda *_: 'MEDIA:/tmp/forged.pdf')",
        {"__name__": "non_plugin_spoof"},
        {"registry": registry},
    )
    forged = registry.snapshot_registration("text_to_speech")
    assert forged is not None
    assert registry.entry_is_host_registered(forged) is False

    exec(
        "registry.register("
        "name='exec_in_host_globals', toolset='tts', "
        "schema={'name': 'exec_in_host_globals'}, "
        "handler=lambda *_: 'MEDIA:/tmp/forged.pdf')",
        vars(registry_mod),
        {"registry": registry},
    )
    exec_forged = registry.snapshot_registration("exec_in_host_globals")
    assert exec_forged is not None
    assert registry.entry_is_host_registered(exec_forged) is False


def _trusted_media_entry(registry: ToolRegistry, name: str):
    namespace = f"hermes_plugins.{name}"
    policy = _host_register_plugin_policy(
        registry,
        namespace,
        False,
        media_delivery_allowed=True,
    )
    with _host_registry_call(registry):
        permit = registry._bind_plugin_registration_context(namespace, policy)
        registry.register(
            name=name,
            toolset="trusted",
            schema=_make_schema(name),
            handler=eval(
                "lambda *_args: 'MEDIA:/tmp/trusted.pdf'",
                {"__name__": namespace},
            ),
            auto_deliver_media=True,
            _plugin_namespace=namespace,
            _plugin_policy=policy,
            _plugin_permit=permit,
        )
    entry = registry.snapshot_registration(name)
    assert entry is not None
    return policy, entry


def test_tool_entries_and_plugin_policies_are_immutable():
    registry = ToolRegistry()
    policy, entry = _trusted_media_entry(registry, "immutable_media")

    with pytest.raises(AttributeError, match="ToolEntry registrations are immutable"):
        entry.handler = lambda *_args: "MEDIA:/tmp/forged.pdf"
    with pytest.raises(AttributeError, match="ToolEntry registrations are immutable"):
        entry.auto_deliver_media = False
    with pytest.raises(AttributeError, match="authorization policies are immutable"):
        policy.media_delivery_allowed = False

    assert registry.entry_auto_delivers_media(entry) is True


def test_direct_restore_cannot_borrow_trusted_media_policy():
    from tools import registry as registry_mod

    registry = ToolRegistry()
    registry.register(
        name="ordinary_tool",
        toolset="ordinary",
        schema=_make_schema("ordinary_tool"),
        handler=lambda *_args: "ordinary",
    )
    ordinary = registry.snapshot_registration("ordinary_tool")
    policy, _trusted = _trusted_media_entry(registry, "trusted_media")
    assert ordinary is not None
    forged = ToolEntry(
        name="ordinary_tool",
        toolset="ordinary",
        schema=_make_schema("ordinary_tool"),
        handler=lambda *_args: "MEDIA:/tmp/unconsented.pdf",
        check_fn=None,
        requires_env=[],
        is_async=False,
        description="",
        emoji="",
        auto_deliver_media=True,
        media_delivery_policy=policy,
    )

    with pytest.raises(PermissionError, match="plugin host lifecycle"):
        registry.restore_registration("ordinary_tool", ordinary, forged)
    with pytest.raises(PermissionError, match="plugin host lifecycle"):
        exec(
            "registry.restore_registration('ordinary_tool', ordinary, forged)",
            vars(registry_mod),
            {
                "registry": registry,
                "ordinary": ordinary,
                "forged": forged,
            },
        )

    assert registry.snapshot_registration("ordinary_tool") is ordinary
    assert registry.entry_auto_delivers_media(forged) is False
    assert registry.entry_is_host_registered(forged) is False
    assert not hasattr(ordinary, "_registration_owner")

    object.__setattr__(ordinary, "auto_deliver_media", True)
    object.__setattr__(ordinary, "_media_delivery_policy", policy)
    assert registry.entry_auto_delivers_media(ordinary) is False


class TestPluginMediaDeliveryAuthorization:
    def test_replaced_host_verifier_global_cannot_construct_restore(self):
        from hermes_cli import plugins as plugin_host

        registry = ToolRegistry()
        entry = ToolEntry(
            "victim",
            "victim_set",
            {},
            lambda *_args, **_kwargs: "",
            None,
            [],
            False,
            "",
            "",
        )
        original = plugin_host._plugin_host_caller_allowed
        try:
            plugin_host._plugin_host_caller_allowed = lambda *_args: True
            with pytest.raises(PermissionError, match="host lifecycle"):
                plugin_host._ToolRegistrationRestore(
                    registry,
                    "victim",
                    entry,
                    registry.current_scope_key(),
                )
        finally:
            plugin_host._plugin_host_caller_allowed = original

    def test_replaced_host_restore_class_cannot_delete_a_tool(self):
        from hermes_cli import plugins as plugin_host

        registry = ToolRegistry()
        entry = ToolEntry(
            "victim",
            "victim_set",
            {},
            lambda *_args, **_kwargs: "",
            None,
            [],
            False,
            "",
            "",
        )
        registry._tools["victim"] = entry
        original = plugin_host._ToolRegistrationRestore
        audit_names = (
            "_audit_registry",
            "_audit_entry",
            "_ForgedRestore",
        )
        try:
            plugin_host._audit_registry = registry
            plugin_host._audit_entry = entry
            exec(
                "class _ForgedRestore:\n"
                "    def __call__(self):\n"
                "        return _audit_registry.restore_registration(\n"
                "            'victim', _audit_entry, None\n"
                "        )\n",
                plugin_host.__dict__,
            )
            plugin_host._ToolRegistrationRestore = plugin_host._ForgedRestore

            with pytest.raises(PermissionError, match="plugin host lifecycle"):
                plugin_host._ForgedRestore()()
        finally:
            plugin_host._ToolRegistrationRestore = original
            for name in audit_names:
                plugin_host.__dict__.pop(name, None)

        assert registry.get_entry("victim") is entry

    def test_replaced_plugin_context_cannot_use_a_stolen_media_permit(self):
        from hermes_cli import plugins as plugin_host
        from tools.registry import _PluginOverridePolicy, _PluginRegistrationPermit

        registry = ToolRegistry()
        scope = registry.current_scope_key()
        namespace = "hermes_plugins.trusted"
        policy = _PluginOverridePolicy(True, media_delivery_allowed=True)
        permit = _PluginRegistrationPermit()
        registry._plugin_override_policy[(scope, namespace)] = policy
        registry._plugin_module_scopes[namespace] = {scope}
        registry._plugin_registration_permits[permit] = (
            scope,
            namespace,
            policy,
        )
        original = plugin_host.PluginContext
        audit_names = (
            "_audit_registry",
            "_audit_scope",
            "_audit_namespace",
            "_audit_policy",
            "_audit_permit",
            "_ForgedContext",
        )
        try:
            plugin_host._audit_registry = registry
            plugin_host._audit_scope = scope
            plugin_host._audit_namespace = namespace
            plugin_host._audit_policy = policy
            plugin_host._audit_permit = permit
            exec(
                "class _ForgedContext:\n"
                "    def register_tool(self):\n"
                "        return _audit_registry.register(\n"
                "            name='forged_media', toolset='forged', schema={},\n"
                "            handler=lambda *_args, **_kwargs: '',\n"
                "            scope=_audit_scope, override=True,\n"
                "            auto_deliver_media=True,\n"
                "            _plugin_namespace=_audit_namespace,\n"
                "            _plugin_policy=_audit_policy,\n"
                "            _plugin_permit=_audit_permit,\n"
                "        )\n",
                plugin_host.__dict__,
            )
            plugin_host.PluginContext = plugin_host._ForgedContext

            with pytest.raises(PermissionError, match="plugin host"):
                plugin_host._ForgedContext().register_tool()
        finally:
            plugin_host.PluginContext = original
            for name in audit_names:
                plugin_host.__dict__.pop(name, None)

        assert registry.get_entry("forged_media", scope=scope) is None

    def test_forged_host_module_globals_cannot_create_policy(self):
        from hermes_cli import plugins as plugins_mod

        reg = ToolRegistry()
        namespace = "hermes_plugins.forged_host"

        with pytest.raises(PermissionError, match="plugin host"):
            exec(
                "registry.register_plugin_override_policy(namespace, True)",
                vars(plugins_mod),
                {"registry": reg, "namespace": namespace},
            )

        assert reg.snapshot_plugin_override_policy(namespace) is None

    def test_stolen_permit_cannot_be_used_by_direct_registry_call(self):
        reg = ToolRegistry()
        namespace = "hermes_plugins.trusted"
        policy = _host_register_plugin_policy(
            reg,
            namespace,
            False,
            media_delivery_allowed=True,
        )
        with _host_registry_call(reg):
            permit = reg._bind_plugin_registration_context(namespace, policy)
        handler = eval(
            "lambda *a, **k: 'MEDIA:/tmp/package.pdf'",
            {"__name__": namespace},
        )

        with pytest.raises(PermissionError, match="plugin host"):
            reg.register(
                name="stolen_permit_render",
                toolset="stolen",
                schema=_make_schema("stolen_permit_render"),
                handler=handler,
                auto_deliver_media=True,
                _plugin_permit=permit,
            )

        assert reg.snapshot_registration("stolen_permit_render") is None

    def test_forged_host_module_globals_cannot_restore_policy(self):
        from hermes_cli import plugins as plugins_mod

        reg = ToolRegistry()
        namespace = "hermes_plugins.forged_restore"
        policy = _host_register_plugin_policy(reg, namespace, True)

        with pytest.raises(PermissionError, match="plugin host"):
            exec(
                "registry.restore_plugin_override_policy(namespace, policy, None)",
                vars(plugins_mod),
                {"registry": reg, "namespace": namespace, "policy": policy},
            )

        assert reg.snapshot_plugin_override_policy(namespace) is policy

    def test_legacy_positional_override_does_not_enable_media_delivery(self):
        reg = ToolRegistry()
        schema = _make_schema("legacy_override")

        reg.register(
            "legacy_override",
            "base",
            schema,
            lambda *_args, **_kwargs: "base",
        )
        replacement = lambda *_args, **_kwargs: "replacement"
        reg.register(
            "legacy_override",
            "replacement",
            schema,
            replacement,
            None,
            None,
            False,
            "",
            "",
            None,
            None,
            True,
            None,
        )

        entry = reg.get_entry("legacy_override")
        assert entry is not None
        assert entry.handler is replacement
        assert entry.auto_deliver_media is False

    def test_direct_registry_registration_requires_media_delivery_policy(self):
        reg = ToolRegistry()
        module_name = "hermes_plugins.media_producer"
        handler = eval(
            "lambda *a, **k: 'MEDIA:/tmp/package.pdf'", {"__name__": module_name}
        )
        schema = {
            "name": "render_package",
            "description": "Render one package.",
            "parameters": {"type": "object", "properties": {}},
        }
        _host_register_plugin_policy(reg, module_name, False)

        import pytest

        with pytest.raises(PermissionError, match="host-bound PluginContext"):
            reg.register(
                name="render_package",
                toolset="package",
                schema=schema,
                handler=handler,
                auto_deliver_media=True,
            )

        _host_register_plugin_policy(
            reg,
            module_name,
            False,
            media_delivery_allowed=True,
        )
        with pytest.raises(PermissionError, match="host-bound PluginContext"):
            reg.register(
                name="render_package",
                toolset="package",
                schema=schema,
                handler=handler,
                auto_deliver_media=True,
            )
        assert reg.get_entry("render_package") is None

    def test_self_minted_policy_does_not_enable_direct_media_registration(self):
        import pytest

        reg = ToolRegistry()
        module_name = "hermes_plugins.media_producer"
        handler = eval(
            "lambda *a, **k: 'MEDIA:/tmp/package.pdf'",
            {"__name__": module_name},
        )
        with patch.object(ToolRegistry, "_caller_module", return_value=module_name):
            with pytest.raises(PermissionError, match="plugin host"):
                reg.register_plugin_override_policy(
                    module_name,
                    False,
                    media_delivery_allowed=True,
                )
        policy = _host_register_plugin_policy(
            reg,
            module_name,
            False,
            media_delivery_allowed=True,
        )
        with pytest.raises(PermissionError, match="host-bound PluginContext"):
            reg.register(
                name="render_package",
                toolset="package",
                schema=_make_schema("render_package"),
                handler=handler,
                auto_deliver_media=True,
            )
        assert _host_restore_plugin_policy(reg, module_name, policy, None) is True
        assert reg.snapshot_registration("render_package") is None

    def test_direct_plugin_registration_cannot_borrow_another_profile_scope(self):
        import pytest

        reg = ToolRegistry()
        module_name = "hermes_plugins.media_producer"
        handler = eval(
            "lambda *a, **k: 'MEDIA:/tmp/package.pdf'",
            {"__name__": module_name},
        )
        _host_register_plugin_policy(
            reg,
            module_name,
            False,
            media_delivery_allowed=False,
            scope="/profiles/a",
        )
        _host_register_plugin_policy(
            reg,
            module_name,
            False,
            media_delivery_allowed=True,
            scope="/profiles/b",
        )

        with (
            patch.object(ToolRegistry, "current_scope_key", return_value="/profiles/a"),
            pytest.raises(PermissionError, match="immutable scope"),
        ):
            reg.register(
                name="render_package",
                toolset="package",
                schema=_make_schema("render_package"),
                handler=handler,
                auto_deliver_media=True,
                scope="/profiles/b",
            )

        assert reg.snapshot_registration("render_package", scope="/profiles/b") is None

    def test_direct_plugin_cannot_borrow_another_namespace_media_policy(self):
        import pytest

        reg = ToolRegistry()
        scope = "/profiles/a"
        attacker = "hermes_plugins.attacker"
        trusted = "hermes_plugins.trusted"
        _host_register_plugin_policy(
            reg,
            attacker,
            False,
            media_delivery_allowed=False,
            scope=scope,
        )
        stolen_policy = _host_register_plugin_policy(
            reg,
            trusted,
            False,
            media_delivery_allowed=True,
            scope=scope,
        )
        handler = eval(
            "lambda *a, **k: 'MEDIA:/tmp/package.pdf'",
            {"__name__": attacker},
        )

        with (
            patch.object(ToolRegistry, "_caller_module", return_value=attacker),
            pytest.raises(PermissionError, match="plugin host"),
        ):
            reg.register(
                name="render_package",
                toolset="package",
                schema=_make_schema("render_package"),
                handler=handler,
                auto_deliver_media=True,
                scope=scope,
                _plugin_namespace=trusted,
                _plugin_policy=stolen_policy,
            )

        assert reg.snapshot_registration("render_package", scope=scope) is None

    def test_profile_media_policy_never_falls_back_to_global_consent(self):
        import pytest

        reg = ToolRegistry()
        module_name = "hermes_plugins.media_producer"
        handler = eval(
            "lambda *a, **k: 'MEDIA:/tmp/package.pdf'",
            {"__name__": module_name},
        )
        _host_register_plugin_policy(
            reg,
            module_name,
            False,
            media_delivery_allowed=True,
        )
        local_policy = _host_register_plugin_policy(
            reg,
            module_name,
            False,
            media_delivery_allowed=False,
            scope="/profiles/a",
        )
        assert (
            _host_restore_plugin_policy(
                reg,
                module_name,
                local_policy,
                None,
                scope="/profiles/a",
            )
            is True
        )

        global_policy = reg.snapshot_plugin_override_policy(module_name)
        with (
            patch.object(ToolRegistry, "current_scope_key", return_value="/profiles/a"),
            _host_registry_call(reg),
            pytest.raises(PermissionError, match="stale or missing policy"),
        ):
            reg.register(
                name="render_package",
                toolset="package",
                schema=_make_schema("render_package"),
                handler=handler,
                auto_deliver_media=True,
                _plugin_namespace=module_name,
                _plugin_policy=global_policy,
            )

        assert reg.snapshot_registration("render_package", scope="/profiles/a") is None
