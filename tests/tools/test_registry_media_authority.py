from unittest.mock import patch

import pytest

from tests.tools.test_registry import (
    _host_register_plugin_policy,
    _host_restore_plugin_policy,
    _make_schema,
)
from tools.registry import ToolRegistry


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


class TestPluginMediaDeliveryAuthorization:
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
        with patch.object(
            ToolRegistry, "_caller_is_plugin_host_method", return_value=True
        ):
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
            patch.object(
                ToolRegistry, "_caller_is_plugin_host_method", return_value=True
            ),
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
