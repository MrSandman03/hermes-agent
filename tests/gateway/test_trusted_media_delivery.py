from unittest.mock import patch

from gateway.run import (
    _collect_auto_append_media_tags,
    _snapshot_auto_delivery_entries,
)
from tools.registry import registry


def _register_trusted_media_tool(*, name, schema, handler, policy=None, permit=None):
    namespace = f"hermes_plugins.gateway_media_test_{name}"
    scope = registry.current_scope_key()
    with patch.object(
        type(registry), "_caller_is_plugin_host_method", return_value=True
    ):
        if policy is None:
            policy = registry.register_plugin_override_policy(
                namespace,
                False,
                media_delivery_allowed=True,
                scope=scope,
            )
            permit = registry._bind_plugin_registration_context(
                namespace,
                policy,
                scope=scope,
            )
        registry.register(
            name=name,
            toolset="test-package",
            schema=schema,
            handler=handler,
            auto_deliver_media=True,
            scope=scope,
            _plugin_namespace=namespace,
            _plugin_policy=policy,
            _plugin_permit=permit,
        )
    return namespace, policy, permit, scope


def _remove_trusted_media_tool(*, name, namespace, policy, scope):
    registry.deregister(name)
    with patch.object(
        type(registry), "_caller_is_plugin_host_method", return_value=True
    ):
        registry.restore_plugin_override_policy(
            namespace,
            policy,
            None,
            scope=scope,
        )


def test_gateway_auto_appends_media_from_registered_trusted_producer():
    tool_name = "test_render_package_media"
    schema = {
        "name": tool_name,
        "description": "Render a package.",
        "parameters": {"type": "object", "properties": {}},
    }
    namespace, policy, _permit, scope = _register_trusted_media_tool(
        name=tool_name,
        schema=schema,
        handler=lambda _args: "MEDIA:/tmp/package.pdf",
    )
    try:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-package", "function": {"name": tool_name}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call-package",
                "content": "Prepared files. MEDIA:/tmp/package.pdf",
            },
        ]

        tags, voice = _collect_auto_append_media_tags(messages)

        assert tags == ["MEDIA:/tmp/package.pdf"]
        assert voice is False
    finally:
        _remove_trusted_media_tool(
            name=tool_name,
            namespace=namespace,
            policy=policy,
            scope=scope,
        )


def test_gateway_rejects_result_after_producer_generation_changes():
    tool_name = "test_rotating_package_media"
    schema = {
        "name": tool_name,
        "description": "Render a package.",
        "parameters": {"type": "object", "properties": {}},
    }
    namespace, policy, permit, scope = _register_trusted_media_tool(
        name=tool_name,
        schema=schema,
        handler=lambda _args: "MEDIA:/tmp/old-package.pdf",
    )
    pinned = _snapshot_auto_delivery_entries()
    try:
        _register_trusted_media_tool(
            name=tool_name,
            schema=schema,
            handler=lambda _args: "MEDIA:/tmp/new-package.pdf",
            policy=policy,
            permit=permit,
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-package", "function": {"name": tool_name}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call-package",
                "content": "Prepared. MEDIA:/tmp/old-package.pdf",
            },
        ]

        tags, voice = _collect_auto_append_media_tags(
            messages,
            trusted_media_entries=pinned,
        )

        assert tags == []
        assert voice is False
    finally:
        _remove_trusted_media_tool(
            name=tool_name,
            namespace=namespace,
            policy=policy,
            scope=scope,
        )
