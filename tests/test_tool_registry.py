"""Tests for the MCP tool registry."""

import asyncio

from server.app import list_tools, mcp


def test_list_tools_matches_registered_tools():
    """The discovery helper lists every callable MCP tool name."""
    registered_tools = asyncio.run(mcp.list_tools())

    assert set(asyncio.run(list_tools())) == {tool.name for tool in registered_tools}


def test_list_tools_derives_from_registry_not_hardcoded():
    """A registry edit changes the exposed list — names are not hardcoded."""
    exposed_before = set(asyncio.run(list_tools()))

    def remove_probe(name: str) -> None:
        """Remove the probe tool, tolerating an already-removed registry entry."""
        try:
            mcp.local_provider.remove_tool(name)
        except KeyError:
            pass

    async def scenario():
        @mcp.tool
        def _registry_probe_tool() -> str:
            """Probe tool used to prove the exposed list follows the registry."""
            return "probe"

        try:
            exposed_with = set(await list_tools())
            assert "_registry_probe_tool" in exposed_with
            assert exposed_with - exposed_before == {"_registry_probe_tool"}

            remove_probe("_registry_probe_tool")

            exposed_after = set(await list_tools())
            assert "_registry_probe_tool" not in exposed_after
            assert exposed_after == exposed_before
        finally:
            # Never leak the probe tool into the shared module-level registry
            # if an assertion above fails.
            remove_probe("_registry_probe_tool")

    asyncio.run(scenario())
