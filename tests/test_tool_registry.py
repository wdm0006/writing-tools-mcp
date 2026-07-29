"""Tests for the MCP tool registry."""

import asyncio

from server.app import list_tools, mcp


def test_list_tools_matches_registered_tools():
    """The discovery helper lists every callable MCP tool name."""
    registered_tools = asyncio.run(mcp.list_tools())

    assert set(list_tools()) == {tool.name for tool in registered_tools}
