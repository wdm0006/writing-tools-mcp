#!/bin/bash
set -e

# Build script for creating .mcpb bundle for Claude Desktop
# This script packages the Writing Tools MCP server as a Claude Desktop bundle

echo "Building Writing Tools MCP Bundle..."

# Configuration
BUILD_DIR="build/mcpb"
BUNDLE_NAME="writing-tools-mcp.mcpb"
VERSION=$(grep -Po '(?<=version = ")[^"]*' pyproject.toml)

# Clean previous build
echo "Cleaning previous build..."
rm -rf build/
rm -f "$BUNDLE_NAME"

# Create build directory structure
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"

# Copy manifest
echo "Copying manifest.json..."
cp manifest.json "$BUILD_DIR/"

# Copy server code (server/ includes the bundled data/ baselines)
echo "Copying server files..."
cp run_server.py "$BUILD_DIR/"
cp -r server "$BUILD_DIR/"

# Copy configuration
echo "Copying configuration..."
if [ -f ".mcp-config.yaml" ]; then
    cp .mcp-config.yaml "$BUILD_DIR/"
fi

# Copy README and LICENSE
echo "Copying documentation..."
cp README.md "$BUILD_DIR/"
if [ -f "LICENSE" ]; then
    cp LICENSE "$BUILD_DIR/"
fi

# Create the bundle (zip file with .mcpb extension)
echo "Creating bundle archive..."
cd build/mcpb
zip -r "../../$BUNDLE_NAME" .
cd ../..

echo ""
echo "✅ Bundle created successfully: $BUNDLE_NAME"
echo "📦 Version: $VERSION"
echo ""
echo "To install in Claude Desktop:"
echo "  1. Open the bundle file with Claude Desktop"
echo "  2. Or manually install using: mcpb install $BUNDLE_NAME"
echo ""
echo "Note: This bundle uses 'uv' for dependency management."
echo "Claude Desktop will automatically install dependencies on first run."
