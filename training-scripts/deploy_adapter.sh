#!/bin/bash
# Deploy a trained LoRA adapter

set -e

ADAPTER_PATH=$1
if [ -z "$ADAPTER_PATH" ]; then
    echo "Usage: $0 <adapter-path>"
    echo "Example: $0 lora-adapters/tool-calling-20240128-120000"
    exit 1
fi

# Ensure we're in the right directory
cd /root/qwen-server

# Convert relative path to absolute if needed
if [[ ! "$ADAPTER_PATH" = /* ]]; then
    ADAPTER_PATH="/root/qwen-server/$ADAPTER_PATH"
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter directory not found: $ADAPTER_PATH"
    exit 1
fi

echo "Deploying adapter: $ADAPTER_PATH"

CURRENT="/root/qwen-server/lora-adapters/current"

# Handle backup of existing current adapter
if [ -L "$CURRENT" ]; then
    # It's a symlink - back up the target
    echo "Backing up current adapter (symlink)..."
    TARGET=$(readlink -f "$CURRENT")
    if [ -d "$TARGET" ] && [[ "$TARGET" != "$ADAPTER_PATH" ]]; then
        BACKUP_NAME="archive/$(basename $TARGET)"
        mkdir -p "/root/qwen-server/lora-adapters/archive"
        mv "$TARGET" "/root/qwen-server/lora-adapters/$BACKUP_NAME" 2>/dev/null || true
    fi
    rm -f "$CURRENT"
elif [ -d "$CURRENT" ]; then
    # It's a directory - check if it has content
    if [ "$(ls -A $CURRENT 2>/dev/null)" ]; then
        echo "Backing up current adapter (directory)..."
        BACKUP_NAME="archive/current-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "/root/qwen-server/lora-adapters/archive"
        mv "$CURRENT" "/root/qwen-server/lora-adapters/$BACKUP_NAME"
    else
        # Empty directory, just remove it
        rm -rf "$CURRENT"
    fi
fi

# Create symlink to new adapter (use relative path for Docker compatibility)
echo "Linking new adapter..."
ADAPTER_NAME=$(basename "$ADAPTER_PATH")
cd "/root/qwen-server/lora-adapters"
ln -s "$ADAPTER_NAME" "current"

echo "✓ Adapter deployed!"
echo "  Adapter: $ADAPTER_PATH"
echo "  Symlink: $CURRENT -> $ADAPTER_PATH"
echo
echo "Next steps:"
echo "1. Restart vLLM to load new adapter:"
echo "   cd /root/qwen-server"
echo "   docker-compose --profile production restart vllm-server"
echo
