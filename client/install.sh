#!/bin/sh
# Installs the Qwen agent on a Linux or macOS machine.
# The model keeps running on the server; this machine connects to it over SSH.
#
#   ssh user@server cat qwen-server/client/install.sh | sh -s -- user@server
#
# Then run `qwen` (or pick "Qwen" from the applications menu on Linux).
# Re-run it at any time to update to the server's current version.

set -eu

SERVER="${1:-}"
REMOTE_DIR="${2:-qwen-server}"   # repo location on the server, relative to its home
INSTALL_DIR="$HOME/.local/share/qwen-agent"
BIN_DIR="$HOME/.local/bin"
WORKSPACE="$HOME/qwen-workspace"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ -n "$SERVER" ] || fail "usage: install.sh user@server [remote-repo-dir]"

command -v python3 > /dev/null || fail "python3 was not found. Install Python 3.10 or newer."
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || fail "Python 3.10 or newer is required."
command -v ssh > /dev/null || fail "ssh was not found."

# Passwordless SSH to the server (the launcher cannot type a password)
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$SERVER" exit < /dev/null 2> /dev/null; then
    echo "Setting up an SSH key for $SERVER ..."
    [ -f "$HOME/.ssh/id_ed25519.pub" ] || ssh-keygen -q -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" < /dev/null
    echo "Enter your password for $SERVER one last time:"
    ssh-copy-id -i "$HOME/.ssh/id_ed25519.pub" "$SERVER" < /dev/tty
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" exit < /dev/null \
        || fail "could not set up passwordless SSH to $SERVER. Check that 'ssh $SERVER' works, then run this again."
fi

mkdir -p "$INSTALL_DIR" "$BIN_DIR" "$WORKSPACE"
scp -q "$SERVER:$REMOTE_DIR/scripts/agent.py" "$SERVER:$REMOTE_DIR/scripts/agent_ui.html" "$INSTALL_DIR/" < /dev/null \
    || fail "could not copy the agent from $SERVER:$REMOTE_DIR. Pass the repo location as the second argument."

cat > "$BIN_DIR/qwen" << LAUNCHER
#!/bin/sh
# Starts the Qwen agent UI. Extra arguments go to agent.py (try --help).
cd "$WORKSPACE" 2> /dev/null || cd "\$HOME"
exec python3 "$INSTALL_DIR/agent.py" --ui --ssh "$SERVER" "\$@"
LAUNCHER
chmod +x "$BIN_DIR/qwen"

if [ "$(uname)" = "Linux" ]; then
    mkdir -p "$HOME/.local/share/applications"
    cat > "$HOME/.local/share/applications/qwen.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=Qwen
Comment=Chat with the Qwen model on $SERVER
Exec=$BIN_DIR/qwen
Terminal=false
Categories=Development;
DESKTOP
fi

echo ""
echo "Installed. Start it with: $BIN_DIR/qwen"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) echo "($BIN_DIR is not on your PATH; add it to run just 'qwen'.)" ;; esac
echo "The assistant works in $WORKSPACE until you pick another folder with 'Change folder'."
