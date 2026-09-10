#!/usr/bin/env bash
# Open a shell on the Colab GPU runtime from this terminal.
#
# Colab runtimes have no inbound route, so the tunnel is opened from inside the
# runtime (one line in Colab's own terminal) and met here. Argument is the
# trycloudflare hostname that line printed.
#
#   ./colab_connect.sh https://some-words.trycloudflare.com
#
# Leaves an SSH session on the runtime and, while it lives, forwards the
# runtime's vLLM port back to this machine on 127.0.0.1:8000 -- which is what
# resolve_backend() already returns, so nothing in GeneVariate needs reconfiguring.
set -euo pipefail

HOST=${1:?usage: colab_connect.sh <trycloudflare-hostname>}
HOST=${HOST#https://}
HOST=${HOST#http://}
HOST=${HOST%/}

CF=${CLOUDFLARED:-$HOME/.local/bin/cloudflared}
KEY=${COLAB_KEY:-$HOME/.ssh/colab_gv}
PORT=${COLAB_PORT:-2222}

"$CF" access tcp --hostname "$HOST" --url "localhost:$PORT" >/tmp/colab_tunnel.log 2>&1 &
CF_PID=$!
trap 'kill $CF_PID 2>/dev/null || true' EXIT

for _ in $(seq 40); do
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
    sleep 0.5
done

ssh -i "$KEY" -p "$PORT" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o ServerAliveInterval=30 \
    -L 8000:localhost:8000 \
    root@localhost "$@"
