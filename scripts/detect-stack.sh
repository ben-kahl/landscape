#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"

detect_os() {
  case "$(uname -s)" in
    Linux*)  echo "linux" ;;
    Darwin*) echo "macos" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    *)       echo "unknown" ;;
  esac
}

has_nvidia() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

has_amd() {
  # Require rocminfo to succeed AND report at least one GPU agent. lspci-based
  # detection catches integrated Radeon iGPUs (e.g. Ryzen APUs) where the ROCm
  # image is slower than CPU; /dev/kfd alone is the same hazard. If the user
  # has a discrete ROCm-capable GPU, rocminfo is the canonical install check.
  command -v rocminfo >/dev/null 2>&1 || return 1
  rocminfo 2>/dev/null | grep -Eiq '^\s*Device Type:\s*GPU'
}

# NOTE: Only the gpu-nvidia profile currently has a llama-server backend
# (llama-server-nvidia in docker-compose). CPU and AMD variants are a future
# follow-up — their compose services still reference Ollama images that no
# longer exist in the current compose file.
OS="$(detect_os)"
PROFILE=""
LLM_BASE_URL="http://llama-server:8080/v1"
REASON=""

case "$OS" in
  macos)
    PROFILE="host"
    LLM_BASE_URL="http://host.docker.internal:8080/v1"
    REASON="macOS: Docker cannot access Metal. Run llama-server natively on the host."
    ;;
  linux|windows)
    if has_nvidia; then
      PROFILE="gpu-nvidia"
      REASON="NVIDIA GPU detected via nvidia-smi."
    elif has_amd; then
      PROFILE="gpu-amd"
      REASON="AMD GPU detected. (NOTE: CPU/AMD llama-server backend is a future follow-up.)"
    else
      PROFILE="cpu"
      REASON="No GPU detected. Falling back to CPU. (NOTE: CPU llama-server backend is a future follow-up.)"
    fi
    ;;
  *)
    PROFILE="cpu"
    REASON="Unknown OS. Defaulting to CPU."
    ;;
esac

echo "OS: $OS"
echo "Profile: $PROFILE"
echo "Reason: $REASON"
echo "LLM_BASE_URL: $LLM_BASE_URL"
echo

if [[ ! -f "$ENV_EXAMPLE" ]]; then
  echo "error: $ENV_EXAMPLE not found" >&2
  exit 1
fi

SOURCE_FILE="$ENV_EXAMPLE"
if [[ -f "$ENV_FILE" ]]; then
  BACKUP="$ENV_FILE.bak.$(date +%s)"
  cp "$ENV_FILE" "$BACKUP"
  SOURCE_FILE="$BACKUP"
  echo "Preserving existing .env values (backup: $BACKUP)"
fi

awk -v profile="$PROFILE" -v llm_base_url="$LLM_BASE_URL" '
  BEGIN { p=0; o=0 }
  /^COMPOSE_PROFILES=/ { print "COMPOSE_PROFILES=" profile; p=1; next }
  /^LLM_BASE_URL=/     { print "LLM_BASE_URL=" llm_base_url; o=1; next }
                       { print }
  END {
    if (!p) print "COMPOSE_PROFILES=" profile
    if (!o) print "LLM_BASE_URL=" llm_base_url
  }
' "$SOURCE_FILE" > "$ENV_FILE"

echo "Wrote $ENV_FILE"
echo

case "$PROFILE" in
  host)
    cat <<'EOF'
Next steps (macOS host llama-server mode):
  1. Run llama-server natively on the host (see docs for setup).
     llama-server auto-downloads the GGUF via -hf flag on first run.
  2. Bring up the stack:  docker compose up
EOF
    ;;
  gpu-nvidia)
    cat <<'EOF'
Next steps (NVIDIA GPU — full acceleration for LLM + embeddings):
  1. Ensure nvidia-container-toolkit is installed on the host.
  2. docker compose --profile gpu-nvidia up -d
     llama-server-nvidia auto-downloads the GGUF via -hf on first start
     (no manual model pull needed).
  3. docker compose up app-gpu
EOF
    ;;
  gpu-amd)
    cat <<'EOF'
Next steps (AMD GPU / ROCm):
  NOTE: The CPU/AMD llama-server backend is a future follow-up.
        Only the gpu-nvidia profile currently has a llama-server service.
  1. Ensure your user is in the 'video' and 'render' groups.
  2. docker compose up -d neo4j qdrant
  3. docker compose up app
EOF
    ;;
  cpu)
    cat <<'EOF'
Next steps (CPU-only):
  NOTE: The CPU llama-server backend is a future follow-up.
        Only the gpu-nvidia profile currently has a llama-server service.
  1. docker compose up -d neo4j qdrant
  2. docker compose up app
EOF
    ;;
esac
