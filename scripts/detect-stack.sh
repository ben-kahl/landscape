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
  command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1 && return 0
  [[ -e /dev/kfd ]] && return 0
  command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -Eiq 'vga.*(amd|ati|radeon)' && return 0
  return 1
}

OS="$(detect_os)"
PROFILE=""
OLLAMA_URL="http://ollama:11434"
REASON=""

case "$OS" in
  macos)
    PROFILE="host"
    OLLAMA_URL="http://host.docker.internal:11434"
    REASON="macOS: Docker cannot access Metal. Run Ollama natively on the host (brew install ollama && ollama serve)."
    ;;
  linux|windows)
    if has_nvidia; then
      PROFILE="gpu-nvidia"
      REASON="NVIDIA GPU detected via nvidia-smi."
    elif has_amd; then
      PROFILE="gpu-amd"
      REASON="AMD GPU detected. Using ollama/ollama:rocm image."
    else
      PROFILE="cpu"
      REASON="No GPU detected. Falling back to CPU (slow but functional)."
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
echo "OLLAMA_URL: $OLLAMA_URL"
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

awk -v profile="$PROFILE" -v ollama="$OLLAMA_URL" '
  BEGIN { p=0; o=0 }
  /^COMPOSE_PROFILES=/ { print "COMPOSE_PROFILES=" profile; p=1; next }
  /^OLLAMA_URL=/       { print "OLLAMA_URL=" ollama; o=1; next }
                       { print }
  END {
    if (!p) print "COMPOSE_PROFILES=" profile
    if (!o) print "OLLAMA_URL=" ollama
  }
' "$SOURCE_FILE" > "$ENV_FILE"

echo "Wrote $ENV_FILE"
echo

case "$PROFILE" in
  host)
    cat <<'EOF'
Next steps (macOS host-Ollama mode):
  1. Install Ollama on the host:   brew install ollama
  2. Start it:                     ollama serve &
  3. Pull the model:               ollama pull llama3.1:8b
  4. Bring up the stack:           docker compose up
EOF
    ;;
  gpu-nvidia)
    cat <<'EOF'
Next steps (NVIDIA GPU — full acceleration for LLM + embeddings):
  1. Ensure nvidia-container-toolkit is installed on the host.
  2. docker compose up -d neo4j qdrant ollama-nvidia
  3. docker compose exec ollama-nvidia ollama pull llama3.1:8b
  4. docker compose up app-gpu
EOF
    ;;
  gpu-amd)
    cat <<'EOF'
Next steps (AMD GPU / ROCm — LLM on GPU, embeddings on CPU):
  1. Ensure your user is in the 'video' and 'render' groups.
  2. docker compose up -d neo4j qdrant ollama-amd
  3. docker compose exec ollama-amd ollama pull llama3.1:8b
  4. docker compose up app
EOF
    ;;
  cpu)
    cat <<'EOF'
Next steps (CPU-only, expect 5-30s per extraction):
  1. docker compose up -d neo4j qdrant ollama-cpu
  2. docker compose exec ollama-cpu ollama pull llama3.1:8b
  3. docker compose up app
EOF
    ;;
esac
