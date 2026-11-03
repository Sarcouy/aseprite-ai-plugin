#!/usr/bin/env bash
set -euo pipefail

comfyui_dir="${COMFYUI_DIR:-${HOME}/workspace/ComfyUI}"
cd "${comfyui_dir}"
exec .venv/bin/python main.py --lowvram "$@"
