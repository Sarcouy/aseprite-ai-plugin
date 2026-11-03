#!/usr/bin/env bash
# Aseprite skips symlinked extension folders, so the folder is real and each file is a symlink.
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
target_dir="${HOME}/.config/aseprite/extensions/aseprite-ai-inpaint"
bridge="$(cd "${project_dir}" && poetry env info -p)/bin/ai-inpaint-bridge"

mkdir -p "${target_dir}"
for file in package.json plugin.lua inpaint.lua; do
  ln -sfn "${project_dir}/extension/${file}" "${target_dir}/${file}"
done

if [[ ! -f "${target_dir}/__pref.lua" ]]; then
  printf 'return {bridge_command="%s",margin=8}' "${bridge}" > "${target_dir}/__pref.lua"
fi

echo "Extension installed in ${target_dir}, restart Aseprite."
