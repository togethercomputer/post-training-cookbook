#!/usr/bin/env bash
# Symlink the Post Training Cookbook's bundled skills into a `.claude/skills/` directory
# so Claude Code can discover them. Symlinks keep repo edits live; --copy
# freezes a snapshot instead.
set -euo pipefail

usage() {
  cat <<'EOF'
Install the Post Training Cookbook's bundled skills so Claude Code can discover them.

Usage:
  ./skills/install.sh            # install for your user (~/.claude/skills)
  ./skills/install.sh --project  # install for this repo only (.claude/skills)
  ./skills/install.sh --copy     # copy instead of symlink (for a frozen install)
  ./skills/install.sh --uninstall
EOF
}

SKILLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SKILLS_DIR/.." && pwd)"

target="$HOME/.claude/skills"
mode="symlink"
uninstall=false

for arg in "$@"; do
  case "$arg" in
    --project)   target="$REPO_ROOT/.claude/skills" ;;
    --copy)      mode="copy" ;;
    --uninstall) uninstall=true ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$target"

for skill in "$SKILLS_DIR"/*/; do
  [ -f "$skill/SKILL.md" ] || continue
  name="$(basename "$skill")"
  src="$SKILLS_DIR/$name"
  dest="$target/$name"

  rm -rf "$dest"   # replace any prior install so re-running is idempotent
  $uninstall && { echo "removed  $dest"; continue; }

  if [ "$mode" = copy ]; then
    cp -R "$src" "$dest"; echo "copied   $dest"
  else
    ln -s "$src" "$dest"; echo "linked   $dest -> $src"
  fi
done

$uninstall || echo "
Done. Restart Claude Code (or start a new session) to pick up the skill(s)."
