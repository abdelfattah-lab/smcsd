#!/usr/bin/env bash
# Apply the SMC core-hook patches onto the vendored (pristine upstream) SGLang.
#
# The 3rdparty/sglang submodule pins an UNMODIFIED upstream release tag; the
# SGLang-side changes SMC needs live in patches/*.patch in this repo and are
# applied here as git commits (git am).  Run this after every
# `git submodule update` — updating the submodule resets it to the pristine
# pin and drops the applied patches.  Idempotent: re-running is a no-op.
#
# Usage:  scripts/apply_sglang_patches.sh [--reverse]
#   --reverse   remove the applied patches (reset back to the pinned base)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGL="$REPO_ROOT/3rdparty/sglang"
PATCHES=("$REPO_ROOT"/patches/*.patch)
MARKER="SMC speculative decoding: consolidated core hooks"

if [ ! -e "$SGL/.git" ]; then
    echo "error: submodule not initialized — run: git submodule update --init --recursive" >&2
    exit 1
fi

applied() {
    # No pipeline here: `grep -q | git log` races SIGPIPE under pipefail.
    [ -n "$(git -C "$SGL" log --grep="$MARKER" --fixed-strings --format=%H -1)" ]
}

if [ "${1:-}" = "--reverse" ]; then
    if ! applied; then
        echo "patches not applied; nothing to reverse."
        exit 0
    fi
    base="$(git -C "$SGL" log --grep="$MARKER" --fixed-strings --format=%H | tail -1)"
    git -C "$SGL" reset --hard "$base~1"
    echo "reversed: submodule back at $(git -C "$SGL" log --oneline -1)"
    exit 0
fi

if applied; then
    echo "patches already applied: $(git -C "$SGL" log --oneline -1)"
    exit 0
fi

if ! git -C "$SGL" diff --quiet || ! git -C "$SGL" diff --cached --quiet; then
    echo "error: 3rdparty/sglang has local modifications; commit/stash them first." >&2
    exit 1
fi

echo "applying ${#PATCHES[@]} patch(es) onto $(git -C "$SGL" log --oneline -1) ..."
if ! git -C "$SGL" am --3way "${PATCHES[@]}"; then
    echo "error: git am failed (likely a submodule pin / patch mismatch)." >&2
    echo "       inspect with: git -C 3rdparty/sglang am --show-current-patch" >&2
    echo "       abort with:   git -C 3rdparty/sglang am --abort" >&2
    exit 1
fi
echo "done: $(git -C "$SGL" log --oneline -1)"
echo "next: uv pip install -e 3rdparty/sglang/python && uv pip install -e ."
