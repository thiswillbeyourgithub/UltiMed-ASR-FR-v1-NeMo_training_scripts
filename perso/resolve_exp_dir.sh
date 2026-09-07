# shellcheck shell=sh
# Resolve NEMO_EXP_DIR, the directory the training stack writes its experiment
# output to: checkpoints, .nemo exports and tensorboard events.
#
# This file is meant to be SOURCED, not executed: `. ./perso/resolve_exp_dir.sh`.
# It is sourced by training_start.sh and tensorboard_start.sh so the launcher and
# the viewer always agree on where the runs live. The caller must already have
# cd'd to the repo root, because .env is looked up next to it.
#
# Deliberately POSIX sh: training_start.sh is bash and tensorboard_start.sh is
# zsh, and both source this file, so no bashisms and no zshisms here.
#
# On success NEMO_EXP_DIR is set and exported. On failure it calls exit 1, which
# terminates the sourcing script: no caller should carry on without a usable
# path. It never creates anything, callers that need the directory to exist
# create it themselves.
#
# Added with Claude Code.

# NEMO_EXP_DIR comes from the .env file at the repo root. That file is gitignored
# because the path contains the username, so nothing machine-specific ends up in
# git; the tracked template is .env.example.
#
# Kept off the system disk: checkpoints for this model are ~2.5 GB each and
# exp_manager keeps the top 2 plus a -last plus a .nemo export.
#
# An already-exported NEMO_EXP_DIR wins over the .env value, so a one-off
# `NEMO_EXP_DIR=/somewhere ./training_start.sh` still works.
_preset_exp_dir="${NEMO_EXP_DIR:-}"
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
if [ -n "$_preset_exp_dir" ]; then
  NEMO_EXP_DIR="$_preset_exp_dir"
fi
unset _preset_exp_dir

if [ -z "${NEMO_EXP_DIR:-}" ]; then
  echo "ERROR: NEMO_EXP_DIR is not set." >&2
  echo "       Run 'cp .env.example .env' and set NEMO_EXP_DIR to the" >&2
  echo "       checkpoint directory on your external drive." >&2
  exit 1
fi
export NEMO_EXP_DIR

# Refuse to run if the drive is not mounted. When it is unplugged the mount
# point does not exist, and a bare mkdir -p would happily build the whole tree
# on the system disk and then fill it with multi-GB checkpoints.
_exp_parent="$(dirname "$NEMO_EXP_DIR")"
if [ ! -d "$_exp_parent" ]; then
  echo "ERROR: $_exp_parent does not exist, the drive looks unmounted." >&2
  echo "       Plug it in (or fix NEMO_EXP_DIR in .env) before training." >&2
  exit 1
fi
unset _exp_parent
