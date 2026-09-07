#!/usr/bin/env zsh
# Tensorboard launcher for the UltiMed fine-tune.
#
# Serves exactly the directory the checkpoints go to: NEMO_EXP_DIR is resolved
# from the .env file next to this script by perso/resolve_exp_dir.sh, the same
# helper training_start.sh uses. So when the runs move to the external drive the
# graphs follow them, with no --logdir to keep in sync by hand.
#
# Tensorboard itself is run through uvx, so nothing has to be installed in the
# training venv. Any argument this script does not recognise is passed straight
# through to tensorboard.
#
# Written with Claude Code.

emulate -L zsh
set -u

PORT=6006
typeset -a passthrough
passthrough=()

# Captured out here because zsh sets $0 to the function name inside a function.
script_name="${0:t}"

usage() {
  print "Usage: $script_name [options] [extra tensorboard args]"
  print ""
  print "Options:"
  print "  -p, --port PORT  Port to serve on (default: 6006)"
  print "  -h, --help       Show this help message"
  print ""
  print "Everything else is passed straight through to tensorboard; use -- to"
  print "stop option parsing if an argument would otherwise be eaten here."
  print ""
  print "The log directory is \$NEMO_EXP_DIR, read from the .env file next to"
  print "this script, so it always matches where training_start.sh writes its"
  print "checkpoints and event files."
}

while (( $# > 0 )); do
  case "$1" in
    -p|--port)
      # Guard the missing-value case: "$2" would be an unset-parameter error
      # under set -u.
      if (( $# < 2 )); then
        print -u2 "ERROR: $1 needs a port number."
        exit 1
      fi
      PORT="$2"
      shift 2
      ;;
    --port=*)
      # Handled here too, otherwise it would reach tensorboard as a second
      # --port and fight with the one this script passes.
      PORT="${1#--port=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      passthrough+=("$@")
      break
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

cd "${0:A:h}" || exit 1

# Sets and exports NEMO_EXP_DIR, or exits 1 if it is unset or the drive holding
# it is not mounted. Shared with training_start.sh on purpose.
source ./perso/resolve_exp_dir.sh

if [[ ! -d "$NEMO_EXP_DIR" ]]; then
  print -u2 "ERROR: $NEMO_EXP_DIR does not exist."
  print -u2 "       There is nothing to show until training has run at least once."
  exit 1
fi

# (N) is the nullglob qualifier: with no match the pattern expands to nothing
# instead of failing, so an empty run directory is a warning and not an error.
typeset -a event_files
event_files=("$NEMO_EXP_DIR"/**/events.out.tfevents.*(N))
if (( ${#event_files} == 0 )); then
  print -u2 "WARNING: no events.out.tfevents.* files under $NEMO_EXP_DIR yet."
  print -u2 "         Starting anyway, tensorboard picks up a run as it appears."
else
  print "Found ${#event_files} tensorboard event file(s) under $NEMO_EXP_DIR"
fi

print "Serving on:"
print "  http://localhost:$PORT"
lan_ip=""
if (( $+commands[hostname] )); then
  lan_ip="$(hostname -I 2>/dev/null)" || lan_ip=""
  lan_ip="${lan_ip%% *}"
fi
if [[ -n "$lan_ip" ]]; then
  print "  http://$lan_ip:$PORT  (other machines on the LAN, thanks to --bind_all)"
fi

# setuptools is pinned because tensorboard 2.20 still imports pkg_resources,
# which newer setuptools no longer provides; 81.0.0 is the version recorded as
# known-working here.
exec uvx --python 3.10.12 --with setuptools==81.0.0 --from tensorboard==2.20.0 \
  tensorboard --logdir "$NEMO_EXP_DIR" --bind_all --port "$PORT" "${passthrough[@]}"
