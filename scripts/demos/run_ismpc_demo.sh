#!/usr/bin/env bash
# Launches the ISMPC CoM-height sine demo in a viewer.
#
# IMPORTANT: this task only does something visible if etc/mc_rtc.yaml's
# Enabled is the ISMPC walking controller, not Posture (the default the repo
# ships with). Check/update that before running, e.g.:
#   Enabled: ismpc_walking
# (use whatever exact name `Enabled` needs -- check
# lib/mc_controller/etc/ismpc_walking.yaml or your own controller's
# CONTROLLER_CONSTRUCTOR name if unsure).
#
# Extra args are forwarded to `play`, same as run_test_mc_rtc.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."

TASK_ID=$(uv run list-envs 2>/dev/null | grep -i "Ismpc-Demo" | head -1)
if [ -z "$TASK_ID" ]; then
  echo "Could not find an Ismpc-Demo task id via list-envs." >&2
  echo "Check etc/mc_rtc.yaml's Enabled/MainRobot and that ismpc_demo/__init__.py registered without error." >&2
  exit 1
fi

echo "[ismpc_demo] launching: $TASK_ID"
uv run play "$TASK_ID" --agent zero "$@"