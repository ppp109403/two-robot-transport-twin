#!/usr/bin/env bash
# ROS 2 <-> Isaac 공유메모리 다리. 플래너와 Isaac 사이에서 돈다.
set -eo pipefail
source "$(dirname "$0")/env.sh"
echo "== 환경 =="; twin_env_check
exec "$TWIN_PY" "$TWIN_REPO/twin/ros/twin_bridge.py" "$@"
