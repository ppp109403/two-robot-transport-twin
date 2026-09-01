#!/usr/bin/env bash
# ROS 2 쪽 공통 환경. run_planner.sh / run_bridge.sh 가 source 한다.
#
# 세 가지를 해결한다:
#
#  1) conda 가 python3 를 가로챈다
#     쉘 프롬프트에 (base) 가 떠 있으면 conda 의 python3 가 먼저 잡히고, 그러면
#     rclpy 가 안 보인다. PATH 에서 conda 를 걷어낸다.
#
#  2) casadi 가 ROS 파이썬에 없다
#     /opt/ros/jazzy 는 py3.12.3 인데 casadi 는 사용자 site-packages
#     (~/.local/lib/python3.12/site-packages) 에만 있다. ABI 가 같으므로
#     PYTHONPATH 로 얹으면 그대로 import 된다. 설치할 필요 없다.
#
#  3) ros2_ws 오버레이 (플래너를 colcon 으로 빌드해 둔 경우)
#
# ★ set -u 금지 — ROS setup.bash 들이 언바운드 변수를 참조한다.
#
# 환경변수로 덮을 수 있는 것::
#
#     TWIN_ROS_DISTRO   ROS 2 배포판          (기본: jazzy)
#     TWIN_ROS_WS       colcon 워크스페이스   (기본: $HOME/ros2_ws)
#     TWIN_PY           인터프리터            (기본: /usr/bin/python3)

TWIN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TWIN_REPO

# conda 와 다른 venv 를 PATH 에서 제거.
#   (base) 프롬프트의 conda python3 와 .bashrc 가 켜 두는 venv 가 둘 다 python3 를
#   가로챈다. 어느 쪽이 잡히느냐에 따라 rclpy/casadi 유무가 달라져서 "어제는 됐는데"
#   가 생긴다. 그래서 **인터프리터를 못박는다.**
_strip="/miniconda3|/anaconda3|/miniforge3"
[ -n "$VIRTUAL_ENV" ] && _strip="$_strip|^${VIRTUAL_ENV}/bin\$"
PATH="$(echo "$PATH" | tr ':' '\n' | grep -Ev "$_strip" | paste -sd: -)"
export PATH
unset _strip
unset PYTHONHOME
unset CONDA_PREFIX
unset VIRTUAL_ENV

#: 이 스택이 쓰는 유일한 인터프리터. 시스템 3.12.3 = Jazzy 가 빌드된 그 파이썬이다.
export TWIN_PY="${TWIN_PY:-/usr/bin/python3}"

source "/opt/ros/${TWIN_ROS_DISTRO:-jazzy}/setup.bash"
TWIN_ROS_WS="${TWIN_ROS_WS:-$HOME/ros2_ws}"
if [ -f "$TWIN_ROS_WS/install/setup.bash" ]; then
    source "$TWIN_ROS_WS/install/setup.bash"
fi

# casadi (사용자 site-packages). 플래너가 NLP 를 풀려면 필요하다.
_pyver="$("$TWIN_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
export PYTHONPATH="$HOME/.local/lib/python${_pyver}/site-packages:${PYTHONPATH}"
unset _pyver

# 확인용
twin_env_check() {
    "$TWIN_PY" - <<'PY'
import sys
print("  python  ", sys.version.split()[0], sys.executable)
for m in ("rclpy", "casadi", "numpy", "scipy"):
    try:
        mm = __import__(m); print("  %-8s %s" % (m, getattr(mm, "__version__", "ok")))
    except Exception:
        print("  %-8s **없음**" % m)
PY
}
