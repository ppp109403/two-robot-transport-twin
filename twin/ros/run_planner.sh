#!/usr/bin/env bash
# MPC 플래너 노드 + map_server + RViz.
#
# RViz 가 뜨면:
#   "2D Pose Estimate"  -> 물체 시작 자세
#   "2D Goal Pose"      -> 물체 목표 자세  (여기서 풀기 시작. 수십 초 ~ 수 분)
#   화면의 INITIAL 버튼 -> 시작 자세 유지 (Isaac 로봇이 그 자리로 간다)
#   EXECUTE 버튼        -> 재생 시작
#
# 파라미터는 **v29 프로파일**이다 (twin/configs/policy_v29.yaml).
# 노드 기본값은 우리 한계의 3 배가 넘어서 그대로 쓰면 EE 속력이 1.2 m/s 가 된다.
set -eo pipefail
source "$(dirname "$0")/env.sh"

MAP=${MAP:-$TWIN_REPO/handoff/map_munji_3f_2025_wide/munji_3f_2025_wide_unknown_fixed.yaml}
CSV=${CSV:-$TWIN_REPO/twin/data/plans/live}
mkdir -p "$CSV"

echo "== 환경 =="; twin_env_check
echo "== 맵 $MAP"
echo "== CSV $CSV  (풀린 계획이 여기 남는다)"

exec ros2 launch mobile_manipulator_trajectory two_robot_nlp.launch.py \
    map_yaml:="$MAP" csv_dir:="$CSV" use_rviz:=true \
    obj_len:=1.50 obj_wid:=0.30 n_obj_disks:=10 obj_margin:=0.05 \
    base_len:=0.68 base_wid:=0.41 n_base_disks:=5 \
    base_margin:=0.18 map_inflation:=0.05 \
    reach_min:=0.15 reach_max:=0.60 ee_forward_min:=-0.30 \
    v_max:=0.30 v_min:=-0.20 w_max:=0.60 \
    obj_v_max:=0.12 obj_w_max:=0.08 ee_v_max:=0.20 \
    v_nom:=0.10 dt:=0.15 max_steps:=700 astar_clearance:=0.35 \
    stream_rate:=100.0 "$@"
