# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMR 주행(navigation) 태스크 모음.

- ``config/amr_diff`` : 차동구동 AMR 의 목표 pose 도달 (위치 + 방향)

관측은 실로봇의 ROS2 Nav2 스택 산출물(AMCL pose, 휠 오도메트리)에 대응하고,
액션은 ``/cmd_vel`` (v, ω) 에 대응한다. 즉 학습된 정책이 그대로 ``/cmd_vel``
퍼블리셔가 되는 구성이다.
"""
