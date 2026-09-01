# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""엔드이펙터가 목표 pose를 추종(reach/tracking)하는 태스크 모음.

- ``config/franka``          : 고정형 매니퓰레이터 (Franka Panda, 7DOF)  - 기준 실험
- ``config/fairino_fr3``     : 고정형 매니퓰레이터 (Fairino FR3, 6DOF)   - franka 와 동일 보상
- ``config/ridgeback_franka``: 모바일 매니퓰레이터 (Clearpath Ridgeback + Franka) - 홀로노믹 베이스
- ``config/amr_fr3``         : 모바일 매니퓰레이터 (자체 AMR + Fairino FR3) - **차동구동 베이스**
  ridgeback 의 월드 고정 보상 스택 + ``nav`` 태스크의 차동구동 (v, ω) 액션 조합이다.

franka 와 fairino_fr3 는 보상/네트워크/PPO 설정이 동일하고 로봇만 다르므로
그대로 A/B 비교가 된다. 로그는 각각 ``franka_reach`` / ``fr3_reach`` 로 분리된다.
"""
