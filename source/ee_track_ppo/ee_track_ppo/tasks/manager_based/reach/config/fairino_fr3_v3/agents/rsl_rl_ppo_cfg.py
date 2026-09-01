# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 v3 용 PPO 하이퍼파라미터.

v2f 와 동일하다. 이번에 바뀌는 것은 중력과 목표 상자뿐이므로 알고리즘 쪽은
변수 통제를 위해 건드리지 않는다. 로그만 분리한다.

``max_iterations`` 는 3000 으로 올렸다. v2f 가 3000 에서 정체했고(말미 개선율 2.16%),
v3 는 상자 부피가 2.5배라 **같은 밀도로 덮으려면 더 걸린다.**
학습 중 ``Metrics/ee_pose/position_error`` 의 마지막 200 iteration 평균이 직전 200 대비
3% 미만이면 정체로 본다.
"""

from isaaclab.utils import configclass

from ee_track_ppo.tasks.manager_based.reach.config.fairino_fr3_v2.agents.rsl_rl_ppo_cfg import (
    FairinoFR3Sim2RealPPORunnerCfg,
)


@configclass
class FairinoFR3V3PPORunnerCfg(FairinoFR3Sim2RealPPORunnerCfg):
    max_iterations = 3000
    experiment_name = "fr3_reach_v3"


@configclass
class FairinoFR3V3bPPORunnerCfg(FairinoFR3V3PPORunnerCfg):
    """v3b: 보상 균형만 다르고 나머지는 v3 와 동일. 로그만 분리한다."""

    experiment_name = "fr3_reach_v3b"


@configclass
class FairinoFR3V3cPPORunnerCfg(FairinoFR3V3bPPORunnerCfg):
    """v3c: pitch 자유화만 다르다.

    ``max_iterations`` 를 5000 으로 올렸다. v3b 가 5000 통짜 실행에서 단조 하강으로
    2.82 mm / 2.48 deg 에 수렴했고, v3c 는 자세 자유도가 늘어 그보다 어렵다.

    **resume 연장은 쓰지 말 것.** v3b 에서 3000 + 1000 x2 로 이어붙인 판(누적 5000)이
    통짜 5000 보다 나빴다 (3.54 mm / 2.79 deg vs 2.82 / 2.48). resume 마다 성적이
    76 mm / 51 deg 로 튀었다가 회복하는데, 탐색 노이즈가 0.03 까지 죽은 뒤에는
    회복하지 못하고 헤맸다.
    """

    experiment_name = "fr3_reach_v3c"


@configclass
class FairinoFR3V3dPPORunnerCfg(FairinoFR3V3cPPORunnerCfg):
    """v3d: 실제 관절 한계(j3 ±150) 반영 + 상자 재조정. 나머지는 v3c 와 동일."""

    experiment_name = "fr3_reach_v3d"
