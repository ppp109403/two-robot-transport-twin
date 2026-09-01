# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 v4 사다리용 PPO 하이퍼파라미터.

v3 계열과 동일하다. 사다리의 각 단계가 **환경 쪽 단일 변수**만 바꾸므로
알고리즘은 변수 통제를 위해 건드리지 않는다. 로그만 분리한다.

``max_iterations`` 는 5000. **resume 연장은 쓰지 말 것** - v3b 에서 3000 + 1000 x2 로
이어붙인 판이 통짜 5000 보다 나빴다 (3.54 mm / 2.79 deg vs 2.82 / 2.48).
resume 마다 성적이 76 mm / 51 deg 로 튀었다가 회복하는데, 탐색 노이즈가 0.03 까지
죽은 뒤에는 회복하지 못하고 헤맸다. 더 돌려야 하면 **처음부터 10000 으로** 돌린다.
"""

from isaaclab.utils import configclass

from ee_track_ppo.tasks.manager_based.reach.config.fairino_fr3_v3.agents.rsl_rl_ppo_cfg import (
    FairinoFR3V3dPPORunnerCfg,
)


@configclass
class FR3V4aPPORunnerCfg(FairinoFR3V3dPPORunnerCfg):
    max_iterations = 5000
    experiment_name = "fr3_reach_v4a"


@configclass
class FR3V4bPPORunnerCfg(FR3V4aPPORunnerCfg):
    """v4b: 6D 회전 관측. 관측이 38 -> 42 로 늘지만 네트워크 크기는 그대로 둔다."""

    experiment_name = "fr3_reach_v4b"


@configclass
class FR3V4cPPORunnerCfg(FR3V4bPPORunnerCfg):
    """v4c: 원통 껍질 + pitch ±45도.

    ``max_iterations`` 10000. v4b 에서 6D 관측이 5000 으로는 부족하다는 것이 확인됐다
    (5000 시점 5.41도 -> 10000 시점 2.01도). 껍질과 pitch 확대로 문제가 더 어려워졌으니
    10000 을 유지한다.
    """

    max_iterations = 10000
    experiment_name = "fr3_reach_v4c"


@configclass
class FR3V4ePPORunnerCfg(FR3V4cPPORunnerCfg):
    experiment_name = "fr3_reach_v4e"


@configclass
class FR3V4fPPORunnerCfg(FR3V4cPPORunnerCfg):
    experiment_name = "fr3_reach_v4f"


@configclass
class FR3V4gPPORunnerCfg(FR3V4cPPORunnerCfg):
    experiment_name = "fr3_reach_v4g"


@configclass
class FR3V4hPPORunnerCfg(FR3V4cPPORunnerCfg):
    """v4h: 감김 여유 관측. 관측 42 -> 44."""

    experiment_name = "fr3_reach_v4h"


@configclass
class AmrFr3ArmPPORunnerCfg(FR3V4cPPORunnerCfg):
    """차체 정지 팔 태스크. v4c 러너를 그대로 쓰고 실험명만 분리한다."""

    experiment_name = "amr_fr3_arm"
