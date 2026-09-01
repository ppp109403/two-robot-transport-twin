# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 v2 (sim-to-real) 용 PPO 하이퍼파라미터.

v1 과 **동일**하다. 이번 사이클에서 바뀐 것은 액추에이터 현실성(지연/랜덤화)과
관측의 회전 표현뿐이므로, 알고리즘 쪽은 변수 통제를 위해 그대로 둔다.
바꾼 것은 두 가지::

    experiment_name : fr3_reach -> fr3_reach_v2   (로그 분리)
    max_iterations  : 1000 -> 2000                (지연 + 랜덤화로 문제가 어려워졌다)

max_iterations 를 올린 근거: v1 은 1000 iter 에서 위치 5.7 cm 였고 2000 까지 이어
학습해서 5.0 mm 가 나왔다 (최저 4.24 mm @it1854). v2 는 DR 이 들어가 더 어려우므로
처음부터 2000 으로 간다.

**학습 중 판단 기준**: ``Metrics/ee_pose/position_error`` 를 보다가 it1500 근처에서
10 mm 위로 정체하면 연장을 고려한다. v1 과 비슷한 궤적이면 정상이다.
"""

from isaaclab.utils import configclass

from ee_track_ppo.tasks.manager_based.reach.config.fairino_fr3.agents.rsl_rl_ppo_cfg import (
    FairinoFR3ReachPPORunnerCfg,
)


@configclass
class FairinoFR3Sim2RealPPORunnerCfg(FairinoFR3ReachPPORunnerCfg):
    max_iterations = 2000
    experiment_name = "fr3_reach_v2"


##
# 보상 스윕 (v2a ~ v2e). 액추에이터 설정은 v2 와 동일하고 보상만 다르므로
# PPO 하이퍼파라미터도 v2 와 동일하게 두고 로그만 분리한다.
##


@configclass
class FR3SweepAPPORunnerCfg(FairinoFR3Sim2RealPPORunnerCfg):
    experiment_name = "fr3_reach_v2a"


@configclass
class FR3SweepBPPORunnerCfg(FairinoFR3Sim2RealPPORunnerCfg):
    experiment_name = "fr3_reach_v2b"


@configclass
class FR3SweepCPPORunnerCfg(FairinoFR3Sim2RealPPORunnerCfg):
    experiment_name = "fr3_reach_v2c"


@configclass
class FR3SweepDPPORunnerCfg(FairinoFR3Sim2RealPPORunnerCfg):
    experiment_name = "fr3_reach_v2d"


@configclass
class FR3SweepEPPORunnerCfg(FairinoFR3Sim2RealPPORunnerCfg):
    experiment_name = "fr3_reach_v2e"


@configclass
class FR3SweepFPPORunnerCfg(FairinoFR3Sim2RealPPORunnerCfg):
    experiment_name = "fr3_reach_v2f"
