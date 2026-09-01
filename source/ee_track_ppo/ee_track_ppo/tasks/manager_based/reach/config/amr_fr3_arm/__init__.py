# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""차체 정지 + 팔만 제어하는 태스크 등록 (2단계 격리 검증용)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Reach-AMR-FR3-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arm_only_env_cfg:AmrFr3ArmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ArmPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-Arm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arm_only_env_cfg:AmrFr3ArmEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ArmPPORunnerCfg",
    },
)
