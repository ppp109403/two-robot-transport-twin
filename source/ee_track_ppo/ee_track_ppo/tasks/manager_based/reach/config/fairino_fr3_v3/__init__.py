# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Gym 환경 등록
#
# v2 계열(``Reach-FR3-PPO-v2*``)은 손대지 않는다. v2f 는 실기 검증이 끝난 정책이라
# 재현 가능해야 하고, v3 의 A/B 기준선이기도 하다.
##

gym.register(
    id="Reach-FR3-PPO-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3PPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3PPORunnerCfg",
    },
)

##
# v3b: v3 가 드러낸 위치/자세 보상 불균형을 잡은 갈래 (B3안).
# 중력·상자·DR·지연은 v3 와 완전히 동일하므로 v3 대비 단일 변수다.
##

gym.register(
    id="Reach-FR3-PPO-v3b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3bPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v3b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3bPPORunnerCfg",
    },
)

##
# v3c: v3b 에 pitch 자유화(pi +- 0.5)를 더한 마지막 단.
# 커맨드의 자세 범위 두 줄만 다르므로 v3b 대비 단일 변수다.
##

gym.register(
    id="Reach-FR3-PPO-v3c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3cPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v3c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3cEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3cPPORunnerCfg",
    },
)

##
# v3d: 실기 컨트롤러의 실제 j3 한계(±150도)를 반영하고 상자를 다시 잡은 이관 후보.
# 에셋(URDF/USD) 변경은 모든 태스크에 적용되므로 여기서는 상자만 지정한다.
##

gym.register(
    id="Reach-FR3-PPO-v3d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3dEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3dPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v3d",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_comp_env_cfg:FairinoFR3V3dEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3V3dPPORunnerCfg",
    },
)
