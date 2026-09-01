# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 v4 사다리 - 팔 단독 작업의 마지막 사이클.

각 단계가 직전 대비 단일 변수다. 자세한 근거는 ``ladder_env_cfg.py`` 참고.
v3 계열은 손대지 않는다 (v3d 는 실기 검증이 끝난 정책이고 v4 의 A/B 기준선이다).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Reach-FR3-PPO-v4a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4aPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v4a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4aEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4aPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-v4b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4bPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v4b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4bPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-v4c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4cPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v4c",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4cEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4cPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-v4e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4eEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4ePPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v4e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4eEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4ePPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-v4f",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4fEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4fPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v4f",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4fEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4fPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-v4g",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4gEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4gPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v4g",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4gEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4gPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-v4h",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4hEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4hPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v4h",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ladder_env_cfg:FR3V4hEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3V4hPPORunnerCfg",
    },
)
