# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Gym 환경 등록
##

gym.register(
    id="Reach-AMR-FR3-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mobile_reach_env_cfg:AmrFr3ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-PPO-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mobile_reach_env_cfg:AmrFr3ReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

##
# 협력 운송 — 움직이는 EE 목표 궤적 추종 (STAGE4)
##

gym.register(
    id="Reach-AMR-FR3-Traj-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-Traj-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajDelay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajDelayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajDelay-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajDelayEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV5-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV5EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV6EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV7-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV7EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV8-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV8EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV9-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV9EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV10-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV10EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV11-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV11-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV11EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV12-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV12EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV13-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV13EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV13-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV13EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV14-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV14EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV14-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV14EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV15-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV15-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV15EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV16-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV16EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV16-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV16EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV17-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV17EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV17-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV17EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV18EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV18-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV18EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV19-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV19EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV19-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV19EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV20EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV20EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV21-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV21EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV21-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV21EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV22-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV22EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV22-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV22EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV23-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV23EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV23-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV23EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV23-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV23EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV23-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV23EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV24-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV24EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV24-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV24EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV25-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV25EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV25-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV25EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV26-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV26EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV26-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV26EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV27-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV27EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV27-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV27EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV28-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV28EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV28-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV28EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV29-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV29EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV29-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV29EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV30-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV30EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Reach-AMR-FR3-TrajV30-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.traj_env_cfg:AmrFr3TrajV30EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AmrFr3ReachPPORunnerCfg",
    },
)
