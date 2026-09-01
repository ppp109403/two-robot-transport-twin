# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Gym 환경 등록
#
# v1(``Reach-FR3-PPO-v0``) 은 손대지 않고 그대로 둔다.
# 실기 검증이 끝난 정책이라 재현 가능해야 하고, A/B 비교의 기준선이기도 하다.
##

gym.register(
    id="Reach-FR3-PPO-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_env_cfg:FairinoFR3Sim2RealEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3Sim2RealPPORunnerCfg",
    },
)

gym.register(
    id="Reach-FR3-PPO-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_env_cfg:FairinoFR3Sim2RealEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FairinoFR3Sim2RealPPORunnerCfg",
    },
)

##
# 보상 스윕 v2a ~ v2f (액추에이터 설정은 v2 와 동일, 보상만 다름)
# v2f 는 a~e 결과를 보고 사후에 추가한 갈래다 (sweep_env_cfg.py 참고)
##

for _letter in ("a", "b", "c", "d", "e", "f"):
    _up = _letter.upper()
    gym.register(
        id=f"Reach-FR3-PPO-v2{_letter}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sweep_env_cfg:FR3Sweep{_up}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3Sweep{_up}PPORunnerCfg",
        },
    )
    gym.register(
        id=f"Reach-FR3-PPO-Play-v2{_letter}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sweep_env_cfg:FR3Sweep{_up}EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FR3Sweep{_up}PPORunnerCfg",
        },
    )

del _letter, _up
