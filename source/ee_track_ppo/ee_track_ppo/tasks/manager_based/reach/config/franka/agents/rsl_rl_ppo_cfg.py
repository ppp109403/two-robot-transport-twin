# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka reach 용 PPO(rsl_rl) 하이퍼파라미터.

Isaac Lab 내장 ``Isaac-Reach-Franka-v0`` 에서 검증된 값을 그대로 가져왔다.
(source/isaaclab_tasks/.../manipulation/reach/config/franka/agents/rsl_rl_ppo_cfg.py)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 한 iteration 당 env 별 rollout 길이. num_envs * num_steps_per_env = batch size
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    # logs/rsl_rl/<experiment_name>/<timestamp>/ 에 체크포인트가 쌓인다
    experiment_name = "franka_reach"
    run_name = ""

    # [v4] 네트워크 확대: [64,64] -> [256,128,64]
    #   7축 팔의 6D pose 를 위치·자세 동시에 정밀 추종하기에는 [64,64] 가 작다.
    #   (내장 reach 가 이 크기로 충분했던 것은 자세 정확도를 사실상 포기했기 때문)
    #   관측도 32 -> 45 차원으로 늘었으므로 입력단 용량도 함께 키운다.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
