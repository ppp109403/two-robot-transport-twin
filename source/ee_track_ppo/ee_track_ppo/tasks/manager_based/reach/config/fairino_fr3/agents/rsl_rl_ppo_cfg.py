# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fairino FR3 reach 용 PPO(rsl_rl) 하이퍼파라미터.

franka v5 설정을 그대로 복사했다. 로봇만 바꾼 비교 실험이므로
알고리즘 하이퍼파라미터는 변수 통제를 위해 건드리지 않는다.
(``experiment_name`` 만 분리해서 로그가 섞이지 않게 한다)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FairinoFR3ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 한 iteration 당 env 별 rollout 길이. num_envs * num_steps_per_env = batch size
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    # logs/rsl_rl/<experiment_name>/<timestamp>/ 에 체크포인트가 쌓인다
    # franka_reach 와 분리해서 두 로봇의 학습 곡선을 나란히 비교할 수 있게 한다.
    experiment_name = "fr3_reach"
    run_name = ""

    # franka v5 와 동일한 네트워크. 관측 차원만 45 -> 43 으로 살짝 줄어든다.
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
