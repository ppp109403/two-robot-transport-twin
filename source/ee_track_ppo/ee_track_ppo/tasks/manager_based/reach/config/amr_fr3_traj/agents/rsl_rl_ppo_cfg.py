# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMR + FR3 모바일 매니퓰레이터 reach 용 PPO(rsl_rl) 하이퍼파라미터.

ridgeback 모바일 매니퓰레이터 값을 기준으로 두 가지만 바꿨다.

``gamma = 0.995``
    ridgeback 은 0.99 를 쓴다. 유효 시야가 1/(1-γ) = 100 스텝인데 이 태스크는
    정책이 25 Hz 라 100 스텝 = 4 초다. 목표 하나가 8 초 유지되므로 "8 초 뒤 도달"의
    가치가 현재 시점에서 거의 사라진다. 0.995 -> 200 스텝 = 8 초로 목표 주기와 맞춘다.
    (AMR 단독 주행 태스크에서 같은 이유로 0.995 를 써서 수렴을 확인했다)

``num_steps_per_env = 48``
    한 rollout 이 48 * 0.04 s = 1.92 초. 선회-접근-팔뻗기의 국면 전환이 한 배치 안에
    여러 개 담기도록 ridgeback(32)보다 길게 잡는다.

네트워크는 ridgeback 과 같은 [256,128,64] 를 쓴다.
관측 42 차원 (팔 관절 12 + 베이스 트위스트 2 + 목표 pose 7 + EE pose 7 + EE 오차 6 + 이전 액션 8),
액션 8 차원 (팔 6 + 베이스 트위스트 2) 으로 ridgeback 과 비슷한 규모다.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AmrFr3TrajPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 100
    experiment_name = "amr_fr3_traj"
    run_name = ""
    obs_groups = {}

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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
