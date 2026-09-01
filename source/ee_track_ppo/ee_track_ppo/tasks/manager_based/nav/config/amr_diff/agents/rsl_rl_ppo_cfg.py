# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""차동구동 AMR pose 도달용 PPO(rsl_rl) 하이퍼파라미터.

reach 태스크 값을 기준으로 주행 태스크의 성질만 반영했다.

``gamma = 0.995``
    reach 는 0.99 를 쓴다. 0.99 의 유효 시야는 1/(1-γ) = 100 스텝 = 4 초인데,
    이 태스크는 목표 하나가 8 초(200 스텝) 동안 유지된다. 0.99 로 두면
    "8 초 뒤 목표 도달"의 가치가 현재 시점에서 거의 사라져서, 정책이
    멀리 있는 목표를 향해 출발할 이유를 학습하기 어렵다.
    0.995 -> 시야 200 스텝 = 8 초로 목표 주기와 맞춘다.

``num_steps_per_env = 48``
    한 rollout 이 48 * 0.04 s = 1.92 초. 접근-정렬-유지의 국면 전환이
    한 배치 안에 여러 개 담기도록 reach(24~32)보다 길게 잡는다.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AmrDiffPoseNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 100
    experiment_name = "amr_diff_pose_nav"
    run_name = ""
    # 비워 두면 rsl_rl 이 "policy" 그룹만 쓰는 기본 동작으로 해석한다.
    # 비대칭 actor-critic (critic 에만 참값 pose 를 주는 구성)을 쓰려면
    # 관측에 privileged 그룹을 추가하고 {"critic": ["policy", "privileged"]} 로 지정한다.
    obs_groups = {}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # 관측 9차원 / 액션 2차원으로 작지만, 접근-정렬-유지의 국면별 정책이
        # 하나의 네트워크에 담겨야 하므로 폭을 충분히 준다.
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
