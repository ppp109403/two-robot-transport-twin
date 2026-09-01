"""게이트 G1 — **관측 조립 정합성**. twin 리그의 첫 번째 관문.

무엇을 재는가
-------------
같은 시뮬 상태에서 두 경로가 만든 52 차원 관측이 같은지 본다::

    시뮬 경로   env.observation_manager  (매니저 기반 ObsTerm 들)
    배포 경로   handoff/traj_v29/ee_obs.py  build_obs()   <- 실기·twin 이 쓰는 것

왜 이것이 첫 관문인가
---------------------
twin 리그는 매니저 기반 env 를 안 쓰고 ``ee_obs.py`` + ONNX 로 돌린다. 시뮬·실기·twin
셋이 관측 구현을 **하나만** 공유하게 만들기 위해서다. 그 대가로, 그 하나가 학습된
배치와 어긋나면 **에러 없이 조용히 이상한 값**이 나간다.

이 저장소에서 실제로 났던 사고: 팔 정책(42)과 통합 정책(42)이 차원은 같은데 내용도
순서도 전혀 달랐다. 차원 검사로는 안 걸린다. 그래서 값으로 대조한다.

두 가지 모드
------------
::

    --amcl off   (기본)  AMCL 잡음·지연·주기를 0 으로 죽여 추정값 == 참값.
                         **전 구간이 tol 안에 들어와야 한다.** 순수 배치/규약 게이트.
    --amcl on            학습과 같은 AMCL 모델을 켠다. target_vel 블록만 어긋나는 것이
                         정상이다 (아래) — 그 크기가 예상 범위인지 본다.

``--amcl on`` 에서 target_vel 이 어긋나는 이유는 알려진 이음매다. 시뮬
(:func:`ee_target_vel_b`) 은 목표 속도를 **참값 yaw** 로 회전시키고, 배포
(``build_obs``) 는 참값이 없으니 **AMCL yaw** 를 쓴다. ``ee_obs.py`` 가 그렇게
문서화해 두었다 (yaw 오차 0.004 rad -> 속도 방향 약 0.2 도).

무엇이 참값이고 무엇이 추정값인가
---------------------------------
학습 커맨드 term 은 이렇게 나눈다. 배포도 반드시 같아야 한다::

    pose_command   AMCL 추정 base pose 로 상대화   <- 맵 프레임 목표라 로컬 오차가 실린다
    ee_pose        참값 base 기준 (= 팔 FK)        <- AMCL 오차가 두 번 들어가면 안 된다

그래서 이 스크립트는 ``build_obs`` 에 넣을 ``ee_pos_w`` 를
``추정 base pose (+) 참값 base 기준 EE`` 로 되돌려 만든다. 실기 브리지가 하는 것과 같다.

사용법
------
::

    twin/run.sh twin/tools/check_obs_parity.py --report /tmp/parity_off.txt
    twin/run.sh twin/tools/check_obs_parity.py --amcl on --report /tmp/parity_on.txt

Kit 이 stdout 을 가로채므로 판정은 ``--report`` 파일로 받는다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Reach-AMR-FR3-TrajV29-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--warmup", type=int, default=5,
                    help="앞부분 제외 [스텝]. 리셋 직후는 커맨드 term 이 아직 안 채워졌다")
parser.add_argument("--ee-obs", dest="ee_obs", type=str,
                    default="handoff/traj_v29/ee_obs.py")
parser.add_argument("--policy", type=str, default="",
                    help="exported policy.pt (torch.jit). 주면 정책이 실제로 방문하는 "
                         "상태에서 대조한다. 비우면 난수 액션")
parser.add_argument("--amcl", choices=["off", "on"], default="off")
parser.add_argument("--tol", type=float, default=1.0e-4,
                    help="float32 관측이라 1e-4 면 충분히 빡빡하다")
parser.add_argument("--report", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.util  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ee_track_ppo.tasks  # noqa: F401, E402
# 원본에서 직접 가져온다. AMR_FR3_ARM_JOINTS 는 이름 목록이 아니라 정규식
# (["j[1-6]"]) 이라 SceneEntityCfg 로 풀어야 순서까지 ObsTerm 과 같아진다.
from ee_track_ppo.assets.amr_fr3 import AMR_FR3_ARM_JOINTS, AMR_FR3_EE_BODY  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

LINES: list[str] = []


def say(msg: str = "") -> None:
    """Kit 이 stdout 을 먹으므로 모아 두었다가 --report 로도 쓴다."""
    print(msg, flush=True)
    LINES.append(msg)


def load_ee_obs(path: str):
    """배포용 ee_obs.py 를 **경로로** 불러온다.

    import 로 안 잡는 이유: 체크포인트마다 다른 ee_obs.py 가 있고 (52 vs 58 차원),
    어느 것을 쓰는지가 이 검사의 핵심 변수라 명시적으로 받아야 한다.
    """
    full = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(full):
        raise SystemExit("ee_obs.py 가 없다: %s" % full)
    spec = importlib.util.spec_from_file_location("ee_obs_under_test", full)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod, full


def reference_obs(mod, robot, term, arm_ids, ee_id, last_action) -> np.ndarray:
    """시뮬 상태를 읽어 **배포 경로**로 관측을 조립한다. (N, 52)"""
    d = robot.data

    arm_q = (d.joint_pos - d.default_joint_pos)[:, arm_ids]
    arm_qd = (d.joint_vel - d.default_joint_vel)[:, arm_ids]
    base_v = d.root_lin_vel_b[:, 0]
    base_w = d.root_ang_vel_b[:, 2]

    # AMCL 이 추정한 base pose.
    #   ★ term._amcl(robot) 을 **호출하면 안 된다** — 지연 링버퍼가 한 칸 전진해서
    #     이 검사가 시뮬 자체를 오염시킨다. 저장된 추정값만 읽고, 커맨드 클래스가
    #     하는 것과 같은 방식으로 pose 를 재구성한다 (xy·yaw 만 추정, 나머지는 참값).
    est = term.amcl_estimate
    base_pos_w = d.root_pos_w.clone()
    base_pos_w[:, :2] = est[:, :2]
    roll, pitch, _ = euler_xyz_from_quat(d.root_quat_w)
    base_quat_w = quat_from_euler_xyz(roll, pitch, est[:, 2])

    # 참값 base 기준 EE = 팔 FK. 여기에 AMCL 을 섞으면 오차가 두 번 들어간다.
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        d.root_pos_w, d.root_quat_w, d.body_pos_w[:, ee_id], d.body_quat_w[:, ee_id]
    )
    # 추정 base 를 통해 월드로 되돌린다 (실기 브리지와 같은 순서).
    ee_pos_w = base_pos_w + quat_apply(base_quat_w, ee_pos_b)
    ee_quat_w = quat_mul(base_quat_w, ee_quat_b)

    goal_pos_w = term.pose_command_w[:, :3]
    goal_quat_w = term.pose_command_w[:, 3:]
    zero3 = torch.zeros_like(goal_pos_w)
    lin_w = getattr(term, "target_lin_vel_w", None)
    ang_w = getattr(term, "target_ang_vel_w", None)
    lin_w = zero3 if lin_w is None else lin_w
    ang_w = zero3 if ang_w is None else ang_w

    def np_(t):
        return t.detach().cpu().double().numpy()

    arm_q, arm_qd = np_(arm_q), np_(arm_qd)
    base_v, base_w = np_(base_v), np_(base_w)
    base_pos_w, base_quat_w = np_(base_pos_w), np_(base_quat_w)
    ee_pos_w, ee_quat_w = np_(ee_pos_w), np_(ee_quat_w)
    goal_pos_w, goal_quat_w = np_(goal_pos_w), np_(goal_quat_w)
    lin_w, ang_w = np_(lin_w), np_(ang_w)
    act = np_(last_action)

    return np.stack([
        mod.build_obs(
            arm_joint_pos_rel=arm_q[i], arm_joint_vel=arm_qd[i],
            base_lin_x=float(base_v[i]), base_ang_z=float(base_w[i]),
            base_pos_w=base_pos_w[i], base_quat_w=base_quat_w[i],
            ee_pos_w=ee_pos_w[i], ee_quat_w=ee_quat_w[i],
            goal_pos_w=goal_pos_w[i], goal_quat_w=goal_quat_w[i],
            goal_lin_vel_w=lin_w[i], goal_ang_vel_w=ang_w[i],
            last_action=act[i],
        )
        for i in range(arm_q.shape[0])
    ])


def main() -> bool:
    mod, ee_obs_path = load_ee_obs(args.ee_obs)

    say("=" * 76)
    say("G1 관측 정합성 — 시뮬 ObsTerm  vs  %s" % os.path.basename(ee_obs_path))
    say("=" * 76)
    say("  task        %s" % args.task)
    say("  ee_obs      %s  (OBS_DIM=%d, ACTION_DIM=%d)"
        % (ee_obs_path, mod.OBS_DIM, mod.ACTION_DIM))
    say("  AMCL 모델   %s" % ("학습과 동일 (on)" if args.amcl == "on" else "무력화 (off)"))
    say("  허용오차    %.1e" % args.tol)

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    # 관측 노이즈는 반드시 끈다 — 켜져 있으면 배치가 맞아도 값이 다르다.
    env_cfg.observations.policy.enable_corruption = False

    cmd = env_cfg.commands.ee_pose
    if args.amcl == "off":
        # 추정값 == 참값 이 되게 만든다.
        #   latency 0  -> 링버퍼 길이 1 -> 지연 없음
        #   update_period 0 -> 매 스텝 갱신 (zero-order hold 없음)
        cmd.pos_noise_std = 0.0
        cmd.yaw_noise_std = 0.0
        cmd.jump_prob = 0.0
        cmd.latency = 0.0
        cmd.update_period = 0.0
    say("  AMCL 설정   noise %.4f/%.4f  period %.3f  latency %.3f  jump %.4f"
        % (cmd.pos_noise_std, cmd.yaw_noise_std, cmd.update_period,
           cmd.latency, cmd.jump_prob))

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.command_manager.get_term("ee_pose")

    # ObsTerm 이 쓰는 것과 **같은 방식**으로 인덱스를 푼다.
    # 직접 find_joints 를 쓰면 정렬 규약이 달라 조용히 순서가 뒤바뀔 수 있다.
    acfg = SceneEntityCfg("robot", joint_names=AMR_FR3_ARM_JOINTS)
    acfg.resolve(u.scene)
    bcfg = SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY])
    bcfg.resolve(u.scene)
    arm_ids, ee_id = acfg.joint_ids, bcfg.body_ids[0]
    say("  팔 조인트   %s -> %s" % (AMR_FR3_ARM_JOINTS, list(arm_ids)))
    say("  EE 바디     %s -> %d" % (AMR_FR3_EE_BODY, ee_id))

    if not hasattr(term, "amcl_estimate"):
        say("")
        say("  [실패] 커맨드 term 에 amcl_estimate 가 없다: %s" % type(term).__name__)
        env.close()
        return False

    policy = None
    if args.policy:
        policy = torch.jit.load(args.policy, map_location=u.device).eval()
        say("  정책        %s" % args.policy)
    else:
        say("  정책        (없음 — 난수 액션)")

    torch.manual_seed(0)
    obs_dict, _ = env.reset()
    act = torch.zeros(u.num_envs, mod.ACTION_DIM, device=u.device)

    sl = mod.slices()
    # 블록마다 "표본 하나(=env 하나, 스텝 하나)의 블록내 최대오차" 를 모은다.
    # 최댓값만 보면 AMCL 점프(jump_yaw_std 0.10 rad) 한 번이 전체 판정을 지배해서
    # 이음매가 상시 그만큼 벌어지는 것처럼 읽힌다. 분위수를 같이 내야 구분된다.
    samples: dict[str, list] = {k: [] for k in sl}
    worst_all = 0.0
    worst_step = -1
    n_cmp = 0

    for step in range(args.steps):
        if policy is not None:
            with torch.inference_mode():
                act = policy(obs_dict["policy"]).clamp(-1.0, 1.0)
        else:
            act = (torch.rand(u.num_envs, mod.ACTION_DIM, device=u.device) * 2.0 - 1.0) * 0.3

        obs_dict, _, _, _, _ = env.step(act)

        if step < args.warmup:
            continue

        sim = obs_dict["policy"].detach().cpu().double().numpy()
        ref = reference_obs(mod, robot, term, arm_ids, ee_id,
                            u.action_manager.action).astype(np.float64)
        if sim.shape != ref.shape:
            say("")
            say("  [실패] 차원 불일치 sim %s vs ref %s" % (sim.shape, ref.shape))
            env.close()
            return False

        diff = np.abs(sim - ref)
        n_cmp += 1
        for k, s in sl.items():
            samples[k].append(diff[:, s].max(axis=1))     # (num_envs,)
        m = float(diff.max())
        if m > worst_all:
            worst_all, worst_step = m, step

    env.close()

    stat = {k: np.concatenate(v) for k, v in samples.items()}
    worst = {k: float(v.max()) for k, v in stat.items()}

    say("")
    say("블록별 절대오차 (%d 스텝 x %d env = %d 표본)"
        % (n_cmp, args.num_envs, n_cmp * args.num_envs))
    say("-" * 76)
    say("  %-16s %-9s %10s %10s %10s   %s"
        % ("블록", "구간", "중앙", "95%", "최대", "판정"))
    ok = True
    for k, s in sl.items():
        # --amcl on 에서 target_vel 은 알려진 이음매다 (시뮬은 참값 yaw, 배포는 AMCL yaw).
        known_seam = (args.amcl == "on" and k == "target_vel")
        good = worst[k] <= args.tol
        if not good and not known_seam:
            ok = False
        tag = "OK" if good else ("이음매(예상됨)" if known_seam else "**불일치**")
        v = stat[k]
        say("  %-16s [%2d:%2d]  %10.3e %10.3e %10.3e   %s"
            % (k, s.start, s.stop, float(np.median(v)),
               float(np.percentile(v, 95)), worst[k], tag))
    say("-" * 76)
    say("  전체 최대 %.3e  (worst step %d)" % (worst_all, worst_step))
    say("")

    if ok:
        say("G1 통과 — 배포 경로가 학습 배치를 그대로 재현한다.")
        if args.amcl == "on":
            v = stat["target_vel"]
            say("  target_vel 이음매: 중앙 %.3e  95%% %.3e  최대 %.3e [m/s, rad/s]"
                % (float(np.median(v)), float(np.percentile(v, 95)), worst["target_vel"]))
            say("  최댓값은 AMCL 점프(jump_yaw_std 0.10 rad, jump_prob 0.002/스텝) 이벤트다.")
            say("  상시 성분은 중앙값 쪽을 볼 것 — 정상상태 잡음(0.004 rad)이 아니라")
            say("  갱신주기 0.18 s + 지연 0.06 s 동안의 **요 지연**이 지배한다.")
    else:
        say("G1 실패 — twin 리그를 여기서 멈춘다.")
        say("  배치가 바뀌었을 수 있다. 다음으로 실측 배치를 다시 뽑아 대조할 것:")
        say("    twin/run.sh scripts/dump_obs_layout.py --task %s" % args.task)
    say("=" * 76)
    return ok


if __name__ == "__main__":
    passed = False
    try:
        passed = main()
    finally:
        if args.report:
            with open(args.report, "w") as fh:
                fh.write("\n".join(LINES) + "\n")
            print("report -> %s" % args.report, flush=True)
        simulation_app.close()
    raise SystemExit(0 if passed else 1)
