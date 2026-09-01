"""게이트 G3 — **계획 -> EE 목표 변환 정합성**.

무엇을 재는가
-------------
같은 계획을 놓고 두 경로가 만든 목표가 같은지 본다::

    학습 경로   MovingPoseWorldAmclCommand._update_from_plan   (매니저 기반 커맨드 term)
    twin 경로   twin/plan/plan_to_target.py  TargetStream

G1 이 "상태 -> 관측" 을 맞췄다면 이것은 그 앞단인 "계획 -> 목표" 다. 둘을 다 맞춰야
twin 이 학습과 같은 문제를 푸는 것이 된다.

여기서 틀리면 어떻게 되나
-------------------------
전부 **에러 없이** 틀린다::

    상대화 기준을 궤적 첫 점으로 잡으면   t0 랜덤화가 있는 판에서만 어긋난다
    요 보정을 빠뜨리면                     막대가 90 도 돌아간 채로 완주한다
    피드포워드를 해석적 미분으로 바꾸면    학습이 준 계단형 차분과 미세하게 다르다

각속도 채널은 판정 대상이 아니다
--------------------------------
이 게이트가 처음 찾아낸 것(2026-08-12): 학습 env 는 계획 env 에도 **기하 합성 경로의
각속도 잔여값**을 남긴다 (실측 중앙 0.215 rad/s). 계획과 무관한 값이라 재현 대상이
아니고, 실기 ``plan_adapter`` 는 또 다른 값(계획 요 변화율)을 넣는다. 그래서 pose 와
선속도 xy 만 판정하고 나머지는 크기만 관찰해서 남긴다.

쿼터니언은 직접 비교하지 않는다
-------------------------------
목표 자세는 pitch=pi 라 w=0 인 특이면 위에 있고, 부호(q vs -q)가 부동소수점에
좌우된다. 정책은 rot6d 로 받아 부호에 불변이므로 **rot6d 로 비교**한다.
쿼터니언으로 비교하면 실제로는 같은 회전인데 실패로 뜬다.

요 보정 값에 대하여
-------------------
이 게이트는 **env 가 쓰는 전역 PLAN_YAW_OFFSET 을 그대로** 넘겨 비교한다. 두 경로가
같은 규약을 쓰는지를 보는 검사이기 때문이다. 운용 코드는 이 상수를 import 하지 않고
``twin/configs/policy_*.yaml`` 에서 읽는다 (판마다 값이 다르다).

사용법
------
::

    twin/run.sh twin/tools/check_plan_target.py --report /tmp/g3.txt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Reach-AMR-FR3-TrajV29-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--dataset", type=str, default="data/plans.npz")
parser.add_argument("--tol-pos", dest="tol_pos", type=float, default=1.0e-5)
parser.add_argument("--tol-rot", dest="tol_rot", type=float, default=1.0e-5)
parser.add_argument("--tol-vel", dest="tol_vel", type=float, default=None,
                    help="선속도 허용치. 기본은 위치 오차에서 유도한다 "
                         "(2*max|dp|/step_dt) — 차분이 float32 위치오차를 증폭하므로")
parser.add_argument("--ff-mode", dest="ff_mode", choices=["planner", "zero"],
                    default="planner", help="속도 피드포워드 규약 (plan_to_target 참고)")
parser.add_argument("--report", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ee_track_ppo.tasks  # noqa: F401, E402
from ee_track_ppo.tasks.manager_based.reach.mdp.commands import PLAN_YAW_OFFSET  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plan.plan_to_target import PlanTargetCfg, TargetStream  # noqa: E402

LINES: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    LINES.append(msg)


def rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 6). 부호 불변이라 특이면에서도 안전하다."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    col0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    col1 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)
    return np.concatenate([col0, col1], axis=-1)


def make_streams(term, origins, step_dt, envs=None):
    """현재 term 상태에서 env 별 TargetStream 을 만든다."""
    idx = term._ds_idx.detach().cpu().numpy()
    t0 = term._ds_t0.detach().cpu().numpy()
    scale = term._ds_scale.detach().cpu().numpy()
    gz = term._ds_goal_z.detach().cpu().numpy()
    b0 = term._ds_b0.detach().cpu().numpy()
    ee_all = term._ds_ee.detach().cpu().numpy()
    base_all = term._ds_base.detach().cpu().numpy()

    out = {}
    for i in (range(len(idx)) if envs is None else envs):
        cfg = PlanTargetCfg(yaw_offset=float(PLAN_YAW_OFFSET), goal_z=float(gz[i]),
                            step_dt=step_dt, time_scale=float(scale[i]), t0=float(t0[i]),
                            ff_mode=args.ff_mode)
        out[i] = TargetStream(ee_all[idx[i]], base_all[idx[i]], cfg,
                              b_start=b0[i], origin=origins[i])
    return out


def main() -> bool:
    say("=" * 76)
    say("G3 계획 -> EE 목표 변환 정합성")
    say("=" * 76)
    say("  task          %s" % args.task)
    say("  yaw offset    %+.6f rad (%+.1f deg)  <- env 전역값을 그대로 사용"
        % (PLAN_YAW_OFFSET, np.degrees(PLAN_YAW_OFFSET)))

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    c = env_cfg.commands.ee_pose
    # 전부 데이터셋 경로로 몰아야 비교 대상이 생긴다 (기하 합성 env 는 계획이 없다).
    c.plan_mix = 1.0
    c.plan_dataset = args.dataset
    say("  dataset       %s  (plan_mix=1.0)" % args.dataset)

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    term = u.command_manager.get_term("ee_pose")
    step_dt = float(u.step_dt)
    say("  step_dt       %.6f s (%.1f Hz)" % (step_dt, 1.0 / step_dt))

    need = ("_ds_idx", "_ds_t0", "_ds_scale", "_ds_goal_z", "_ds_b0", "_ds_ee",
            "_ds_base", "_ep_t", "_use_ds")
    missing = [a for a in need if not hasattr(term, a)]
    if missing:
        say("")
        say("  [실패] 커맨드 term(%s)에 없는 속성: %s" % (type(term).__name__, missing))
        env.close()
        return False

    env.reset()
    origins = u.scene.env_origins.detach().cpu().numpy()
    streams = make_streams(term, origins, step_dt)
    prev_ep_t = term._ep_t.detach().cpu().numpy().copy()

    act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    stat = {k: [] for k in ("pos", "rot", "lin_xy",
                            "train_lin_z", "train_ang", "twin_ang")}
    n_cmp = 0
    n_resync = 0

    for _ in range(args.steps):
        # 두 경로를 같은 횟수만큼 전진시킨다. 둘 다 리셋 직후 _ep_t = 0 에서 시작하고
        # 한 번에 step_dt 씩 더하므로 호출 순서는 상관없다 (시계는 아래에서 확인한다).
        mine = {i: s.step() for i, s in streams.items()}
        env.step(act)

        ep_t = term._ep_t.detach().cpu().numpy()
        use_ds = term._use_ds.detach().cpu().numpy()

        # 에피소드가 리셋된 env 는 새 계획을 뽑는다. 그 env 만 스트림을 다시 만들고
        # 이번 스텝 비교에서 뺀다 (리셋 직후는 아직 목표가 안 채워졌다).
        reset = set(np.where(ep_t < prev_ep_t - 1e-9)[0].tolist())
        if reset:
            streams.update(make_streams(term, origins, step_dt, envs=sorted(reset)))
            n_resync += len(reset)
        prev_ep_t = ep_t.copy()

        live = [i for i in range(u.num_envs) if use_ds[i] and i not in reset]
        if not live:
            continue

        # 두 경로의 시계가 같은지 먼저 본다. 어긋나면 아래 비교는 의미가 없다.
        drift = max(abs(streams[i]._ep_t - float(ep_t[i])) for i in live)
        if drift > step_dt * 0.5:
            say("")
            say("  [실패] 시계 어긋남 %.4f s — 두 경로가 다른 시각을 보고 있다" % drift)
            env.close()
            return False

        env_pos = term.pose_command_w[:, :3].detach().cpu().numpy()
        env_quat = term.pose_command_w[:, 3:].detach().cpu().numpy()
        env_lin = term.target_lin_vel_w.detach().cpu().numpy()
        env_ang = getattr(term, "target_ang_vel_w", None)
        env_ang = (np.zeros_like(env_lin) if env_ang is None
                   else env_ang.detach().cpu().numpy())

        for i in live:
            m = mine[i]
            stat["pos"].append(float(np.max(np.abs(env_pos[i] - m["pos_w"]))))
            stat["rot"].append(float(np.max(np.abs(
                rot6d(env_quat[i]) - rot6d(m["quat_w"])))))
            # 계획 경로가 **실제로 정의하는** 것은 xy 뿐이다.
            stat["lin_xy"].append(float(np.max(np.abs(env_lin[i][:2] - m["lin_vel_w"][:2]))))
            # 아래 둘은 판정이 아니라 관찰이다 — 학습 env 가 계획 env 에 남겨 둔
            # 기하 합성 경로의 잔여값 크기. 자세한 것은 plan_to_target 모듈 docstring.
            stat["train_lin_z"].append(abs(float(env_lin[i][2])))
            stat["train_ang"].append(float(np.linalg.norm(env_ang[i])))
            stat["twin_ang"].append(float(np.linalg.norm(m["ang_vel_w"])))
        n_cmp += len(live)

    env.close()

    if n_cmp == 0:
        say("")
        say("  [실패] 비교된 표본이 없다 — plan_mix / 데이터셋 설정을 확인할 것")
        return False

    say("")
    say("비교 %d 표본 (%d 스텝, env %d, 재동기 %d 회)"
        % (n_cmp, args.steps, args.num_envs, n_resync))
    say("-" * 76)
    say("  %-26s %10s %10s %10s   %s" % ("판정 항목", "중앙", "95%", "최대", ""))
    # 선속도는 위치의 **차분**이라 위치 불일치가 1/step_dt 배로 증폭된다.
    # 따라서 허용치를 임의 상수로 두면 안 되고 오차 전파로 유도해야 한다:
    #   |dv| <= (|dp_k| + |dp_{k-1}|) / step_dt  <=  2 * max|dp| / step_dt
    # env 는 float32 라 위치 자체가 1e-6 m 수준에서만 일치하고, 30 Hz 에서 그것이
    # 곧바로 1e-4 m/s 가 된다. 이 바닥을 모르고 상수를 쓰면 정상인데 실패로 뜬다.
    pos_max = float(np.asarray(stat["pos"]).max())
    vel_floor = 2.0 * pos_max / step_dt
    tol_vel = args.tol_vel if args.tol_vel is not None else vel_floor
    limits = {
        "pos": ("목표 위치 [m]", args.tol_pos),
        "rot": ("목표 자세 rot6d", args.tol_rot),
        "lin_xy": ("선속도 xy [m/s]", tol_vel),
    }
    ok = True
    for key, (label, tol) in limits.items():
        v = np.asarray(stat[key])
        worst = float(v.max())
        good = worst <= tol
        ok &= good
        say("  %-26s %10.3e %10.3e %10.3e   %s"
            % (label, float(np.median(v)), float(np.percentile(v, 95)), worst,
               "OK" if good else "**불일치** (허용 %.1e)" % tol))
    say("-" * 76)
    say("  선속도 허용치 %.3e = 2 x 위치최대오차 %.3e / step_dt %.4f"
        % (tol_vel, pos_max, step_dt))
    say("")
    say("관찰 항목 — 판정 아님. 계획 경로가 정의하지 않는 속도 채널의 크기")
    say("-" * 76)
    for key, label in (("train_lin_z", "학습 env 의 lin z 잔여값 [m/s]"),
                       ("train_ang", "학습 env 의 ang 잔여값 [rad/s]"),
                       ("twin_ang", "twin(ff_mode=%s) 의 ang [rad/s]" % args.ff_mode)):
        v = np.asarray(stat[key])
        say("  %-30s %10.3e %10.3e %10.3e"
            % (label, float(np.median(v)), float(np.percentile(v, 95)), float(v.max())))
    say("-" * 76)
    say("  학습 env 는 _update_command 가 기하 합성 경로를 전체 env 에 먼저 돌린 뒤")
    say("  _update_from_plan 이 pose 와 lin xy 만 덮어쓴다. 그래서 계획 env 의 ang 과")
    say("  lin z 는 **계획과 무관한 잔여값**이다. 실기 plan_adapter 는 여기에 계획에서")
    say("  유도한 요 변화율을 넣는다 — 학습·실기가 원래부터 다른 채널이다.")
    say("  twin 은 실기를 따른다 (ff_mode=planner). 민감도는 Phase 1 에서 zero 와 비교.")
    say("")
    if ok:
        say("G3 통과 — 계획 -> 목표 pose 변환이 학습과 같다.")
        say("  상대화 기준(t0 시점 차체), 요 보정, 피치 pi, 목표 높이, 차분형 선속도 xy")
        say("  다섯 규약이 모두 재현됐다.")
    else:
        say("G3 실패 — twin 리그를 여기서 멈춘다.")
        say("  plan_to_target.TargetStream 을 _update_from_plan 과 다시 대조할 것:")
        say("    source/.../reach/mdp/commands.py  MovingPoseWorldAmclCommand._update_from_plan")
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
        simulation_app.close()
    raise SystemExit(0 if passed else 1)
