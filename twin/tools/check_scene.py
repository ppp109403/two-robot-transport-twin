"""Phase 1 씬 스모크 — 로봇 두 대를 계획 t=0 자세에 세우고 **가만히 둔다**.

무엇을 보는가
-------------
::

    [1] 씬이 뜨는가            맵 USD, 로봇 두 대, 콜라이더
    [2] 가라앉지 않는가        영액션으로 N 스텝. 높이·xy 표류
    [3] 시작 자세가 맞는가     차체를 계획 base_A/B[0] 에 놓았을 때
                               tool0 이 계획 EE 와 얼마나 떨어져 있는가
    [4] 두 대 간격             계획값 대비. robot_robot_min 0.90 위반 여부
    [5] 맵 여유                차체 디스크가 벽에서 얼마나 떨어져 있는가

[3] 이 이 검사의 핵심이다
-------------------------
플래너는 ``reach_min..reach_max`` 로 "팔이 닿을 수 있다" 만 보장한다. 그런데 시뮬은
**기본자세**(``DEFAULT_ARM_POS``)에서 출발하므로, 그 자세의 tool0 이 계획 EE 에서
멀면 에피소드가 큰 초기 오차로 시작한다. 그 오차는 정책의 추종 실패가 아니라
**출발 조건**인데, 로그만 보면 구분이 안 된다.

이 값이 크면 Phase 1 은 t=0 에 정착 구간(settle)을 넣어야 한다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=str, default="twin/data/plans/margin_005")
parser.add_argument("--obj-len", dest="obj_len", type=float, default=2.0)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--settle", type=int, default=90,
                    help="기준을 잡기 전 가라앉히는 스텝 (INIT_HEIGHT 가 평형보다 낮다)")
parser.add_argument("--no-map", action="store_true", help="빈 평지와 비교")
parser.add_argument("--report", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ee_track_ppo.assets.amr_fr3 import AMR_FR3_EE_BODY  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

_TWIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _TWIN)
sys.path.insert(0, os.path.join(_TWIN, "world"))
from plan.plan_sampler import Plan  # noqa: E402
from world.scene_builder import (  # noqa: E402
    ASSET, DEFAULT_MAP_USD, SIDES, action_layout, make_twin_cfg)

LINES: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    LINES.append(m)


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """wxyz (Isaac Lab 규약)."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def place(env, plan: Plan) -> dict:
    """두 로봇을 계획 t=0 의 차체 자세에 놓고 팔은 기본자세로."""
    smp = plan.sample_step(0.0)
    for s in SIDES:
        robot = env.scene[ASSET[s]]
        z = float(robot.data.default_root_state[0, 2])
        xy = smp["base"][s]
        w, qx, qy, qz = yaw_to_quat(float(smp["base_yaw"][s]))
        root = robot.data.default_root_state.clone()
        root[:, 0], root[:, 1], root[:, 2] = float(xy[0]), float(xy[1]), z
        root[:, 3], root[:, 4], root[:, 5], root[:, 6] = w, qx, qy, qz
        root[:, 7:] = 0.0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                       torch.zeros_like(robot.data.default_joint_vel))
    env.sim.forward()
    return smp


def main() -> bool:
    plan_dir = args.plan
    if not os.path.isdir(plan_dir):
        say("계획 디렉토리가 없다: %s" % plan_dir)
        say("  먼저 만들 것: <casadi 있는 python> "
            "twin/tools/solve_plan.py --margins 0.05")
        return False
    plan = Plan.from_csv_dir(plan_dir, obj_len=args.obj_len)

    say("=" * 76)
    say("Phase 1 씬 스모크")
    say("=" * 76)
    say("  계획      %s   N=%d  dt=%.3f  %.1f s  obj_len=%.2f"
        % (plan_dir, plan.N, plan.dt, plan.duration, plan.obj_len))

    cfg = make_twin_cfg(map_usd=None if args.no_map else DEFAULT_MAP_USD)
    env = ManagerBasedEnv(cfg=cfg)
    say("  맵        %s" % ("없음 (빈 평지)" if args.no_map else "압출 USD"))
    say("  sim dt    %.6f  decimation %d  -> 제어 %.1f Hz"
        % (env.cfg.sim.dt, env.cfg.decimation, 1.0 / env.step_dt))
    lay = action_layout(env)
    say("  액션 배치 %s (총 %d)"
        % ({k: (v.start, v.stop) for k, v in lay.items()},
           env.action_manager.total_action_dim))

    env.reset()
    smp = place(env, plan)
    z_spawn = {s: float(env.scene[ASSET[s]].data.root_pos_w[0, 2]) for s in SIDES}

    # --- 정착 --------------------------------------------------------------
    # ArticulationCfg 의 INIT_HEIGHT 는 0.010 m 인데 구동륜 반경은 0.085 m 다.
    # 즉 스폰 자세는 평형보다 **아래**라, 놓자마자 PhysX 가 밀어 올린다. 그 과도를
    # 표류로 재면 멀쩡한 씬이 실패로 뜬다. 먼저 가라앉힌 뒤 기준을 잡는다.
    #
    # 실기에도 대응물이 있다 — 로봇은 이미 서 있고, 관제는 EXECUTE 전에 INITIAL 로
    # 시작 자세를 잡아 준다. 정착 없이 t=0 부터 재생하는 쪽이 오히려 비현실적이다.
    act0 = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.settle):
        env.step(act0)

    ee_id = {}
    for s in SIDES:
        c = SceneEntityCfg(ASSET[s], body_names=[AMR_FR3_EE_BODY])
        c.resolve(env.scene)
        ee_id[s] = c.body_ids[0]

    def snap():
        out = {}
        for s in SIDES:
            d = env.scene[ASSET[s]].data
            out[s] = {
                "base": d.root_pos_w[0, :3].cpu().numpy().copy(),
                "ee": d.body_pos_w[0, ee_id[s], :3].cpu().numpy().copy(),
                "yaw": float(d.heading_w[0]),
            }
        return out

    s0 = snap()

    # --- [3] 시작 자세 ---------------------------------------------------
    say("")
    say("[3] 계획 t=0 vs 실제 기본자세")
    say("-" * 76)
    say("  %-6s %-26s %-26s %10s" % ("로봇", "계획 EE (x, y)", "실제 tool0 (x, y, z)", "xy 오차"))
    start_err = {}
    for s in SIDES:
        p = smp["ee"][s]
        q = s0[s]["ee"]
        e = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        start_err[s] = e
        say("  %-6s (%8.3f, %8.3f)      (%7.3f, %7.3f, %5.3f)   %8.3f m"
            % (s.upper(), p[0], p[1], q[0], q[1], q[2], e))
    say("  계획 EE 높이는 2D 계획에 없다. 정책 목표 z 는 0.55 m (v29 계약)")
    say("  정착 후 tool0 높이 %.3f / %.3f m  -> 목표 0.55 까지 %+.3f / %+.3f m"
        % (s0["a"]["ee"][2], s0["b"]["ee"][2],
           0.55 - s0["a"]["ee"][2], 0.55 - s0["b"]["ee"][2]))
    say("  차체 높이  스폰 %.3f -> 정착 %.3f / %.3f m (INIT_HEIGHT 0.010, 바퀴 r 0.085)"
        % (z_spawn["a"], s0["a"]["base"][2], s0["b"]["base"][2]))

    # --- [4] 두 대 간격 --------------------------------------------------
    d_plan = float(np.linalg.norm(smp["base"]["b"] - smp["base"]["a"]))
    d_sim = float(np.linalg.norm(s0["b"]["base"][:2] - s0["a"]["base"][:2]))
    say("")
    say("[4] 차체 간격  계획 %.3f m   실제 %.3f m   (robot_robot_min 0.90)"
        % (d_plan, d_sim))

    # --- [2] 정지 안정성 -------------------------------------------------
    for _ in range(args.steps):
        env.step(act0)
    s1 = snap()
    say("")
    say("[2] 정착 %d 스텝 뒤, 다시 영액션 %d 스텝 (%.1f s) 동안의 표류"
        % (args.settle, args.steps, args.steps * env.step_dt))
    say("-" * 76)
    ok = True
    dt_win = args.steps * env.step_dt
    for s in SIDES:
        dz = float(s1[s]["base"][2] - s0[s]["base"][2])
        dxy = float(np.linalg.norm(s1[s]["base"][:2] - s0[s]["base"][:2]))
        dyaw = math.degrees((s1[s]["yaw"] - s0[s]["yaw"] + math.pi) % (2 * math.pi) - math.pi)
        dee = float(np.linalg.norm(s1[s]["ee"] - s0[s]["ee"]))
        # 요 표류가 진짜 문제다. EE 목표는 map 프레임이라 차체 요가 틀어지면
        # 그만큼 목표가 통째로 회전해 EE 오차로 직행한다 (팔 길이 ~0.5 m 기준
        # 1 deg 가 약 9 mm — v29 의 추종오차 중앙 9.5 mm 와 같은 자릿수다).
        bad = abs(dz) > 0.02 or dxy > 0.02 or abs(dyaw) > 1.0
        ok &= not bad
        say("  %-6s 높이 %+.4f m   xy %.4f m   요 %+.3f deg (%.3f deg/s)   tool0 %.4f m   %s"
            % (s.upper(), dz, dxy, dyaw, dyaw / dt_win, dee, "**표류**" if bad else "OK"))

    say("")
    say("=" * 76)
    env.close()
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
