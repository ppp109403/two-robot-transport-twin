"""MPC 계획을 **정책 없이 고전 제어기로** 추종한다 — 계획 추종 가능성의 기준선.

::

    twin/run.sh twin/runner/run_classic.py --track-a npz:98 --track-b npz:110 \\
        --out twin/data/runs/classic_98_110 --report /tmp/classic.txt

왜 이걸 먼저 돌리는가
---------------------
"MPC 계획을 로봇이 따라갈 수 있는가" 와 "정책이 잘 학습됐는가" 는 다른 질문이다.
정책으로만 재생하면 둘이 섞여서, 추종이 나쁠 때 계획 탓인지 정책 탓인지 알 수 없다.

여기서는 차체를 유니사이클 궤적 추종기로, 팔을 미분 IK 로 돌린다. 플랜트(씬·액션
항·지연·램프·로컬라이제이션)는 정책 경로와 **완전히 같고 제어기만 다르다.**

읽는 법::

    고전 O / 정책 O    계획도 정책도 문제 없음
    고전 O / 정책 X    정책 문제.  학습을 손볼 차례
    고전 X / 정책 X    **계획 문제.**  정책을 아무리 학습해도 안 된다
    고전 X / 정책 O    정책이 IK 보다 낫다는 뜻 (드물지만 가능 — 차체를 같이 쓰므로)

비교 대상은 ``run_episode.py`` 의 같은 궤적 결과다. 지표 이름과 계산이 같다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--track-a", dest="track_a", default="npz:98")
parser.add_argument("--track-b", dest="track_b", default="npz:110")
parser.add_argument("--policy-cfg", dest="policy_cfg", default="twin/configs/policy_v29.yaml",
                    help="여기서 읽는 것은 target 블록뿐이다 (yaw_offset / goal_z). "
                         "정책은 안 쓴다 — 목표 규약을 정책 경로와 같게 하려는 것")
parser.add_argument("--out", default="")
parser.add_argument("--ff-mode", dest="ff_mode", choices=["planner", "zero"], default="planner")
parser.add_argument("--no-map", action="store_true")
parser.add_argument("--settle", type=int, default=300)
parser.add_argument("--hold", type=int, default=300)
parser.add_argument("--max-steps", dest="max_steps", type=int, default=0)
parser.add_argument("--caster-damping", dest="caster_damping", type=float, default=None)
parser.add_argument("--yaw-offset", dest="yaw_offset", type=float, default=None)
parser.add_argument("--spawn-a", dest="spawn_a", default="")
parser.add_argument("--spawn-b", dest="spawn_b", default="")
parser.add_argument("--min-sep", dest="min_sep", type=float, default=5.0)
parser.add_argument("--max-sep", dest="max_sep", type=float, default=None)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--real-time", dest="real_time", action="store_true")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--raw-coords", dest="raw_coords", action="store_true")
parser.add_argument("--localization", choices=["amcl", "ground_truth"], default="amcl")
parser.add_argument("--hold-only", dest="hold_only", action="store_true")
parser.add_argument("--bar", type=int, default=1)
parser.add_argument("--report", default="")
# --- 제어기 이득 ------------------------------------------------------------
parser.add_argument("--kx", type=float, default=1.5, help="차체 전진오차 이득")
parser.add_argument("--ky", type=float, default=6.0, help="차체 횡오차 이득")
parser.add_argument("--kth", type=float, default=2.5, help="차체 방위오차 이득")
parser.add_argument("--ik-lam", dest="ik_lam", type=float, default=0.05,
                    help="미분 IK 의 DLS 감쇠. 크면 안정하고 느리다")
parser.add_argument("--ik-w-rot", dest="ik_w_rot", type=float, default=0.35,
                    help="자세 오차 가중치. 1.0 은 Isaac Lab 기본(위치[m]과 자세[rad]를 "
                         "같게 봄) 인데, 그러면 자세 60 도가 위치 1 m 와 맞먹어 팔이 "
                         "위치를 포기한다. 낮추면 위치 우선")
parser.add_argument("--ik-max-step", dest="ik_max_step", type=float, default=0.15,
                    help="제어주기당 관절 변화 상한 [rad]")
parser.add_argument("--lead", type=float, default=0.10,
                    help="목표 선행보상 [s]. IK 는 순수 위치 피드백이라 움직이는 목표를 "
                         "뒤따르고 액션 지연(33~117 ms)까지 겹친다. 0 이면 끔")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
if args.gui and not args.real_time:
    args.real_time = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.util  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402

_TWIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _TWIN)
from plan import tracks as trk  # noqa: E402
from runner.classic_episode import ClassicEpisode  # noqa: E402
from runner.recorder import Recorder, md5  # noqa: E402
from world.scene_builder import ASSET, DEFAULT_MAP_USD, make_twin_cfg  # noqa: E402

LINES: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    LINES.append(m)


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("ee_obs_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> bool:
    cfgp = yaml.safe_load(open(args.policy_cfg))
    obs_path = cfgp["observation"]["module"]
    mod = load_module(obs_path)          # 쿼터니언 헬퍼·자세오차 지표용
    yoff = (args.yaw_offset if args.yaw_offset is not None
            else float(cfgp["target"]["yaw_offset"]))
    goal_z = float(cfgp["target"]["goal_z"])

    tracks = {"a": trk.parse(args.track_a), "b": trk.parse(args.track_b)}
    spawn = {}
    if not args.raw_coords:
        from world.spawn_points import MapClearance, pick_two
        mc = MapClearance()

        def _parse(s_):
            v = [float(x) for x in s_.split(",")]
            return (v[0], v[1], v[2])

        if args.spawn_a and args.spawn_b:
            pa, pb = _parse(args.spawn_a), _parse(args.spawn_b)
            ca = mc.track_clearance(tracks["a"], *pa)
            cb = mc.track_clearance(tracks["b"], *pb)
        else:
            (pa, ca), (pb, cb) = pick_two(mc, tracks["a"], tracks["b"],
                                          min_sep=args.min_sep,
                                          max_sep=(args.max_sep if args.max_sep
                                                   else (12.0 if args.gui else None)))
        spawn = {"a": (pa, ca), "b": (pb, cb)}
        tracks = {"a": trk.place_at(tracks["a"], *pa),
                  "b": trk.place_at(tracks["b"], *pb)}
    sep = trk.separation(tracks["a"], tracks["b"])

    say("=" * 76)
    say("고전 제어기 기준선 — 차체 유니사이클 추종 + 팔 미분 IK  (정책 없음)")
    say("=" * 76)
    say("  궤적 A      %s   %.1f s" % (tracks["a"].name, tracks["a"].duration))
    say("  궤적 B      %s   %.1f s" % (tracks["b"].name, tracks["b"].duration))
    for k in ("a", "b"):
        if k in spawn:
            (x, y, yw), c = spawn[k]
            say("  배치 %s      (%.2f, %.2f, %+.2f rad)   궤적 최소 맵여유 %.3f m"
                % (k.upper(), x, y, yw, c))
    say("  두 궤적 최소거리 %.2f m %s"
        % (sep, "" if sep > 2.0 else "  ⚠ 가깝다 — 충돌이 추종오차로 섞인다"))
    say("  차체 이득   k_x %.2f  k_y %.2f  k_th %.2f" % (args.kx, args.ky, args.kth))
    say("  팔 IK       가중 DLS  lambda %.3f  w_rot %.2f  max_step %.2f rad  lead %.3f s"
        % (args.ik_lam, args.ik_w_rot, args.ik_max_step, args.lead))
    say("  목표 규약   %s (yaw %+.1f deg, z %.3f m)  <- 정책 경로와 동일"
        % (os.path.basename(args.policy_cfg), math.degrees(yoff), goal_z))
    say("  ff_mode %s   맵 %s   로컬 %s"
        % (args.ff_mode, "없음(빈 평지)" if args.no_map else "압출 USD", args.localization))

    env = ManagerBasedEnv(cfg=make_twin_cfg(
        map_usd=None if args.no_map else DEFAULT_MAP_USD))
    env.reset()

    if args.caster_damping is not None:
        for s in ("a", "b"):
            r = env.scene[ASSET[s]]
            jn = list(r.data.joint_names)
            cw = [i for i, x in enumerate(jn)
                  if x.startswith("caster") and x.endswith("wheel_joint")]
            r.write_joint_damping_to_sim(
                torch.full((env.num_envs, len(cw)), float(args.caster_damping),
                           device=env.device), joint_ids=cw)
        say("  캐스터 감쇠 %.4f 로 덮어씀" % args.caster_damping)

    viz = None
    if args.gui:
        from world.viz import BarMarker, DebugViz
        viz = DebugViz(goal_z=goal_z)
        viz.draw_tracks(tracks)
        if args.bar:
            BarMarker()
        c = 0.5 * (tracks["a"].centre() + tracks["b"].centre())
        span = max(4.0, float(np.linalg.norm(tracks["a"].centre() - tracks["b"].centre())))
        env.sim.set_camera_view(
            eye=(float(c[0]) - 0.7 * span, float(c[1]) - 0.7 * span, 0.6 * span + 4.0),
            target=(float(c[0]), float(c[1]), 0.4))
        say("  GUI  카메라 중심 (%.1f, %.1f), 스팬 %.1f m" % (c[0], c[1], span))

    ep = ClassicEpisode(
        env, tracks, mod,
        gains=dict(k_x=args.kx, k_y=args.ky, k_theta=args.kth), ik_lam=args.ik_lam,
        ik_w_rot=args.ik_w_rot, ik_max_step=args.ik_max_step, lead=args.lead,
        yaw_offset=yoff, goal_z=goal_z, ff_mode=args.ff_mode,
        settle=args.settle, hold=args.hold, localization=args.localization,
        viz=viz, real_time=args.real_time, speed=args.speed, log=say)

    rec = Recorder(meta={
        "controller": "classic", "gains": {"k_x": args.kx, "k_y": args.ky,
                                           "k_theta": args.kth},
        "ik_lambda": args.ik_lam, "obs_md5": md5(obs_path),
        "yaw_offset": yoff, "goal_z": goal_z, "ff_mode": args.ff_mode,
        "track_a": args.track_a, "track_b": args.track_b, "track_sep": sep,
        "spawn": {k: v[0] for k, v in spawn.items()},
        "map": "none" if args.no_map else DEFAULT_MAP_USD,
        "localization": args.localization, "step_dt": float(env.step_dt),
        "settle": args.settle, "hold": args.hold,
        "caster_damping": args.caster_damping,
    })

    say("")
    info = ep.run(rec, max_steps=args.max_steps or None, hold_only=args.hold_only)
    if args.hold_only:
        # hold 가 수렴하지 않을 때 원인을 가르는 최소 정보. 관절이 한계에 붙어 있으면
        # 도달 문제이고, 안 붙었는데 잔차가 크면 IK/프레임 문제다.
        say("")
        say("hold 종료 시점 진단")
        say("-" * 76)
        for s in ("a", "b"):
            g = ep.diag[s]
            r = env.scene[ASSET[s]]
            q = r.data.joint_pos[0, ep.arm_ids[s]].cpu().numpy()
            lo = r.data.soft_joint_pos_limits[0, ep.arm_ids[s], 0].cpu().numpy()
            hi = r.data.soft_joint_pos_limits[0, ep.arm_ids[s], 1].cpu().numpy()
            near = np.minimum(q - lo, hi - q)
            say("  [%s] IK 잔차 %6.1f mm / %5.1f deg   리치(xy) %.3f m   한계접촉 %s"
                % (s.upper(), g.get("res_pos", 0) * 1e3,
                   math.degrees(g.get("res_ang", 0)), g.get("reach", 0),
                   "예" if g.get("clamped") else "아니오"))
            say("       관절 [deg]  " + "  ".join("%+7.1f" % math.degrees(x) for x in q))
            say("       한계여유    " + "  ".join("%7.1f" % math.degrees(x) for x in near))
            say("       차체오차 종 %+.3f m  횡 %+.3f m  방위 %+.1f deg"
                % (g.get("e_x", 0), g.get("e_y", 0), math.degrees(g.get("e_th", 0))))
        say("-" * 76)
        env.close()
        return True
    say("  재생 %d 스텝 (%.1f s), 완주 %s"
        % (info["steps"], info["steps"] * env.step_dt, info["finished"]))

    dur = info["steps"] * env.step_dt
    say("")
    say("추종 요약   (정책 v29 held-out 참고값: EE 중앙 9.5 / 95%% 17.1 mm, 자세 1.7 deg)")
    say("-" * 76)
    say("  %-18s %10s %10s %10s" % ("항목", "중앙", "95%", "최대"))
    for s in ("a", "b"):
        e = rec.array("ee_err_%s" % s).ravel() * 1e3
        o = rec.array("ori_err_%s" % s).ravel()
        d = np.linalg.norm(rec.array("base_%s" % s)[:, :2]
                           - rec.array("base_ref_%s" % s)[:, :2], axis=1)
        say("  EE %s [mm]          %10.2f %10.2f %10.2f"
            % (s.upper(), np.median(e), np.percentile(e, 95), e.max()))
        say("  자세 %s [deg]        %10.2f %10.2f %10.2f"
            % (s.upper(), np.median(o), np.percentile(o, 95), o.max()))
        say("  차체편차 %s [m]      %10.3f %10.3f %10.3f"
            % (s.upper(), np.median(d), np.percentile(d, 95), d.max()))
    rd = rec.array("robot_dist").ravel()
    say("  로봇간격 [m]       %10.3f %10.3f %10.3f   (최소 %.3f)"
        % (np.median(rd), np.percentile(rd, 95), rd.max(), rd.min()))

    say("")
    say("제어기 진단   — '못 따라갔다' 를 원인별로 가른다")
    say("-" * 76)
    for s in ("a", "b"):
        ipos = rec.array("ik_res_pos_%s" % s).ravel() * 1e3
        iang = np.degrees(rec.array("ik_res_ang_%s" % s).ravel())
        rch = rec.array("reach_%s" % s).ravel()
        be = rec.array("berr_%s" % s)
        tw = rec.array("bref_tw_%s" % s)
        sat = rec.array("bsat_%s" % s).ravel()
        say("  [%s] IK 위치잔차 [mm]   중앙 %7.2f   95%% %7.2f   최대 %7.2f"
            % (s.upper(), np.median(ipos), np.percentile(ipos, 95), ipos.max()))
        say("       IK 자세잔차 [deg]  중앙 %7.2f   95%% %7.2f   최대 %7.2f"
            % (np.median(iang), np.percentile(iang, 95), iang.max()))
        say("       목표까지 리치 [m]  중앙 %7.3f   95%% %7.3f   최대 %7.3f"
            % (np.median(rch), np.percentile(rch, 95), rch.max()))
        say("       차체오차 종/횡/방위  %+.3f m  %+.3f m  %+.1f deg   (|중앙|)"
            % (np.median(np.abs(be[:, 0])), np.median(np.abs(be[:, 1])),
               math.degrees(np.median(np.abs(be[:, 2])))))
        say("       참조속도 v_ref [m/s] 중앙 %+.3f  최소 %+.3f  최대 %+.3f   후진 %.0f%%"
            % (np.median(tw[:, 0]), tw[:, 0].min(), tw[:, 0].max(),
               100.0 * float(np.mean(tw[:, 0] < -1e-3))))
        say("       속도지령 포화       %.0f%% 의 스텝" % (100.0 * float(np.mean(sat))))
        w = rec.array("act_%s" % s)[:, 7]
        flips = int(np.sum(np.diff(np.sign(w)) != 0))
        say("       각속도 지령 부호반전 %.2f 회/초" % (flips / max(1e-6, dur)))
    say("-" * 76)
    say("")
    say("판정 참고")
    say("  IK 잔차가 크고 리치가 reach_max(0.60) 를 넘으면  -> 계획이 팔 밖을 요구한다")
    say("  차체오차가 크고 포화가 잦으면                    -> 계획이 차체 한계보다 빠르다")
    say("  둘 다 작은데 EE 오차가 크면                      -> 제어기 이득 문제 (kx/ky/kth)")

    if args.out:
        say("  로그 %s" % rec.save(args.out))
    env.close()
    return True


if __name__ == "__main__":
    try:
        main()
    finally:
        if args.report:
            with open(args.report, "w") as fh:
                fh.write("\n".join(LINES) + "\n")
        simulation_app.close()
