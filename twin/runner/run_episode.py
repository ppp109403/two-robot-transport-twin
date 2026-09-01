"""두 대를 한 씬에 올려 **각자 자기 궤적을 자기 정책으로** 추종시킨다.

::

    twin/run.sh twin/runner/run_episode.py --track-a npz:98 --track-b npz:110 \\
        --out twin/data/runs/pair_98_110 --report /tmp/ep.txt

궤적 지정::

    npz:<i>              data/plans.npz 의 i 번 (실기 MPC 주행, munji 맵 좌표)
    npz:<path>:<i>       다른 npz
    csv:<dir>:<a|b>      협력 운송 계획에서 한쪽만 (편대를 일부러 볼 때만)

``data/plans.npz`` 의 **뒤 24 개는 학습에서 뺀 held-out** 이고, v29 가 거기서
EE 중앙 9.5 mm 를 냈다. 그래서 그 구간의 궤적으로 돌리면 twin 자체가 종단 검증된다
— 9.5 mm 근처가 나와야 씬·관측·액션·보간이 전부 맞는 것이다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--track-a", dest="track_a", default="npz:98")
parser.add_argument("--track-b", dest="track_b", default="npz:110")
parser.add_argument("--policy-cfg", dest="policy_cfg", default="twin/configs/policy_v29.yaml")
parser.add_argument("--out", default="")
parser.add_argument("--ff-mode", dest="ff_mode", choices=["planner", "zero"], default="planner")
parser.add_argument("--no-map", action="store_true",
                    help="빈 평지. 맵이 원인인지 가를 때 쓴다 (학습은 빈 평지였다)")
parser.add_argument("--settle", type=int, default=300)
parser.add_argument("--hold", type=int, default=300)
parser.add_argument("--max-steps", dest="max_steps", type=int, default=0)
parser.add_argument("--caster-damping", dest="caster_damping", type=float, default=None)
parser.add_argument("--yaw-offset", dest="yaw_offset", type=float, default=None)
parser.add_argument("--spawn-a", dest="spawn_a", default="",
                    help="x,y,yaw 로 A 배치 자리 지정. 비우면 맵에서 자동 선택")
parser.add_argument("--spawn-b", dest="spawn_b", default="")
parser.add_argument("--min-sep", dest="min_sep", type=float, default=5.0)
parser.add_argument("--max-sep", dest="max_sep", type=float, default=None,
                    help="두 로봇을 이만큼 안쪽에 둔다. GUI 로 한 화면에 담을 때")
parser.add_argument("--gui", action="store_true",
                    help="Isaac Sim 창을 띄우고 계획·목표·오차선을 그린다")
parser.add_argument("--real-time", dest="real_time", action="store_true",
                    help="벽시계에 맞춰 재생 (기본은 최대 속도)")
parser.add_argument("--speed", type=float, default=1.0,
                    help="실시간 재생 배속. 84 초짜리 계획을 4배속으로 보려면 4")
parser.add_argument("--raw-coords", dest="raw_coords", action="store_true",
                    help="궤적 절대좌표를 그대로 쓴다. plans.npz 는 맵과 무관하니 쓰지 말 것")
parser.add_argument("--localization", choices=["amcl", "ground_truth"],
                    default="amcl", help="base pose 를 추정값으로 줄지 참값으로 줄지")
parser.add_argument("--hold-only", dest="hold_only", action="store_true")
parser.add_argument("--bar", type=int, default=1,
                    help="GUI 에서 두 EE 사이를 갈색 막대로 잇는다 (시각물)")
parser.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# --gui 면 렌더링을 켠다. 나머지는 전부 헤드리스가 기본이다.
args.headless = not args.gui
if args.gui:
    args.real_time = True if not args.real_time else args.real_time

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
from runner.episode import TwinEpisode  # noqa: E402
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
    obs_path, jit_path = cfgp["observation"]["module"], cfgp["checkpoint"]["jit"]
    mod = load_module(obs_path)
    yoff = (args.yaw_offset if args.yaw_offset is not None
            else float(cfgp["target"]["yaw_offset"]))

    tracks = {"a": trk.parse(args.track_a), "b": trk.parse(args.track_b)}
    spawn = {}
    if not args.raw_coords:
        # 궤적의 절대좌표는 맵과 무관하다 (tracks 모듈 경고 참고). 모양만 쓰고
        # 놓을 자리는 맵의 자유공간에서 고른다 — 학습의 상대화 규약과 같다.
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
    say("두 대 동시 재생 — %s" % cfgp["name"])
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
    say("  정책        %s (md5 %s)" % (jit_path, md5(jit_path)[:8]))
    say("  관측        %s (md5 %s, %d 차원)" % (obs_path, md5(obs_path)[:8], mod.OBS_DIM))
    say("  요 보정     %+.6f rad (%+.1f deg)  <- %s"
        % (yoff, math.degrees(yoff),
           "CLI 덮어씀" if args.yaw_offset is not None else "config 값"))
    say("  목표 z      %.3f m   ff_mode %s   맵 %s   로컬 %s"
        % (cfgp["target"]["goal_z"], args.ff_mode,
           "없음(빈 평지)" if args.no_map else "압출 USD", args.localization))

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
        viz = DebugViz(goal_z=float(cfgp["target"]["goal_z"]))
        viz.draw_tracks(tracks)
        if args.bar:
            ep_bar = BarMarker()
        # 두 로봇을 한 화면에 담는다 — 중점 위 비스듬한 곳에서 내려다본다
        c = 0.5 * (tracks["a"].centre() + tracks["b"].centre())
        span = max(4.0, float(np.linalg.norm(tracks["a"].centre() - tracks["b"].centre())))
        env.sim.set_camera_view(
            eye=(float(c[0]) - 0.7 * span, float(c[1]) - 0.7 * span, 0.6 * span + 4.0),
            target=(float(c[0]), float(c[1]), 0.4))
        say("  GUI  카메라 중심 (%.1f, %.1f), 스팬 %.1f m, debug_draw %s"
            % (c[0], c[1], span, "ON" if viz.enabled else "없음"))

    policy = torch.jit.load(jit_path, map_location=env.device).eval()
    ep = TwinEpisode(env, tracks, policy, mod, yaw_offset=yoff,
                     goal_z=float(cfgp["target"]["goal_z"]), ff_mode=args.ff_mode,
                     settle=args.settle, hold=args.hold,
                     localization=args.localization, viz=viz,
                     real_time=args.real_time, speed=args.speed, log=say)
    rec = Recorder(meta={
        "policy": cfgp["name"], "jit_md5": md5(jit_path), "obs_md5": md5(obs_path),
        "obs_dim": mod.OBS_DIM, "yaw_offset": yoff, "goal_z": cfgp["target"]["goal_z"],
        "ff_mode": args.ff_mode, "track_a": args.track_a, "track_b": args.track_b,
        "track_sep": sep, "spawn": {k: v[0] for k, v in spawn.items()}, "map": "none" if args.no_map else DEFAULT_MAP_USD,
        "localization": args.localization, "step_dt": float(env.step_dt),
        "settle": args.settle, "hold": args.hold, "caster_damping": args.caster_damping,
    })

    say("")
    info = ep.run(rec, max_steps=args.max_steps or None, hold_only=args.hold_only)
    if args.hold_only:
        env.close()
        return True
    say("  재생 %d 스텝 (%.1f s), 완주 %s"
        % (info["steps"], info["steps"] * env.step_dt, info["finished"]))

    say("")
    say("추종 요약   (v29 held-out 기대: EE 중앙 9.5 / 95%% 17.1 mm, 자세 1.7 deg)")
    say("-" * 76)
    say("  %-14s %10s %10s %10s" % ("항목", "중앙", "95%", "최대"))
    for s in ("a", "b"):
        e = rec.array("ee_err_%s" % s).ravel() * 1e3
        o = rec.array("ori_err_%s" % s).ravel()
        d = np.linalg.norm(rec.array("base_%s" % s)[:, :2]
                           - rec.array("base_ref_%s" % s)[:, :2], axis=1)
        say("  EE %s [mm]      %10.2f %10.2f %10.2f" % (s.upper(), np.median(e),
                                                        np.percentile(e, 95), e.max()))
        say("  자세 %s [deg]    %10.2f %10.2f %10.2f" % (s.upper(), np.median(o),
                                                         np.percentile(o, 95), o.max()))
        say("  차체편차 %s [m]  %10.3f %10.3f %10.3f" % (s.upper(), np.median(d),
                                                        np.percentile(d, 95), d.max()))
    rd = rec.array("robot_dist").ravel()
    say("  로봇간격 [m]   %10.3f %10.3f %10.3f   (최소 %.3f)"
        % (np.median(rd), np.percentile(rd, 95), rd.max(), rd.min()))
    for s in ("a", "b"):
        w = rec.array("act_%s" % s)[:, 7]
        flips = int(np.sum(np.diff(np.sign(w)) != 0))
        say("  %s 각속도 지령 부호반전 %.2f 회/초"
            % (s.upper(), flips / max(1e-6, info["steps"] * env.step_dt)))
    say("-" * 76)

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
