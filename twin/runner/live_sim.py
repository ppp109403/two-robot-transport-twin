"""Isaac Sim 을 띄워 놓고 **MPC 노드가 내는 셋포인트를 실시간으로** 두 대에 물린다.

흐름 — RViz 에서 조작한다
-------------------------
::

    1  2D Pose Estimate / 2D Goal Pose   물체 시작·목표를 찍는다 -> 노드가 푼다
    2  RViz 애니메이션으로 계획을 확인    (여기서 마음에 안 들면 다시 찍으면 된다)
    3  [SETUP SIM] 버튼                  Isaac 이 로봇 두 대를 계획 t=0 자세로 스폰,
                                         팔이 첫 파지점으로 붙는다 (정책이 스스로)
    4  [EXECUTE] 버튼                    계획대로 재생. Isaac 도 같이 움직인다

**SETUP 전에는 로봇을 씬 밖에 세워 둔다.** 계획이 오기 전에 두 대를 원점에 올리면
서로 겹쳐서 밀어내느라 버벅인다 (실제로 그랬다). 스폰 시점을 사람이 정하게 하는 것이
이 버튼의 존재 이유다.

상태
----
::

    park   대기.  맵 밖 먼 곳에 떨어뜨려 두고 액션 0
    hold   SETUP 을 받아 시작 자세로 옮긴 뒤, **t=0 EE 목표를 붙잡고** 있다.
           EXECUTE 전에는 노드가 셋포인트를 아예 발행하지 않으므로 목표는
           latched ee_path 의 첫 점을 다리가 실어 준 것을 쓴다
    run    셋포인트가 흐르는 동안. 다리가 도착 시각으로 판정해 준다
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy-cfg", dest="policy_cfg", default="twin/configs/policy_v29.yaml")
parser.add_argument("--shm", default="/dev/shm/twin_bridge.bin")
parser.add_argument("--gui", action="store_true", help="Isaac Sim 창을 띄운다")
parser.add_argument("--no-map", action="store_true")
parser.add_argument("--localization", choices=["amcl", "ground_truth"], default="amcl")
parser.add_argument("--settle", type=int, default=90, help="스폰 뒤 가라앉히는 스텝")
parser.add_argument("--park", type=float, nargs=2, default=[60.0, 60.0],
                    help="대기 자리 (맵 밖). 두 대를 8 m 떼어 놓는다")
parser.add_argument("--bar", type=int, default=1,
                    help="두 EE 사이를 갈색 막대로 잇는다 (시각물, 물리 아님). 0 이면 끔")
parser.add_argument("--bar-radius", dest="bar_radius", type=float, default=0.075)
parser.add_argument("--wait", type=float, default=120.0, help="다리 연결 대기 [s]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.util  # noqa: E402
import inspect  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from ee_track_ppo.assets.amr_fr3 import AMR_FR3_ARM_JOINTS, AMR_FR3_EE_BODY  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

_TWIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _TWIN)
from plan.plan_to_target import quat_pitch_pi  # noqa: E402
from ros.shm import MODE_RUN, SIDES, Bridge  # noqa: E402
from runner.episode import PoseEstimator, yaw_quat_wxyz  # noqa: E402
from world.scene_builder import ASSET, DEFAULT_MAP_USD, action_layout, make_twin_cfg  # noqa: E402


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("ee_obs_live", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quat_yaw(w, x, y, z) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class LiveTarget:
    """들어오는 **2D** 셋포인트를 정책이 먹는 3D 목표로 바꾼다.

    실기 ``plan_adapter.py`` 와 같은 일이다. 플래너의 EE 셋포인트는 z=0, pitch=0 이라
    그대로는 못 쓴다::

        z      -> goal_z (v29 는 0.55)
        pitch  -> pi (손을 아래로)
        yaw    -> 막대방향 + yaw_offset
        속도    -> 수치미분 (플래너가 EE 속도를 안 준다)
    """

    def __init__(self, yaw_offset: float, goal_z: float, step_dt: float):
        self.yoff, self.goal_z, self.dt = yaw_offset, goal_z, step_dt
        self.reset()

    def reset(self):
        self._prev_xy = None
        self._prev_yaw = None

    def from_xyyaw(self, x, y, bar_yaw, moving=True) -> dict:
        yaw = bar_yaw + self.yoff
        lin, ang = np.zeros(3), np.zeros(3)
        if moving and self._prev_xy is not None:
            lin[0] = (x - self._prev_xy[0]) / self.dt
            lin[1] = (y - self._prev_xy[1]) / self.dt
            d = (yaw - self._prev_yaw + math.pi) % (2 * math.pi) - math.pi
            ang[2] = d / self.dt
        self._prev_xy, self._prev_yaw = (x, y), yaw
        return {"pos_w": np.array([x, y, self.goal_z]),
                "quat_w": quat_pitch_pi(yaw), "yaw": yaw,
                "lin_vel_w": lin, "ang_vel_w": ang}


def main() -> None:
    cfg = yaml.safe_load(open(args.policy_cfg))
    mod = load_module(cfg["observation"]["module"])
    yoff = float(cfg["target"]["yaw_offset"])
    goal_z = float(cfg["target"]["goal_z"])

    print("=" * 76, flush=True)
    print("Isaac 실시간 — %s, 관측 %d 차원, 요 보정 %+.1f deg, 목표 z %.2f"
          % (cfg["name"], mod.OBS_DIM, math.degrees(yoff), goal_z), flush=True)
    print("=" * 76, flush=True)

    env = ManagerBasedEnv(cfg=make_twin_cfg(
        map_usd=None if args.no_map else DEFAULT_MAP_USD))
    env.reset()
    step_dt = float(env.step_dt)
    layout = action_layout(env)
    n_act = env.action_manager.total_action_dim
    zero = torch.zeros(env.num_envs, n_act, device=env.device)
    policy = torch.jit.load(cfg["checkpoint"]["jit"], map_location=env.device).eval()

    robot, ee_id, arm_ids = {}, {}, {}
    for s in SIDES:
        robot[s] = env.scene[ASSET[s]]
        b = SceneEntityCfg(ASSET[s], body_names=[AMR_FR3_EE_BODY]); b.resolve(env.scene)
        j = SceneEntityCfg(ASSET[s], joint_names=AMR_FR3_ARM_JOINTS); j.resolve(env.scene)
        ee_id[s], arm_ids[s] = b.body_ids[0], j.joint_ids

    def put(s, x, y, yaw):
        r = robot[s]
        root = r.data.default_root_state.clone()
        w, qx, qy, qz = yaw_quat_wxyz(float(yaw))
        root[:, 0], root[:, 1] = float(x), float(y)
        root[:, 3], root[:, 4], root[:, 5], root[:, 6] = w, qx, qy, qz
        root[:, 7:] = 0.0
        r.write_root_state_to_sim(root)
        r.write_joint_state_to_sim(r.data.default_joint_pos.clone(),
                                   torch.zeros_like(r.data.default_joint_vel))

    # 대기 자리 — 맵 밖. 겹쳐 두면 서로 밀어내며 버벅인다.
    for i, s in enumerate(SIDES):
        put(s, args.park[0] + 8.0 * i, args.park[1], 0.0)
    env.sim.forward()

    viz, bar = None, None
    if args.gui:
        from world.viz import BarMarker, DebugViz
        viz = DebugViz(goal_z=goal_z)
        if args.bar:
            bar = BarMarker(radius=args.bar_radius)

    # --- 다리 연결 --------------------------------------------------------
    t0, br = time.time(), None
    while time.time() - t0 < args.wait and simulation_app.is_running():
        try:
            br = Bridge(args.shm, create=False)
            break
        except (FileNotFoundError, ValueError) as exc:
            print("  다리 대기: %s" % exc, flush=True)
            for _ in range(60):
                env.step(zero)
    if br is None:
        raise SystemExit("다리를 못 찾았다: %s  (run_bridge.sh 를 먼저)" % args.shm)
    print("  다리 연결. RViz 에서 목표를 찍고 SETUP 을 누르면 스폰한다.", flush=True)

    est = {s: PoseEstimator(step_dt, mode=args.localization) for s in SIDES}
    tgen = {s: LiveTarget(yoff, goal_z, step_dt) for s in SIDES}
    last_action = {s: np.zeros(mod.ACTION_DIM) for s in SIDES}
    obs_args = set(inspect.signature(mod.build_obs).parameters)

    state = "park"
    cur_spawn = 0
    last_cmd = None
    wall0, k = time.time(), 0

    while simulation_app.is_running():
        c = br.read_cmd() or last_cmd
        last_cmd = c
        act = zero

        if c is not None and c["spawn_id"] != cur_spawn and c["spawn_id"] > 0:
            cur_spawn = c["spawn_id"]
            for s in SIDES:
                st = c["start"][s]
                put(s, st[0], st[1], st[2])
                est[s] = PoseEstimator(step_dt, mode=args.localization)
                tgen[s].reset()
                last_action[s][:] = 0.0
            env.sim.forward()
            for _ in range(args.settle):
                env.step(zero)
            state = "hold"
            print("  SETUP #%d — 시작 자세로 스폰. 팔이 첫 파지점으로 붙는다."
                  % cur_spawn, flush=True)

        if c is not None and state != "park":
            running = c["mode"] == MODE_RUN
            if running and state != "run":
                state = "run"
                print("  EXECUTE — 계획 재생 시작", flush=True)
            elif not running and state == "run":
                state = "hold"
                print("  재생 끝 (또는 정지). 목표를 붙잡고 대기", flush=True)

            targets = {}
            act = torch.zeros(env.num_envs, n_act, device=env.device)
            for s in SIDES:
                if running:
                    ec = c["side"][s]
                    tgt = tgen[s].from_xyyaw(ec[0], ec[1],
                                             quat_yaw(ec[3], ec[4], ec[5], ec[6]))
                else:
                    st = c["start"][s]      # latched ee_path 의 첫 점
                    tgt = tgen[s].from_xyyaw(st[3], st[4], st[5], moving=False)
                targets[s] = tgt

                d = robot[s].data
                bp, bq = est[s](d.root_pos_w[0].cpu().numpy(),
                                d.root_quat_w[0].cpu().numpy())
                tp = d.root_pos_w[0].cpu().numpy()
                tq = d.root_quat_w[0].cpu().numpy()
                ee_b = mod.quat_apply_inverse(
                    tq, d.body_pos_w[0, ee_id[s]].cpu().numpy() - tp)
                ee_qb = mod.quat_mul(mod.quat_conj(tq),
                                     d.body_quat_w[0, ee_id[s]].cpu().numpy())
                kw = dict(
                    arm_joint_pos_rel=(d.joint_pos - d.default_joint_pos)[0, arm_ids[s]].cpu().numpy(),
                    arm_joint_vel=(d.joint_vel - d.default_joint_vel)[0, arm_ids[s]].cpu().numpy(),
                    base_lin_x=float(d.root_lin_vel_b[0, 0]),
                    base_ang_z=float(d.root_ang_vel_b[0, 2]),
                    base_pos_w=bp, base_quat_w=bq,
                    ee_pos_w=bp + mod.quat_apply(bq, ee_b),
                    ee_quat_w=mod.quat_mul(bq, ee_qb),
                    goal_pos_w=tgt["pos_w"], goal_quat_w=tgt["quat_w"],
                    goal_lin_vel_w=tgt["lin_vel_w"], goal_ang_vel_w=tgt["ang_vel_w"],
                    last_action=last_action[s])
                if "arm_joint_abs" in obs_args:
                    kw["arm_joint_abs"] = d.joint_pos[0, arm_ids[s]].cpu().numpy()
                if "base_ref_w" in obs_args:
                    bc = c["side"][s][7:14]
                    kw["base_ref_w"] = np.array([bc[0], bc[1],
                                                 quat_yaw(bc[3], bc[4], bc[5], bc[6])])
                    kw["base_ref_twist_in"] = np.array(c["side"][s][14:16])
                o = torch.as_tensor(mod.build_obs(**kw), dtype=torch.float32,
                                    device=env.device).unsqueeze(0)
                with torch.inference_mode():
                    a = policy(o)[0]      # ★ clamp 하지 않는다 (학습과 동일)
                last_action[s] = a.cpu().numpy().astype(np.float64)
                act[:, layout["arm_%s" % s]] = a[:6]
                act[:, layout["base_%s" % s]] = a[6:8]

            if viz is not None and viz.enabled:
                viz.update(targets, {s: robot[s].data.body_pos_w[0, ee_id[s]].cpu().numpy()
                                     for s in SIDES})

        if bar is not None and state != "park":
            bar.update(robot["a"].data.body_pos_w[0, ee_id["a"]].cpu().numpy(),
                       robot["b"].data.body_pos_w[0, ee_id["b"]].cpu().numpy())

        env.step(act)

        meas = {}
        for s in SIDES:
            d = robot[s].data
            meas[s] = (*d.body_pos_w[0, ee_id[s]].cpu().numpy(),
                       *d.body_quat_w[0, ee_id[s]].cpu().numpy(),
                       *d.root_pos_w[0].cpu().numpy(),
                       *d.root_quat_w[0].cpu().numpy())
        br.write_meas(meas)

        k += 1
        ahead = k * step_dt - (time.time() - wall0)
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -1.0:
            wall0, k = time.time(), 0       # 크게 밀리면 몰아치지 않고 기준을 다시 잡는다

    br.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
