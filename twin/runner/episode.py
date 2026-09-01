"""한 씬에 두 대 — **각자 자기 궤적을 자기 정책으로** 30 Hz 추종.

두 로봇은 서로 독립이다
-----------------------
막대로 묶여 있지도, 같은 물체를 나르지도 않는다. 같은 맵 위에 같이 있을 뿐이고
각자 자기 EE 궤적을 추종한다. 그래서 궤적도 로봇마다 따로 받는다
(:class:`~plan.tracks.Track`).

초판은 협력 운송 계획에서 두 목표를 유도했는데, 그건 한 대가 계속 후진해야 하는
편대라 v29 학습 분포 밖이었다 (README Phase 1 참고). 여기서는 그 구성을 쓰지 않는다.

세 국면
-------
::

    settle   액션 0. 스폰 자세(INIT_HEIGHT 0.010)가 평형보다 낮아 물리가 자리를 잡는다
    hold     궤적 t=0 목표를 **고정**하고 정책을 돌린다. 기본자세 tool0 이 첫 목표에서
             떨어져 있으므로, 안 하면 큰 초기오차로 시작하고 추종 실패로 오독된다
    run      궤적을 실제로 흘린다

``hold`` 는 실기 관제의 INITIAL 모드와 같은 것이다 — 실기는 EXECUTE 전에 항상
시작 자세를 잡는다.

로컬라이제이션 — 참값이 "안전한 기본값"이 아니다
--------------------------------------------------
학습은 목표를 base 프레임으로 옮길 때 **AMCL 추정값**을 쓴다 (갱신 0.18 s,
지연 0.06 s, 잡음 5 mm). 처음에 나는 "참값은 그 분포의 무잡음 끝이니 안전하다" 고
적었는데, 그것은 **정상상태 이야기**다.

갱신주기와 지연은 잡음이 아니라 **동역학**이다. 참값을 주면 그 지연이 사라져
되먹임 루프의 위상여유가 달라진다. v29 는 루프이득 0.864 로 이미 여유가 크지 않다.
그래서 기본값은 ``localization="amcl"`` 이다.
"""

from __future__ import annotations

import inspect
import math
import time

import numpy as np
import torch

from ee_track_ppo.assets.amr_fr3 import AMR_FR3_ARM_JOINTS, AMR_FR3_EE_BODY
from isaaclab.managers import SceneEntityCfg

from plan.plan_to_target import PlanTargetCfg, TargetStream
from world.scene_builder import ASSET, SIDES, action_layout


def yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


class PoseEstimator:
    """학습의 ``AmclPoseEstimator`` 와 같은 모델 — 로봇 한 대분, numpy.

    **잡음만이 아니라 동역학이다.** 갱신주기 0.18 s 동안 값이 고정되고(zero-order
    hold) 지연 0.06 s 만큼 과거를 본다. 정책은 그 조건에서 학습됐다.

    참값을 주면 이 지연이 사라져 **되먹임 루프의 위상여유가 달라진다** — "참값은
    잡음의 무잡음 끝이니 안전하다" 는 것은 정상상태 이야기이고, 닫힌 루프에서는
    성립하지 않는다.

    ``mode="ground_truth"`` 는 참값을 그대로 통과시킨다 (비교용).
    """

    def __init__(self, step_dt: float, mode: str = "amcl",
                 pos_noise: float = 0.005, yaw_noise: float = 0.004,
                 update_period: float = 0.18, latency: float = 0.06,
                 jump_prob: float = 0.002, jump_pos: float = 0.10,
                 jump_yaw: float = 0.10, rng=None):
        self.mode, self.dt = mode, step_dt
        self.pn, self.yn = pos_noise, yaw_noise
        self.period, self.jump_prob = update_period, jump_prob
        self.jp, self.jy = jump_pos, jump_yaw
        self.hist_len = int(round(latency / step_dt)) + 1
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._buf: list = []
        self._est = None
        self._t_since = 0.0

    def __call__(self, pos_w: np.ndarray, quat_w: np.ndarray):
        if self.mode == "ground_truth":
            return pos_w, quat_w
        yaw = math.atan2(2.0 * (quat_w[0] * quat_w[3] + quat_w[1] * quat_w[2]),
                         1.0 - 2.0 * (quat_w[2] ** 2 + quat_w[3] ** 2))
        truth = np.array([pos_w[0], pos_w[1], yaw])
        self._buf.append(truth)
        if len(self._buf) > self.hist_len:
            self._buf.pop(0)
        delayed = self._buf[0]
        if self._est is None:
            self._est = truth.copy()
        self._t_since += self.dt
        if self._t_since >= self.period:
            cand = delayed + self.rng.normal(0.0, [self.pn, self.pn, self.yn])
            if self.rng.random() < self.jump_prob:
                cand = cand + self.rng.normal(0.0, [self.jp, self.jp, self.jy])
            self._est = cand
            self._t_since = 0.0
        # AMCL 은 2D 다 — z 와 롤·피치는 참값을 그대로 둔다 (학습과 동일)
        est_pos = np.array([self._est[0], self._est[1], pos_w[2]])
        # 참값 쿼터니언에서 yaw 만 추정값으로 갈아끼운다
        w, x, y, z = quat_w
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        q_yaw_inv = np.array([cy, 0.0, 0.0, -sy])
        rp = np.array([w * q_yaw_inv[0] - z * q_yaw_inv[3],
                       x * q_yaw_inv[0] + y * q_yaw_inv[3],
                       y * q_yaw_inv[0] - x * q_yaw_inv[3],
                       z * q_yaw_inv[0] + w * q_yaw_inv[3]])
        e = self._est[2] * 0.5
        q_new = np.array([math.cos(e), 0.0, 0.0, math.sin(e)])
        est_quat = np.array([
            q_new[0] * rp[0] - q_new[3] * rp[3],
            q_new[0] * rp[1] - q_new[3] * rp[2],
            q_new[0] * rp[2] + q_new[3] * rp[1],
            q_new[0] * rp[3] + q_new[3] * rp[0]])
        return est_pos, est_quat


class TwinEpisode:
    """두 대가 각자 궤적을 추종하는 에피소드 하나."""

    def __init__(self, env, tracks: dict, policy, obs_mod, *, yaw_offset: float,
                 goal_z: float, ff_mode: str = "planner", settle: int = 300,
                 hold: int = 300, hold_tol: float = 0.01,
                 localization: str = "amcl", viz=None,
                 real_time: bool = False, speed: float = 1.0, log=print):
        self.env, self.tracks, self.policy, self.mod = env, tracks, policy, obs_mod
        self.log = log
        self.ff_mode, self.settle_n, self.hold_n, self.hold_tol = ff_mode, settle, hold, hold_tol
        self.step_dt = float(env.step_dt)

        self.tcfg = PlanTargetCfg(yaw_offset=yaw_offset, goal_z=goal_z,
                                  step_dt=self.step_dt, ff_mode=ff_mode)
        self.layout = action_layout(env)
        self.n_act = env.action_manager.total_action_dim

        self.robot, self.ee_id, self.arm_ids = {}, {}, {}
        for s in SIDES:
            self.robot[s] = env.scene[ASSET[s]]
            b = SceneEntityCfg(ASSET[s], body_names=[AMR_FR3_EE_BODY]); b.resolve(env.scene)
            j = SceneEntityCfg(ASSET[s], joint_names=AMR_FR3_ARM_JOINTS); j.resolve(env.scene)
            self.ee_id[s], self.arm_ids[s] = b.body_ids[0], j.joint_ids

        self.localization = localization
        self.viz, self.real_time = viz, real_time
        self.speed = max(1e-3, float(speed))
        self._t0 = None
        self.est = {s: PoseEstimator(self.step_dt, mode=localization,
                                     rng=np.random.default_rng(1 + i))
                    for i, s in enumerate(SIDES)}
        self.last_action = {s: np.zeros(self.mod.ACTION_DIM) for s in SIDES}
        self.stream: dict[str, TargetStream] = {}
        self.hold_detail: dict[str, tuple] = {}
        # 판마다 관측 인자가 다르다 (52 차원 v29 vs 58 차원 v12/v28 은
        # arm_joint_abs / base_ref_w / base_ref_twist_in 을 더 받는다).
        # 이름을 하드코딩하지 않고 **서명을 읽어** 맞춘다.
        self._obs_args = set(inspect.signature(self.mod.build_obs).parameters)

    # ------------------------------------------------------------------ 배치
    def place(self) -> None:
        """각 로봇을 **자기 궤적의 t=0 차체 자세**에 놓는다."""
        for s in SIDES:
            r, tr = self.robot[s], self.tracks[s]
            root = r.data.default_root_state.clone()
            x, y, yaw = tr.base[0, 1], tr.base[0, 2], float(tr.base[0, 3])
            w, qx, qy, qz = yaw_quat_wxyz(yaw)
            root[:, 0], root[:, 1] = float(x), float(y)
            root[:, 3], root[:, 4], root[:, 5], root[:, 6] = w, qx, qy, qz
            root[:, 7:] = 0.0
            r.write_root_state_to_sim(root)
            r.write_joint_state_to_sim(r.data.default_joint_pos.clone(),
                                       torch.zeros_like(r.data.default_joint_vel))
        self.env.sim.forward()

    def _make_streams(self) -> None:
        """궤적을 **map 절대좌표 그대로** 흘린다.

        학습은 시작 차체 기준으로 상대화했지만 twin 은 맵과 궤적이 같은 절대좌표에
        있다. ``b_start`` 와 ``origin`` 을 0 으로 두면 변환이 항등이 되어 그대로 나온다.
        관측은 전부 base 프레임이라 정책에는 절대좌표가 보이지 않는다.
        """
        for s in SIDES:
            tr = self.tracks[s]
            self.stream[s] = TargetStream(tr.ee, tr.base, self.tcfg,
                                          b_start=np.zeros(3), origin=np.zeros(3))

    # ------------------------------------------------------------------ 관측
    def _obs(self, s: str, tgt: dict) -> np.ndarray:
        d = self.robot[s].data
        arm_q = (d.joint_pos - d.default_joint_pos)[0, self.arm_ids[s]].cpu().numpy()
        arm_qd = (d.joint_vel - d.default_joint_vel)[0, self.arm_ids[s]].cpu().numpy()
        # base pose 는 **추정값**이다. 학습이 그랬고, 그 지연이 루프 특성의 일부다.
        bp, bq = self.est[s](d.root_pos_w[0].cpu().numpy(),
                             d.root_quat_w[0].cpu().numpy())
        # EE 는 팔 FK 로 base 기준을 구한 뒤 **추정 base 를 통해** 월드로 되돌린다.
        # 그래야 AMCL 오차가 두 번 들어가지 않는다 (ee_obs.py 주석 참고).
        tp = d.root_pos_w[0].cpu().numpy()
        tq = d.root_quat_w[0].cpu().numpy()
        ep = d.body_pos_w[0, self.ee_id[s]].cpu().numpy()
        eq = d.body_quat_w[0, self.ee_id[s]].cpu().numpy()
        ee_b = self.mod.quat_apply_inverse(tq, ep - tp)
        ee_qb = self.mod.quat_mul(self.mod.quat_conj(tq), eq)
        kw = dict(
            arm_joint_pos_rel=arm_q, arm_joint_vel=arm_qd,
            base_lin_x=float(d.root_lin_vel_b[0, 0]),
            base_ang_z=float(d.root_ang_vel_b[0, 2]),
            base_pos_w=bp, base_quat_w=bq,
            ee_pos_w=bp + self.mod.quat_apply(bq, ee_b),
            ee_quat_w=self.mod.quat_mul(bq, ee_qb),
            goal_pos_w=tgt["pos_w"], goal_quat_w=tgt["quat_w"],
            goal_lin_vel_w=tgt["lin_vel_w"], goal_ang_vel_w=tgt["ang_vel_w"],
            last_action=self.last_action[s])
        if "arm_joint_abs" in self._obs_args:
            kw["arm_joint_abs"] = d.joint_pos[0, self.arm_ids[s]].cpu().numpy()
        if "base_ref_w" in self._obs_args:
            kw["base_ref_w"] = tgt["base_ref"]
            kw["base_ref_twist_in"] = tgt["base_ref_twist"]
        return self.mod.build_obs(**kw)

    def _act(self, targets: dict) -> torch.Tensor:
        act = torch.zeros(self.env.num_envs, self.n_act, device=self.env.device)
        for s in SIDES:
            o = torch.as_tensor(self._obs(s, targets[s]), dtype=torch.float32,
                                device=self.env.device).unsqueeze(0)
            with torch.inference_mode():
                a = self.policy(o)[0]
            # ★ 자르지 않는다. 학습의 JointPositionAction 은 팔 액션에 clamp 가 없고
            #   (관절목표 = default + scale * action), 관측의 last_action 도
            #   action_manager.action = **raw** 값이다. 여기서 ±1 로 자르면
            #   팔이 뻗을 수 있는 범위가 잘리고, 그 잘린 값이 관측으로 되먹여진다.
            #   실측 팔 액션의 12~20 %가 |a|>0.98 이라 이 차이가 그대로 드러난다.
            #   차체(DiffDriveAction)는 자기 안에서 max_lin_vel/max_ang_vel 로 자른다.
            self.last_action[s] = a.cpu().numpy().astype(np.float64)
            act[:, self.layout["arm_%s" % s]] = a[:6]
            act[:, self.layout["base_%s" % s]] = a[6:8]
        return act

    # -------------------------------------------------------------- 표시/페이싱
    def _tick(self, targets: dict, k: int) -> None:
        """그리기와 벽시계 맞추기. 헤드리스면 둘 다 무시된다."""
        if self.viz is not None and self.viz.enabled:
            self.viz.update(targets, {
                s: self.robot[s].data.body_pos_w[0, self.ee_id[s]].cpu().numpy()
                for s in SIDES})
        if self.real_time:
            if self._t0 is None:
                self._t0 = time.time()
            # 앞서 있으면만 기다린다. 시뮬이 느리면 그냥 흘려보낸다 (밀린 걸 몰아치지 않는다)
            ahead = (k + 1) * self.step_dt / self.speed - (time.time() - self._t0)
            if ahead > 0:
                time.sleep(ahead)

    # ------------------------------------------------------------------ 지표
    def _ori_err_deg(self, s: str, tgt: dict) -> float:
        """목표 자세와 실제 tool0 자세의 상대회전 크기. 규약에 무관하다.

        요만 빼서 비교하면 안 된다 — 목표는 pitch=pi 라 ZYX 요 추출이 pi 만큼
        어긋나 보인다 (실제로 이 검사에서 167 도가 나와 한참 헤맸다).
        """
        d = self.robot[s].data
        q = d.body_quat_w[0, self.ee_id[s]].cpu().numpy()
        q = q / np.linalg.norm(q)
        qe = self.mod.quat_mul(tgt["quat_w"], self.mod.quat_conj(q))
        if qe[0] < 0:
            qe = -qe
        return math.degrees(2.0 * math.atan2(float(np.linalg.norm(qe[1:])), float(qe[0])))

    def snapshot(self, targets: dict) -> dict:
        out, ee = {}, {}
        for s in SIDES:
            d = self.robot[s].data
            ee[s] = d.body_pos_w[0, self.ee_id[s]].cpu().numpy()
            tgt = targets[s]
            out["ee_%s" % s] = ee[s]
            out["ee_quat_%s" % s] = d.body_quat_w[0, self.ee_id[s]].cpu().numpy()
            out["base_%s" % s] = d.root_pos_w[0].cpu().numpy()
            out["base_yaw_%s" % s] = [float(d.heading_w[0])]
            out["twist_%s" % s] = [float(d.root_lin_vel_b[0, 0]),
                                   float(d.root_ang_vel_b[0, 2])]
            out["jpos_%s" % s] = d.joint_pos[0, self.arm_ids[s]].cpu().numpy()
            out["jvel_%s" % s] = d.joint_vel[0, self.arm_ids[s]].cpu().numpy()
            out["act_%s" % s] = self.last_action[s]
            out["tgt_%s" % s] = tgt["pos_w"]
            out["tgt_yaw_%s" % s] = [tgt["yaw"]]
            out["tgt_lin_%s" % s] = tgt["lin_vel_w"]
            out["base_ref_%s" % s] = tgt["base_ref"]
            out["ee_err_%s" % s] = [float(np.linalg.norm(ee[s] - tgt["pos_w"]))]
            out["ori_err_%s" % s] = [self._ori_err_deg(s, tgt)]
        # 두 대가 서로 방해하는지 — 독립 궤적이라도 한 씬에 있으면 부딪힐 수 있다
        out["robot_dist"] = [float(np.linalg.norm(
            out["base_b"][:2] - out["base_a"][:2]))]
        out["ee_dist"] = [float(np.linalg.norm(ee["b"] - ee["a"]))]
        return out

    # ------------------------------------------------------------------ 실행
    def run(self, rec, max_steps: int | None = None, hold_only: bool = False) -> dict:
        self.place()
        act0 = torch.zeros(self.env.num_envs, self.n_act, device=self.env.device)
        for _ in range(self.settle_n):
            self.env.step(act0)
        self._make_streams()

        # --- hold ------------------------------------------------------------
        frozen = {}
        for s in SIDES:
            st = self.stream[s]
            frozen[s] = st.step()
            st._ep_t = 0.0            # 시계를 되돌린다 — hold 는 궤적을 소비하지 않는다
            st._prev_ee_xy = None
            st._prev_yaw = None
            st._vel[:] = 0.0
        e0 = max(float(np.linalg.norm(
            self.robot[s].data.body_pos_w[0, self.ee_id[s]].cpu().numpy()
            - frozen[s]["pos_w"])) for s in SIDES)
        n_hold = 0
        self._t0 = None
        for k in range(self.hold_n):
            self.env.step(self._act(frozen))
            self._tick(frozen, k)
            n_hold = k + 1
            e = max(float(np.linalg.norm(
                self.robot[s].data.body_pos_w[0, self.ee_id[s]].cpu().numpy()
                - frozen[s]["pos_w"])) for s in SIDES)
            if e < self.hold_tol and k > 30:
                break
        for s in SIDES:
            d = self.robot[s].data
            ee = d.body_pos_w[0, self.ee_id[s]].cpu().numpy()
            self.hold_detail[s] = (float(np.linalg.norm(ee - frozen[s]["pos_w"])),
                                   self._ori_err_deg(s, frozen[s]))
        self.log("  hold %d 스텝, 시작 %.4f m -> " % (n_hold, e0)
                 + "   ".join("%s %.4f m / %.1f deg" % (s.upper(), *self.hold_detail[s])
                              for s in SIDES))
        if hold_only:
            return {"steps": 0, "finished": False, "hold_steps": n_hold,
                    "hold_detail": self.hold_detail}

        # --- run -------------------------------------------------------------
        dur = max(self.tracks[s].duration for s in SIDES)
        n_max = max_steps or int(math.ceil(dur / self.step_dt)) + 1
        finished = False
        self._t0 = None
        for k in range(n_max):
            tgt = {s: self.stream[s].step() for s in SIDES}
            self.env.step(self._act(tgt))
            self._tick(tgt, k)
            rec.add(t=[k * self.step_dt], plan_t=[tgt["a"]["plan_t"]],
                    **self.snapshot(tgt))
            if all(tgt[s]["finished"] for s in SIDES):
                finished = True
                break
        return {"steps": len(rec), "finished": finished,
                "hold_steps": n_hold, "hold_detail": self.hold_detail}
