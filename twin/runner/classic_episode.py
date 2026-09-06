"""정책 대신 **고전 제어기**로 같은 계획을 추종한다.

무엇을 확인하려는 것인가
------------------------
"MPC 가 낸 계획을 실제 로봇의 차체와 팔이 따라갈 수 있는가" 는 정책과 **무관한
질문**이다. 그런데 지금까지는 정책으로만 재생해 왔기 때문에, 추종이 나쁠 때
그것이 계획 탓인지 정책 탓인지 가를 수가 없었다.

이 모듈은 그 기준선을 만든다::

    차체   base_ref 를 유니사이클 궤적 추종기로   (control/base_ctrl.py)
    팔     ee 목표를 미분 IK 로                   (control/arm_ik.py)

여기서도 못 따라가면 **계획이 문제**다 (너무 빠르거나, 리치 밖이거나, 후진을
요구하거나). 여기서는 따라가는데 정책이 못 따라가면 그때 정책 문제다.

무엇을 공유하는가 — 전부
------------------------
:class:`~runner.episode.TwinEpisode` 를 그대로 상속하고 **``_act`` 하나만** 바꾼다.
씬, 배치, settle/hold/run 국면, 목표 생성(``TargetStream``), 로컬라이제이션 모델,
지표 계산이 전부 같은 코드다. 그래야 두 결과를 나란히 놓을 수 있다.

정책과 같은 조건으로 맞춘 것
----------------------------
::

    목표        TargetStream 산출물 그대로. 제어기용으로 다시 만들지 않는다
    로컬라이제이션  같은 AMCL 모델. 참값을 주면 실기가 못 가진 정보로 제어하는 셈이다
    액션 공간   같은 DiffDriveAction / DelayedJointPositionAction 을 통과한다
                -> 가감속 램프, 액션 지연(4~14 물리스텝), 속도 한계가 동일하게 걸린다
    hold 국면   시작 자세를 먼저 잡고 시작

즉 **플랜트는 완전히 같고 제어기만 다르다.**

한계 — 이건 상한이 아니다
-------------------------
IK 는 제어주기마다 한 스텝만 도는 국소 해법이라, 목표가 리치 밖이거나 특이점을
지나면 조용히 뒤처진다. 그래서 이 기준선이 나쁘다고 "계획이 불가능하다" 로 바로
읽으면 안 된다. ``ik_res_*`` 와 ``reach_*`` 를 같이 봐야 원인이 갈린다.
"""

from __future__ import annotations

import numpy as np
import torch

from control.arm_ik import ArmIK
from control.base_ctrl import UnicycleTracker
from runner.episode import TwinEpisode
from world.scene_builder import ASSET, SIDES


class ClassicEpisode(TwinEpisode):
    """정책 자리에 차체 추종기 + 팔 IK 를 넣은 에피소드."""

    def __init__(self, env, tracks, obs_mod, *, gains=None, ik_lam: float = 0.05,
                 ik_w_rot: float = 0.35, ik_max_step: float = 0.15,
                 lead: float = 0.10, **kw):
        """
        Args:
            obs_mod: ``ee_obs.py``. 관측을 만들지는 않지만 쿼터니언 헬퍼와
                자세오차 지표가 여기 있다 — 정책 경로와 **같은 구현**을 쓰기 위해
                그대로 받는다.
            gains: ``dict(k_x=, k_y=, k_theta=)``. 없으면 기본값.
            ik_lam: DLS 감쇠.
            ik_w_rot: 자세 오차 가중치 (``control.arm_ik`` docstring 참고).
            ik_max_step: 제어주기당 관절 변화 상한 [rad].
            lead: 선행보상 [s]. 목표를 이 시간만큼 앞서 준다 — 아래 참고.
        """
        super().__init__(env, tracks, policy=None, obs_mod=obs_mod, **kw)
        g = dict(k_x=1.5, k_y=6.0, k_theta=2.5)
        g.update(gains or {})
        self.gains = g

        terms = dict(zip(env.action_manager.active_terms,
                         env.action_manager._terms.values()))

        self.base_ctrl, self.arm, self._arm_sc, self._blim = {}, {}, {}, {}
        for s in SIDES:
            bt = terms["base_%s" % s]
            lim = dict(v_max=float(bt.cfg.max_lin_vel),
                       v_min=-float(bt.cfg.max_reverse_vel),
                       w_max=float(bt.cfg.max_ang_vel))
            self._blim[s] = lim
            self.base_ctrl[s] = UnicycleTracker(self.step_dt, **g, **lim)

            self.arm[s] = ArmIK(self.robot[s], self.ee_id[s], self.arm_ids[s],
                                lam=ik_lam, w_rot=ik_w_rot, max_step=ik_max_step,
                                device=str(env.device))

            # 팔 액션 규약을 **런타임에 읽는다**: q_des = offset + scale * action
            at = terms["arm_%s" % s]
            sc, off = at._scale, at._offset
            sc = (np.full(6, float(sc)) if not torch.is_tensor(sc)
                  else sc[0].detach().cpu().numpy().astype(np.float64))
            off = (np.full(6, float(off)) if not torch.is_tensor(off)
                   else off[0].detach().cpu().numpy().astype(np.float64))
            self._arm_sc[s] = (sc, off)

        self.lead = float(lead)
        self.diag: dict[str, dict] = {s: {} for s in SIDES}

    # ------------------------------------------------------------------ 선행보상
    def _lead_target(self, tgt: dict) -> tuple[np.ndarray, np.ndarray]:
        """목표를 ``lead`` 초만큼 앞서 준다.

        왜 필요한가 — IK 는 **순수 위치 피드백**이라 움직이는 목표를 구조적으로
        뒤따른다. 게다가 플랜트에 액션 지연(4~14 물리스텝 = 33~117 ms)이 있어서
        지연이 두 겹이다. 정책은 관측으로 목표 속도를 받아 이걸 메우는데
        (실기 실측: 피드포워드를 막으면 0.15 m/s 에서 9.9 -> 26.9 mm), 제어기에는
        그런 채널이 없으므로 여기서 명시적으로 넣는다.

        쓰는 값은 ``TargetStream`` 이 이미 만들어 둔 피드포워드 속도다. 목표를
        새로 만들지 않으므로 "제어기가 더 쉬운 목표를 받았다" 가 되지 않는다.
        """
        from plan.plan_to_target import quat_pitch_pi
        if self.lead <= 0.0:
            return (np.asarray(tgt["pos_w"], dtype=np.float64),
                    np.asarray(tgt["quat_w"], dtype=np.float64))
        p = np.asarray(tgt["pos_w"], dtype=np.float64) \
            + np.asarray(tgt["lin_vel_w"], dtype=np.float64) * self.lead
        yaw = float(tgt["yaw"]) + float(tgt["ang_vel_w"][2]) * self.lead
        return p, np.asarray(quat_pitch_pi(yaw), dtype=np.float64)

    # ------------------------------------------------------------------ 제어
    def _act(self, targets: dict) -> torch.Tensor:
        act = torch.zeros(self.env.num_envs, self.n_act, device=self.env.device)
        for s in SIDES:
            d = self.robot[s].data
            # 실기가 가진 것과 같은 자세 추정값으로 제어한다
            bp, bq = self.est[s](d.root_pos_w[0].cpu().numpy(),
                                 d.root_quat_w[0].cpu().numpy())
            yaw = self.mod.yaw_from_quat(bq)

            # --- 차체 ---------------------------------------------------------
            bc = self.base_ctrl[s].step(np.array([bp[0], bp[1], yaw]),
                                        targets[s]["base_ref"])
            lim = self._blim[s]
            a_base = np.array([bc["v"] / lim["v_max"], bc["w"] / lim["w_max"]])

            # --- 팔 -----------------------------------------------------------
            t_pos, t_quat = self._lead_target(targets[s])
            ik = self.arm[s].solve(t_pos, t_quat, bp, bq)
            sc, off = self._arm_sc[s]
            a_arm = (ik["q_des"] - off) / sc

            # 정책 경로와 같은 자리에 같은 의미로 채운다 (부호반전 지표 등이 그대로 돈다)
            self.last_action[s] = np.concatenate([a_arm, a_base])
            self.diag[s] = {**bc, **{k: v for k, v in ik.items() if k != "q_des"}}

            act[:, self.layout["arm_%s" % s]] = torch.as_tensor(
                a_arm, dtype=torch.float32, device=self.env.device)
            act[:, self.layout["base_%s" % s]] = torch.as_tensor(
                a_base, dtype=torch.float32, device=self.env.device)
        return act

    # ------------------------------------------------------------------ 지표
    def snapshot(self, targets: dict) -> dict:
        """정책 경로의 채널 전부 + 제어기 진단.

        진단이 있어야 "못 따라갔다" 를 원인별로 가를 수 있다::

            ik_res_pos 크다   목표가 리치 밖이거나 특이점  -> 계획/기구 문제
            reach 크다        차체가 뒤처져 목표가 멀어짐  -> 차체 추종 문제
            sat 잦다          속도 한계 포화              -> 계획이 너무 빠르다
        """
        out = super().snapshot(targets)
        for s in SIDES:
            g = self.diag.get(s) or {}
            out["ik_res_pos_%s" % s] = [float(g.get("res_pos", 0.0))]
            out["ik_res_ang_%s" % s] = [float(g.get("res_ang", 0.0))]
            out["reach_%s" % s] = [float(g.get("reach", 0.0))]
            out["berr_%s" % s] = [float(g.get("e_x", 0.0)),
                                  float(g.get("e_y", 0.0)),
                                  float(g.get("e_th", 0.0))]
            out["bref_tw_%s" % s] = [float(g.get("v_ref", 0.0)),
                                     float(g.get("w_ref", 0.0))]
            out["bsat_%s" % s] = [1.0 if g.get("sat") else 0.0]
            out["ikclamp_%s" % s] = [1.0 if g.get("clamped") else 0.0]
        return out
