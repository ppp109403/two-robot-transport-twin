"""한 씬에 로봇 두 대 + 압출한 실제 맵. Phase 1 의 무대.

설계 원칙 — **설정을 다시 적지 않는다**
---------------------------------------
액션 스케일, 액션 지연(4~14 물리스텝), diff-drive 한계·가감속 램프, 페이로드 USD,
sim dt, decimation. 이걸 여기 손으로 옮겨 적으면 학습과 갈라진 **두 벌째**가 생기고,
이 저장소가 반복해서 당한 사고가 정확히 그것이다.

그래서 ``AmrFr3TrajV29EnvCfg`` 를 **인스턴스로 만들어 그 안의 설정을 꺼내 쓴다.**
v29 가 무엇을 바꾸든 자동으로 따라온다. 바꾸는 것은 셋뿐::

    asset_name    robot        -> robot_a / robot_b
    prim_path     {ENV}/Robot  -> {ENV}/Robot_A / Robot_B
    씬            빈 평지       -> 평지 + 압출한 맵

왜 ManagerBasedEnv 인가 (RL env 가 아니라)
------------------------------------------
필요한 것은 물리 스테핑과 **학습과 동일한 액션 적용**뿐이다. 관측은 ``ee_obs.py`` 로
직접 만들고 (G1 이 그 등가성을 확인했다), 보상·종료·커맨드는 쓰지 않는다.
RL env 를 상속하면 커맨드 term 이 로봇 하나를 전제로 깔려 있어 두 대로 늘릴 때
전부 이중화해야 한다 — 얻는 것 없이 표면적만 늘어난다.

좌표계
------
계획 CSV 도 맵 USD 도 **map 절대좌표**다 (origin -46, -31). 그래서 로봇을 계획의
t=0 자세에 **그대로** 놓는다. 학습은 시작 차체 기준으로 상대화했지만, 정책 관측은
전부 base 프레임이라 절대 위치는 정책에 보이지 않는다. 맵과 계획을 같은 좌표에
두는 편이 충돌 판정과 시각화가 훨씬 간단하다.
"""

from __future__ import annotations

import copy
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ee_track_ppo.assets.amr_fr3 import AMR_FR3_ARM_JOINTS
from ee_track_ppo.tasks.manager_based.reach import mdp
from ee_track_ppo.tasks.manager_based.reach.config.amr_fr3.traj_env_cfg import (
    AmrFr3TrajV29EnvCfg,
)

#: 두 로봇의 이름. 계획 CSV 의 base_A / base_B 와 짝이다.
SIDES = ("a", "b")
ASSET = {"a": "robot_a", "b": "robot_b"}
PRIM = {"a": "Robot_A", "b": "Robot_B"}

DEFAULT_MAP_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets", "munji_3f_2025_wide.usd")


@configclass
class TwinSceneCfg(InteractiveSceneCfg):
    """평지 + 압출한 맵 + 로봇 두 대.

    맵은 ``{ENV_REGEX_NS}`` 바깥 (``/World/map``) 에 둔다. env 를 복제해도 맵은
    한 벌만 있어야 한다 — 57x56 m 짜리를 env 마다 복제하면 메모리도 물리도 못 버틴다.
    (Phase 1 은 num_envs=1 이지만 규약을 미리 맞춰 둔다)
    """

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    # robot_a / robot_b / map 은 make_twin_cfg 가 채운다 (v29 설정에서 복사).


@configclass
class TwinObsCfg:
    """자리만 채우는 관측. **정책 입력이 아니다.**

    ManagerBasedEnv 가 관측 매니저를 요구하므로 최소한만 둔다. 정책이 먹는 52 차원은
    ``handoff/traj_v29/ee_obs.py`` 로 직접 만든다 (G1 이 등가성 확인).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        dummy = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_a", joint_names=AMR_FR3_ARM_JOINTS)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TwinEnvCfg(ManagerBasedEnvCfg):
    scene: TwinSceneCfg = TwinSceneCfg(num_envs=1, env_spacing=0.0)
    observations: TwinObsCfg = TwinObsCfg()
    actions: object = None          # make_twin_cfg 가 채운다


def make_twin_cfg(map_usd: str | None = DEFAULT_MAP_USD, num_envs: int = 1,
                  source_cfg=None, randomize: bool = False) -> TwinEnvCfg:
    """v29 학습 설정에서 **에셋·액션·물리를 복사**해 두 로봇 씬을 만든다.

    Args:
        map_usd: ``twin/world/map_extrude.py`` 산출물. None 이면 빈 평지
                 (기준선 비교용 — 맵이 원인인지 정책이 원인인지 가를 때 쓴다)
        source_cfg: 기본은 :class:`AmrFr3TrajV29EnvCfg`. 다른 판을 검증할 때 교체한다.
        randomize: 학습의 물성 랜덤화(마찰·질량)를 켠다. Phase 1 은 결정성을
                   위해 끈다 — 같은 계획을 두 번 돌리면 같은 수가 나와야 한다.
    """
    src = source_cfg if source_cfg is not None else AmrFr3TrajV29EnvCfg()

    cfg = TwinEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 0.0          # 맵이 절대좌표라 env 오프셋을 주면 안 된다

    # --- 물리 --- 학습과 **같은 값**이어야 액션 지연·램프의 시간 의미가 유지된다
    cfg.sim = src.sim
    cfg.decimation = src.decimation
    cfg.sim.render_interval = src.decimation

    # --- 로봇 두 대 --- v29 의 ArticulationCfg (페이로드 USD, 액추에이터, 기본자세)
    for s in SIDES:
        setattr(cfg.scene, ASSET[s],
                src.scene.robot.replace(prim_path="{ENV_REGEX_NS}/%s" % PRIM[s]))

    # --- 맵 --- env 바깥에 한 벌. USD 에 콜라이더가 이미 박혀 있다.
    if map_usd:
        if not os.path.exists(map_usd):
            raise FileNotFoundError(
                "맵 USD 가 없다: %s\n  먼저 만들 것: twin/run.sh twin/world/map_extrude.py"
                % map_usd)
        cfg.scene.map = AssetBaseCfg(
            prim_path="/World/map",
            spawn=sim_utils.UsdFileCfg(usd_path=map_usd),
        )

    # --- 액션 --- v29 의 항을 그대로, 이름만 바꿔 네 개로.
    #   arm  : DelayedJointPositionAction (지연 4~14 물리스텝)
    #   base : DiffDriveAction (한계·가감속 램프·이득 랜덤화)
    @configclass
    class _Actions:
        pass

    acts = _Actions()
    for s in SIDES:
        setattr(acts, "arm_%s" % s, src.actions.arm_action.replace(asset_name=ASSET[s]))
        setattr(acts, "base_%s" % s, src.actions.base_action.replace(asset_name=ASSET[s]))
    cfg.actions = acts

    # --- 이벤트 --- ★ 통째로 버리면 안 된다.
    #
    #   이 태스크는 **전역 중력이 0** 이다 (`sim.gravity = (0,0,0)`). 실기 FAIRINO
    #   컨트롤러가 팔 중력을 보상하므로 정책 입장의 팔은 무중력이어야 하고, 차체만
    #   `chassis_gravity` 이벤트가 외력으로 눌러 준다.
    #
    #   그래서 이벤트를 없애면 **아무 데도 중력이 없어** 로봇이 떠서 텀블링한다.
    #   (Phase 1 씬 스모크가 이걸 잡았다: 300 스텝에 z 0.01 -> 2.25 m, roll 159 deg)
    #
    #   가져오는 것은 chassis_gravity 뿐이다. 리셋 랜덤화(reset_base 등)는 계획
    #   시작자세를 흔들어 버리므로 빼고, 물성 랜덤화는 `randomize` 로 고를 수 있게 둔다.
    @configclass
    class _Events:
        pass

    evs = _Events()
    src_grav = src.events.chassis_gravity
    body_names = src_grav.params["asset_cfg"].body_names
    for s in SIDES:
        ev = copy.deepcopy(src_grav)
        ev.params = dict(ev.params)
        ev.params["asset_cfg"] = SceneEntityCfg(ASSET[s], body_names=body_names)
        setattr(evs, "chassis_gravity_%s" % s, ev)
    if randomize:
        for name in ("wheel_friction", "base_mass"):
            src_ev = getattr(src.events, name, None)
            if src_ev is None:
                continue
            for s in SIDES:
                ev = copy.deepcopy(src_ev)
                ev.params = dict(ev.params)
                old = src_ev.params["asset_cfg"]
                ev.params["asset_cfg"] = SceneEntityCfg(
                    ASSET[s], body_names=old.body_names)
                setattr(evs, "%s_%s" % (name, s), ev)
    cfg.events = evs
    return cfg


def action_layout(env) -> dict[str, slice]:
    """액션 벡터에서 각 항이 차지하는 구간. 순서를 손으로 가정하지 않는다.

    ``ActionManager`` 가 항을 등록한 순서가 곧 벡터 배치이고, 그것을 여기서
    **읽어서** 돌려준다. 하드코딩하면 항을 하나 추가하는 순간 조용히 어긋난다.
    """
    out, i = {}, 0
    for name, term in zip(env.action_manager.active_terms,
                          env.action_manager._terms.values()):
        n = term.action_dim
        out[name] = slice(i, i + n)
        i += n
    return out
