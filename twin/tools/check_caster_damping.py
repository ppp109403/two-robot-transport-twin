"""캐스터 감쇠 가설 — **명령 0 에서의 차체 자전이 캐스터 때문인가.**

무엇을 재는가
-------------
캐스터 조인트 감쇠를 env 마다 다르게 주고, 액션 0 으로 두었을 때의 요 표류를 잰다.
감쇠를 올려서 자전이 사라지면 원인이 캐스터로 확정된다.

왜 이게 정책 문제일 수 있는가
-----------------------------
현재 에셋은 캐스터가 ``stiffness 0 / damping 0 / velocity_limit 200`` 이다. 즉
저항 없이 상한까지 도는 바퀴 넷이 달려 있고, 차체는 구동륜 두 점으로만 지지된다.
그 결과 **명령이 0 인데 차체가 초당 몇 도씩 돈다** (학습 env 도 동일: 중앙 3.6 deg/s).

정책은 그 조건에서 학습됐으므로 자전을 상시 상쇄하도록 배웠을 가능성이 있다.
실기 바퀴는 정지 시 잠기므로 자전이 없고, 그러면 그 상쇄가 **불필요한 좌우 진동**으로
남는다 — v29 handoff 가 못 잡았다고 적은 "속도 출렁임 0.0593, 부호 반전 1.22 회/초" 다.

이 스크립트는 그 사슬의 **첫 고리**만 확인한다 (자전의 원인이 캐스터인가).
두 번째 고리(정책이 그걸 상쇄하는가)는 정책을 물려야 보이므로 재생 루프에서 잰다.

사용법
------
::

    twin/run.sh twin/tools/check_caster_damping.py --report /tmp/caster.txt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--damping", type=float, nargs="*",
                    default=[0.0, 0.001, 0.01, 0.05, 0.2, 1.0])
parser.add_argument("--effort", type=float, nargs="*", default=None,
                    help="감쇠와 짝지을 effort_limit. 기본 0.5 라 감쇠 토크가 잘린다")
parser.add_argument("--reps", type=int, default=4,
                    help="감쇠값당 env 수. 자전은 확률적이라 표본 하나로는 못 잰다")
parser.add_argument("--settle", type=int, default=300)
parser.add_argument("--steps", type=int, default=300)
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

from isaaclab.envs import ManagerBasedEnv  # noqa: E402

_TWIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _TWIN)
from world.scene_builder import ASSET, make_twin_cfg  # noqa: E402

LINES: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    LINES.append(m)


def main() -> bool:
    damps = list(args.damping)
    n = len(damps) * args.reps

    # 맵 없이, env 를 넉넉히 벌려 놓는다 — 이 검사는 차체 자체의 성질만 본다.
    cfg = make_twin_cfg(map_usd=None, num_envs=n)
    cfg.scene.env_spacing = 8.0
    env = ManagerBasedEnv(cfg=cfg)
    robot = env.scene[ASSET["a"]]

    jn = list(robot.data.joint_names)
    caster_wheel = [i for i, x in enumerate(jn)
                    if x.startswith("caster") and x.endswith("wheel_joint")]
    caster_all = [i for i, x in enumerate(jn) if x.startswith("caster")]

    say("=" * 76)
    say("캐스터 감쇠 가설 — 명령 0 에서의 차체 자전")
    say("=" * 76)
    say("  env %d 개 = 감쇠 %d 종 x %d 반복. 맵 없음, env_spacing %.1f m"
        % (n, len(damps), args.reps, cfg.scene.env_spacing))
    say("  캐스터 조인트 %d (바퀴 %d): %s"
        % (len(caster_all), len(caster_wheel), [jn[i] for i in caster_wheel]))

    env.reset()

    # env 별로 감쇠를 다르게 준다. 바퀴 회전 조인트에만 걸고 스위블은 건드리지 않는다
    # — 스위블까지 잠그면 캐스터가 방향을 못 틀어 주행 자체가 달라진다.
    grp = np.repeat(np.arange(len(damps)), args.reps)      # env -> 감쇠 인덱스
    efforts = args.effort if args.effort else [None] * len(damps)
    if len(efforts) != len(damps):
        raise SystemExit("--effort 는 --damping 과 같은 개수여야 한다")
    for i in range(n):
        eid = torch.tensor([i], device=env.device)
        val = torch.full((1, len(caster_wheel)), float(damps[grp[i]]), device=env.device)
        robot.write_joint_damping_to_sim(val, joint_ids=caster_wheel, env_ids=eid)
        e = efforts[grp[i]]
        if e is not None:
            # 감쇠 토크는 effort_limit 에서 잘린다. 기본값 0.5 N.m 로는 200 rad/s 로
            # 도는 바퀴를 세울 수 없으므로, 감쇠만 올린 검사는 결론이 안 난다.
            robot.write_joint_effort_limit_to_sim(
                torch.full((1, len(caster_wheel)), float(e), device=env.device),
                joint_ids=caster_wheel, env_ids=eid)

    # **먹었는지 읽어서 확인한다.** 조용히 무시되면 이 검사 전체가 무의미해진다.
    got = robot.root_physx_view.get_dof_dampings().to(env.device)[:, caster_wheel]
    ok_write = True
    for i in range(n):
        want = float(damps[grp[i]])
        if abs(float(got[i].min()) - want) > 1e-6 or abs(float(got[i].max()) - want) > 1e-6:
            ok_write = False
    say("  감쇠 기입 확인: %s  (읽은 값 %s)"
        % ("OK" if ok_write else "**실패 — 무시됐다**",
           np.round(got[:, 0].cpu().numpy(), 4).tolist()))

    # 초기조건을 env 마다 흔든다. 안 그러면 같은 감쇠의 env 들이 **완전히 동일하게**
    # 진화해 표본이 하나뿐인 것과 같다. 자전은 접촉 채터에서 오는 확률적 현상이라
    # 한 표본으로 순위를 매기면 잡음을 읽게 된다 (이 저장소가 이미 두 번 당했다).
    root = robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    rng = np.random.default_rng(0)
    for i in range(n):
        yaw = float(rng.uniform(-math.pi, math.pi))
        root[i, 3], root[i, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        root[i, 4] = root[i, 5] = 0.0
    robot.write_root_state_to_sim(root)
    env.sim.forward()

    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.settle):
        env.step(act)

    y0 = robot.data.heading_w.clone()
    p0 = robot.data.root_pos_w[:, :2].clone()
    for _ in range(args.steps):
        env.step(act)

    dt = args.steps * env.step_dt
    dyaw = (robot.data.heading_w - y0 + math.pi) % (2 * math.pi) - math.pi
    dxy = torch.norm(robot.data.root_pos_w[:, :2] - p0, dim=1)
    cw = robot.data.joint_vel[:, caster_wheel].abs().max(dim=1).values

    say("")
    say("  정착 %d + 측정 %d 스텝 (%.1f s), 액션 0" % (args.settle, args.steps, dt))
    say("-" * 76)
    say("  %-10s %12s %12s %12s %13s"
        % ("감쇠/토크", "|요|평균", "|요|최대", "xy 평균", "캐스터속도"))
    per = np.degrees(np.abs(dyaw.cpu().numpy())) / dt
    rates = []
    for gi, d in enumerate(damps):
        sel = np.where(grp == gi)[0]
        rates.append(float(per[sel].mean()))
        say("  %-10s %12.3f %12.3f %12.4f %13.1f"
            % ("%.4f/%s" % (d, efforts[gi] if efforts[gi] is not None else "-"),
               per[sel].mean(), per[sel].max(),
               float(dxy[sel].mean()), float(cw[sel].mean())))
    say("-" * 76)

    rates = np.array(rates)
    best = int(np.argmin(rates))
    say("  자전이 가장 작은 값: 감쇠 %.4f -> %.3f deg/s (감쇠 %.4f 대비 %.0f%% 감소)"
        % (damps[best], rates[best], damps[0],
           100.0 * (1.0 - rates[best] / max(rates[0], 1e-9))))
    say("")
    if rates[best] < 0.3 * rates[0] and rates[0] > 1.0:
        say("가설 확인 — 자전의 원인은 캐스터다. 감쇠만 올리면 사라진다.")
        say("  다음 질문: 정책이 이 자전을 상쇄하도록 학습됐는가.")
        say("  (재생 루프에서 차체 지령 부호반전을 감쇠 0 / %.4f 로 비교하면 보인다)"
            % damps[best])
    else:
        say("가설 미확정 — 감쇠만으로는 자전이 안 잡힌다. 다른 원인을 봐야 한다")
        say("  (캐스터 장착 높이 CASTER_MOUNT_Z, 접촉 마찰, 솔버 반복수 등)")
    say("=" * 76)
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
