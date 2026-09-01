"""게이트 G2 — **계획 보간 정합성**. Isaac 없이 도는 순수 numpy 검사.

무엇을 재는가
-------------
::

    [1] 규약 일치    twin 의 sample_step  ==  플래너 노드의 _sample_step (그대로 옮긴 것)
    [2] 강체 보존    보간한 두 EE 사이 거리가 항상 obj_len
    [3] 개별보간 오차 "틀린 방식" 이 막대를 얼마나 줄이는지 수치로 남긴다
    [4] 피드포워드   학습이 준 속도 규약 (선속도만, 각속도 0, 초반 정지)
    [5] CSV 왕복     실제 계획 CSV 가 있으면 그것으로도 [1][2] 를 확인

왜 [3] 을 굳이 재나
-------------------
"틀린 방식으로 해도 별 차이 없더라" 를 방지하기 위해서다. 오차가 무시할 만하면
이 게이트 자체가 의미 없고, 크면 **그 크기가 곧 가짜 막대오차의 크기**다. 나중에
Phase 1 에서 막대 길이 오차를 볼 때 "이건 정책 탓인가 보간 탓인가" 를 이 숫자로 가른다.

사용법
------
::

    python3 twin/tools/check_plan_sampler.py
    python3 twin/tools/check_plan_sampler.py --csv-dir /path/to/planner/csv_dir --obj-len 2.0

Isaac 을 안 띄우므로 ``twin/run.sh`` 없이 그냥 python3 로 돈다.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plan.plan_sampler import (  # noqa: E402
    PLANNER_NODE_MD5,
    Plan,
    REGIMES,
    naive_ee_lerp,
    synthetic_plan,
)

NODE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "planner", "mobile_manipulator_trajectory", "two_robot_nlp_node.py")


def node_sample_step_reference(plan: Plan, s: float) -> dict:
    """``two_robot_nlp_node._sample_step`` 을 **그대로 옮긴 것**.

    이것이 기준이다. twin 쪽 구현이 여기서 벗어나면 시뮬이 관제와 다른 목표를 재생한다.
    노드가 바뀌면 :data:`PLANNER_NODE_MD5` 검사가 먼저 경고한다.
    """
    N = plan.N
    s = min(max(s, 0.0), float(N))
    k = int(min(s, N - 1e-9))
    a = s - k
    k1 = min(k + 1, N)

    def lerp(arr):
        return (1.0 - a) * arr[k] + a * arr[k1]

    po = lerp(plan.obj_xy)
    tho = lerp(plan.obj_yaw)
    half = 0.5 * plan.obj_len * np.array([math.cos(tho), math.sin(tho)])

    return {
        "s": s,
        "t": s * plan.dt,
        "obj": po, "obj_yaw": tho,
        "ee": {"a": po - half, "b": po + half},
        "ee_yaw": {"a": tho, "b": tho + math.pi},
        "base": {"a": lerp(plan.base1_xy), "b": lerp(plan.base2_xy)},
        "base_yaw": {"a": lerp(plan.base1_yaw), "b": lerp(plan.base2_yaw)},
        "finished": s >= N,
    }


def feedforward(samples: list[dict], step_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """학습이 준 피드포워드 규약대로 EE 목표 속도를 만든다. (T,3) 선, (T,3) 각.

    ``MovingPoseWorldAmclCommand._update_from_plan`` 이 하는 것과 같아야 한다::

        target_lin_vel_w[:, :2] = (ee_xy - prev_ee_xy) / step_dt      z 성분은 건드리지 않음
        target_ang_vel_w        = **한 번도 안 씀 -> 0 그대로**
        started = _ep_t > step_dt * 1.5                                초반은 0

    각속도가 0 인 것은 실수가 아니라 학습된 계약이다. 여기에 0 이 아닌 값을 넣으면
    정책은 한 번도 본 적 없는 입력을 받는다.
    """
    lin = np.zeros((len(samples), 3))
    ang = np.zeros((len(samples), 3))
    for i in range(1, len(samples)):
        if (i * step_dt) <= step_dt * 1.5:
            continue
        lin[i, :2] = (samples[i]["ee"]["a"] - samples[i - 1]["ee"]["a"]) / step_dt
    return lin, ang


def main() -> bool:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", dest="csv_dir", default="",
                    help="two_robot_nlp_node 의 csv_dir 산출물 (있으면 그것도 검사)")
    ap.add_argument("--obj-len", dest="obj_len", type=float, default=2.0)
    ap.add_argument("--rate", type=float, default=30.0, help="제어주기 [Hz]")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def say(m: str = ""):
        print(m, flush=True)
        lines.append(m)

    ok = True

    def check(tag: str, good: bool, detail: str = ""):
        nonlocal ok
        ok &= good
        say("  %-46s %s%s" % (tag, "OK" if good else "**실패**",
                              ("   " + detail) if detail else ""))

    say("=" * 76)
    say("G2 계획 보간 정합성")
    say("=" * 76)

    # --- [0] 플래너 노드가 그 사이 바뀌지 않았는가 --------------------------
    if os.path.exists(NODE_PATH):
        got = hashlib.md5(open(NODE_PATH, "rb").read()).hexdigest()
        same = got == PLANNER_NODE_MD5
        check("[0] 플래너 노드 md5 (기준 %s)" % PLANNER_NODE_MD5[:8], same, got[:8])
        if not same:
            say("      노드가 바뀌었다. _sample_step 을 다시 대조하고 "
                "plan_sampler.PLANNER_NODE_MD5 를 갱신할 것.")
    else:
        say("  [0] 플래너 노드 없음 (%s) — md5 검사 건너뜀" % NODE_PATH)

    # --- [1] 규약 일치 -----------------------------------------------------
    plan = synthetic_plan(obj_len=args.obj_len)
    say("")
    say("합성 계획: N=%d, dt=%.3f s, %.1f s, obj_len=%.2f m, 회전 포함"
        % (plan.N, plan.dt, plan.duration, plan.obj_len))
    say("")

    grid = np.linspace(0.0, float(plan.N), 977)      # 격자점·경계·중간 모두 밟는다
    worst = 0.0
    for s in grid:
        mine, ref = plan.sample_step(s), node_sample_step_reference(plan, s)
        for key in ("obj", "obj_yaw"):
            worst = max(worst, float(np.max(np.abs(np.asarray(mine[key]) - np.asarray(ref[key])))))
        for key in ("ee", "ee_yaw", "base", "base_yaw"):
            for side in ("a", "b"):
                worst = max(worst, float(np.max(np.abs(
                    np.asarray(mine[key][side]) - np.asarray(ref[key][side])))))
    check("[1] 노드 _sample_step 과 일치 (977 점)", worst <= args.tol, "최대 %.3e" % worst)

    # --- [2] 강체 보존 -----------------------------------------------------
    d = np.array([float(np.linalg.norm(plan.sample_step(s)["ee"]["b"]
                                       - plan.sample_step(s)["ee"]["a"])) for s in grid])
    err = float(np.max(np.abs(d - plan.obj_len)))
    check("[2] 막대 길이 보존", err <= 1e-12, "최대 편차 %.3e m" % err)

    # --- [3] 개별보간이 막대를 얼마나 줄이나 -------------------------------
    #   축소량 ~= (obj_len/2) * (1 - cos(w*dt/2)) 이므로 회전율의 제곱으로 커진다.
    #   그래서 하나의 숫자가 아니라 **체제별로** 재야 의미가 있다.
    say("")
    say("[3] EE 개별보간(틀린 방식)의 막대 축소 — 이 값이 곧 '가짜 막대오차'")
    say("      %-14s %-18s %10s %10s" % ("체제", "(obj_v, obj_w)", "중앙", "최대"))
    for name, (ov, ow) in REGIMES.items():
        p = synthetic_plan(obj_len=args.obj_len, turn_rate=ow, speed=ov)
        g = np.linspace(0.0, float(p.N), 977)
        sh = p.obj_len - np.array(
            [float(np.linalg.norm(np.subtract(*reversed(naive_ee_lerp(p, s))))) for s in g])
        say("      %-14s (%.2f, %.2f)       %8.3f mm %8.3f mm"
            % (name, ov, ow, float(np.median(sh)) * 1e3, float(np.max(sh)) * 1e3))
    say("      (참고: v29 의 EE 추종오차 중앙이 9.5 mm. 같은 자릿수면 "
        "보간 방식만으로 판정이 뒤집힌다)")

    # --- [4] 피드포워드 규약 -----------------------------------------------
    step_dt = 1.0 / args.rate
    smp = plan.resample(args.rate)
    lin, ang = feedforward(smp, step_dt)
    say("")
    check("[4a] 각속도 피드포워드 = 0 (학습 계약)", float(np.max(np.abs(ang))) == 0.0)
    check("[4b] 선속도 z 성분 = 0", float(np.max(np.abs(lin[:, 2]))) == 0.0)
    check("[4c] 초반 1.5 스텝 정지", float(np.max(np.abs(lin[:2]))) == 0.0)
    sp = np.linalg.norm(lin[:, :2], axis=1)
    say("      EE 목표 속력  중앙 %.3f  95%% %.3f  최대 %.3f m/s"
        % (float(np.median(sp)), float(np.percentile(sp, 95)), float(np.max(sp))))
    say("      (v29 권장 ee_v_max 0.20. 넘으면 플래너 파라미터를 먼저 낮출 것)")

    # --- [5] 실제 CSV --------------------------------------------------------
    if args.csv_dir:
        say("")
        say("[5] 실제 계획 CSV: %s" % args.csv_dir)
        try:
            real = Plan.from_csv_dir(args.csv_dir, obj_len=args.obj_len)
        except Exception as exc:  # noqa: BLE001
            check("[5] CSV 로드", False, str(exc))
        else:
            say("      N=%d  dt=%.3f s  %.1f s" % (real.N, real.dt, real.duration))
            # 정수 스텝에서는 재유도 EE 가 CSV EE 와 같아야 한다.
            # 다르면 obj_len 설정이나 요 규약이 어긋난 것이다.
            de = []
            for k in range(real.N + 1):
                m = real.sample_step(float(k))
                de.append(np.linalg.norm(m["ee"]["a"] - real.ee1_xy_csv[k]))
                de.append(np.linalg.norm(m["ee"]["b"] - real.ee2_xy_csv[k]))
            de = float(np.max(de))
            # 허용치는 상수가 아니라 **CSV 포맷 정밀도**에서 유도한다.
            # two_robot_nlp_node._export_csv 는 x,y 를 '%.5f' (반올림 ±5e-6),
            # yaw 를 '%.6f' (±5e-7 rad) 로 쓴다. 그리고 물체 계열과 EE 계열이
            # **각각 독립적으로** 반올림되므로 두 몫을 모두 더해야 한다:
            #
            #   |재유도 - CSV| <= |물체 반올림| + (obj_len/2)|요 반올림| + |EE 반올림|
            #                  =  5e-6*sqrt2   +      1.0 * 5e-7       +  5e-6*sqrt2
            q = 5e-6 * math.sqrt(2.0)
            tol5a = 2.0 * q + (real.obj_len / 2.0) * 5e-7
            check("[5a] 정수 스텝에서 재유도 EE == CSV EE", de <= tol5a,
                  "최대 %.3e m (CSV 양자화 한계 %.3e)" % (de, tol5a))
            # 분해해서 남긴다 — 이 값들이 커지면 CSV 반올림이 아니라 NLP 의
            # 막대 부착 제약 잔차가 커진 것이고, 그때는 계획 자체를 의심해야 한다.
            mid = 0.5 * (real.ee1_xy_csv + real.ee2_xy_csv)
            bar = np.linalg.norm(real.ee2_xy_csv - real.ee1_xy_csv, axis=1)
            say("      분해: 중점-물체 %.2e m   막대길이 편차 %.2e m   (양자화 %.2e)"
                % (float(np.linalg.norm(mid - real.obj_xy, axis=1).max()),
                   float(np.abs(bar - real.obj_len).max()), q))
            g = np.linspace(0.0, float(real.N), 2000)
            dd = np.array([float(np.linalg.norm(real.sample_step(s)["ee"]["b"]
                                                - real.sample_step(s)["ee"]["a"])) for s in g])
            check("[5b] 막대 길이 보존", float(np.max(np.abs(dd - real.obj_len))) <= 1e-12)
            sn = np.array([float(np.linalg.norm(np.subtract(*reversed(naive_ee_lerp(real, s)))))
                           for s in g])
            say("      개별보간 축소  중앙 %6.2f mm  최대 %6.2f mm"
                % (float(np.median(real.obj_len - sn)) * 1e3,
                   float(np.max(real.obj_len - sn)) * 1e3))
    else:
        say("")
        say("[5] 실제 CSV 미지정 — 플래너에 csv_dir 를 주고 한 판 풀어 두면 "
            "이 게이트가 실제 궤적으로도 돈다")

    say("")
    say("G2 %s" % ("통과" if ok else "실패 — twin 리그를 여기서 멈춘다"))
    say("=" * 76)

    if args.report:
        with open(args.report, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
