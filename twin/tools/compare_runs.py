"""같은 계획에 대한 여러 러너의 결과를 **나란히** 비교한다 — 표 + 경로 그림.

왜 별도 도구인가
----------------
``run_classic.py`` 와 ``run_episode.py`` 는 각자 자기 리포트를 낸다. 그런데 판정에
필요한 것은 "둘 중 무엇이 더 나은가" 가 아니라 **무엇을 대가로 무엇을 얻었는가** 다.
정책은 EE 를 얻고 차체를 잃는데, 잃은 차체가 **계획이 보장한 충돌 여유를 먹는지**는
두 로그를 같은 SDF 에 올려 봐야 안다.

Isaac 이 필요 없다
------------------
저장된 ``log.npz`` 와 원본 맵만 쓴다. 그래서 Kit 기동 없이 몇 초 만에 돈다::

    <casadi/scipy 있는 python> twin/tools/compare_runs.py \\
        --plan twin/data/plans/L150_base018_obj005_cmp \\
        --runs 고전=twin/data/runs/cmp_classic 정책=twin/data/runs/cmp_policy \\
        --out /tmp/cmp.png --report /tmp/cmp.txt

무엇을 재나
-----------
::

    EE 오차 / 자세 오차       추종 정확도
    차체 편차                 계획에서 얼마나 벗어났나
    실제 충돌 여유            ★ 실행 궤적을 원본 SDF 로 다시 잰 것.
                              계획이 보장한 값이 실행까지 살아남았는가
    로봇-로봇 최소거리        계획의 robot_robot_min 제약이 지켜졌는가

색
--
계열이 셋(레퍼런스 + 러너 둘)이라 검증된 카테고리 팔레트의 **첫 세 슬롯**을 쓴다.
이 셋은 all-pairs 기준을 두 모드에서 통과하는 조합이다. 로봇 A/B 는 색이 아니라
**선 모양**으로 가른다 (색은 러너 정체성에만 쓴다).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "world"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "planner"))

import solve_plan as sp  # noqa: E402
from mobile_manipulator_trajectory.sdf_utils import SignedDistanceField  # noqa: E402
from mobile_manipulator_trajectory.two_robot_nlp import pose_clearance  # noqa: E402
from robot_geom import BASE_LEN, BASE_WID, N_BASE_DISKS  # noqa: E402

#: 검증된 카테고리 팔레트 슬롯 1~3 (light). 레퍼런스, 러너1, 러너2 순.
#: 이 셋은 all-pairs 기준을 두 모드에서 통과하는 조합이다.
HUES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c5a", "#d8d8d5"


def base_slack(sdf, xy: np.ndarray, yaw: np.ndarray, margin: float,
               inflation: float = sp.MAP_INFLATION) -> np.ndarray:
    """차체 자세열의 여유 슬랙. **플래너의 ``pose_clearance`` 를 그대로 쓴다.**

    직접 ``sdf(중심) - 반경`` 으로 재면 안 된다. 플래너는 사각형을 원판 여러 개로
    덮고 그 **최솟값**을 보므로, 중심만 재면 길이 방향으로 뻗은 부분이 벽에 걸리는
    것을 놓친다. 반경을 손으로 적는 것도 금물이다 — 기하가 바뀌면 조용히 어긋난다.

    ``map_inflation`` 을 빼는 이유
    ------------------------------
    2D 라이다 SLAM 맵은 벽을 실제보다 두껍게 그린다 (munji 는 실측 벽 두께 중앙
    0.100 m = 두 칸). 플래너는 그만큼을 **요구 여유에서 빼서** 푼다::

        요구:  sdf >= r + margin - map_inflation

    이걸 빼먹고 재면 모든 슬랙이 일괄 ``inflation`` 만큼 작게 나오고, 계획이
    성립이라고 한 궤적이 재검에서 위반으로 보인다 (실제로 처음에 그랬다).

    반환값이 음수면 계획이 잡은 여유를 먹었다는 뜻이다.
    """
    eff = margin - max(0.0, inflation)
    return np.array([
        pose_clearance(sdf, (float(x), float(y), float(t)),
                       BASE_LEN / 2.0, BASE_WID / 2.0, N_BASE_DISKS, eff)
        for (x, y), t in zip(xy, yaw)])


def load_sdf(map_yaml: str):
    """맵 -> (SDF, 장애물 마스크, meta).

    ``solve_plan.build_grid`` 을 **그대로 쓴다.** 미지영역 처리와 상하 뒤집기가
    거기 들어 있고, 여기서 다시 적으면 계획을 푼 것과 다른 SDF 로 재검하게 된다 —
    그러면 비교 자체가 무의미해진다.
    """
    data, meta, (free, obst) = sp.build_grid(map_yaml)
    sdf = SignedDistanceField.from_occupancy(
        data, origin=(meta["origin"][0], meta["origin"][1]),
        res=float(meta["resolution"]), unknown_is_occupied=True,
        smooth_sigma_cells=1.0)
    return sdf, obst, meta


def plan_paths(plan_dir: str) -> dict:
    out = {}
    for key, fn in (("a", "base_A.csv"), ("b", "base_B.csv"), ("obj", "object.csv")):
        p = os.path.join(plan_dir, fn)
        if os.path.exists(p):
            out[key] = np.loadtxt(p, delimiter=",", skiprows=1)
    return out


def stats(v: np.ndarray) -> tuple:
    return float(np.median(v)), float(np.percentile(v, 95)), float(v.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="계획 CSV 디렉토리")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="이름=런디렉토리 를 여러 개. 예: 고전=... 정책=...")
    ap.add_argument("--map", default="handoff/map_munji_3f_2025_wide/munji_3f_2025_wide.yaml")
    ap.add_argument("--margin", type=float, default=0.18,
                    help="계획에 쓴 base_margin. 침범 판정 기준")
    ap.add_argument("--map-inflation", dest="inflation", type=float, default=sp.MAP_INFLATION,
                    help="맵의 벽 과대묘사 보정 [m]. 플래너가 요구 여유에서 빼는 값과 같아야 한다")
    ap.add_argument("--rr-min", dest="rr_min", type=float, default=0.90,
                    help="계획의 robot_robot_min")
    ap.add_argument("--out", default="", help="경로 그림 PNG")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def say(m: str = "") -> None:
        print(m, flush=True)
        lines.append(m)

    sdf, occ, info = load_sdf(args.map)
    ref = plan_paths(args.plan)
    runs = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        runs[name] = (np.load(os.path.join(d, "log.npz")), d)

    say("=" * 78)
    say("러너 비교 — 계획 %s" % os.path.basename(args.plan.rstrip("/")))
    say("=" * 78)
    say("  맵 %s   base_margin %.2f   map_inflation %.2f   robot_robot_min %.2f"
        % (os.path.basename(args.map), args.margin, args.inflation, args.rr_min))
    say("  러너 " + ",  ".join("%s=%s" % (k, os.path.basename(v[1].rstrip("/")))
                              for k, v in runs.items()))

    # ---- 계획이 보장한 여유 (레퍼런스 경로를 같은 SDF 로 재검) -------------
    say("")
    say("계획이 보장한 것 (레퍼런스 차체 경로를 같은 SDF 로 재검산)")
    say("-" * 78)
    say("  차체 %.2f x %.2f m 를 원판 %d 개로 덮음 (플래너와 같은 기하)"
        % (BASE_LEN, BASE_WID, N_BASE_DISKS))
    for s in ("a", "b"):
        if s not in ref:
            continue
        sl = base_slack(sdf, ref[s][:, 1:3], ref[s][:, 3], args.margin, args.inflation)
        say("  레퍼런스 %s   최소 슬랙 %+.3f m   %s"
            % (s.upper(), sl.min(),
               "(margin 안에서 성립)" if sl.min() >= 0 else "★ 계획 자체가 margin 을 먹는다"))

    # ---- 계획 자체의 매끄러움 ----------------------------------------------
    #
    # 추종이 나쁠 때 제어기를 의심하기 전에 **계획이 애초에 매끄러운지** 봐야 한다.
    # 차체 경로가 직선 대비 두 배로 구불거리면, 같은 시간에 두 배를 달려야 하므로
    # 속도 한계에 부딪히고 그 결과가 추종오차로 나타난다 — 제어기 탓이 아니다.
    say("")
    say("계획 자체의 매끄러움 (제어와 무관, CSV 만 본다)")
    say("-" * 78)
    say("  %-10s %10s %10s %12s %12s" % ("경로", "경로길이", "직선거리", "구불정도", "진동95%"))
    for key, tag in (("obj", "물체"), ("a", "차체 A"), ("b", "차체 B")):
        if key not in ref:
            continue
        xy = ref[key][:, 1:3]
        L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        D = float(np.linalg.norm(xy[-1] - xy[0]))
        win = max(3, int(round(4.0 / max(1e-6, np.median(np.diff(ref[key][:, 0]))))) | 1)
        k = np.ones(win) / win
        sm = np.stack([np.convolve(xy[:, i], k, mode="same") for i in range(2)], axis=1)
        e = win // 2
        osc = np.linalg.norm(xy - sm, axis=1)[e:-e] if len(xy) > 2 * win else np.zeros(1)
        say("  %-10s %9.2f m %9.2f m %10.2f 배 %9.0f mm%s"
            % (tag, L, D, L / max(D, 1e-6), 1e3 * np.percentile(osc, 95),
               "  ★" if L / max(D, 1e-6) > 1.5 else ""))
    say("  (구불정도 = 경로길이/직선거리.  물체는 매끄러운데 차체만 크면 NLP 가")
    say("   리치 여유(reach_min~reach_max) 안에서 차체를 흔든 것이다 — w_base_vel 을 올릴 것)")

    # ---- 추종 정확도 -------------------------------------------------------
    say("")
    say("추종 정확도")
    say("-" * 78)
    say("  %-8s %-4s %10s %10s %10s   %10s %10s"
        % ("러너", "로봇", "EE중앙", "EE95%", "EE최대", "자세중앙", "자세95%"))
    for name, (L, _) in runs.items():
        for s in ("a", "b"):
            e = L["ee_err_%s" % s].ravel() * 1e3
            o = L["ori_err_%s" % s].ravel()
            say("  %-8s %-4s %9.1f  %9.1f  %9.1f   %9.2f  %9.2f"
                % (name, s.upper(), *stats(e), *stats(o)[:2]))
    say("  (EE 단위 mm, 자세 deg)")

    # ---- 차체 편차와 그 대가 ----------------------------------------------
    say("")
    say("차체 편차와 그 대가 — ★ 계획의 충돌 보장이 실행까지 살아남았는가")
    say("-" * 78)
    say("  %-8s %-4s %10s %10s   %11s %10s %12s"
        % ("러너", "로봇", "편차중앙", "편차95%", "슬랙최소", "슬랙5%", "margin침범"))
    for name, (L, _) in runs.items():
        for s in ("a", "b"):
            xy = L["base_%s" % s][:, :2]
            yaw = L["base_yaw_%s" % s].ravel()
            dev = np.linalg.norm(xy - L["base_ref_%s" % s][:, :2], axis=1)
            sl = base_slack(sdf, xy, yaw, args.margin, args.inflation)
            say("  %-8s %-4s %9.3f  %9.3f   %+10.3f %+9.3f %11.1f%%"
                % (name, s.upper(), np.median(dev), np.percentile(dev, 95),
                   sl.min(), np.percentile(sl, 5), 100.0 * float(np.mean(sl < 0.0))))
    say("  (단위 m.  슬랙 = min(SDF) − 원판반경 − margin + map_inflation.  음수면 침범)")

    # ---- 로봇-로봇 --------------------------------------------------------
    say("")
    say("로봇-로봇 거리 — 계획 제약 %.2f m" % args.rr_min)
    say("-" * 78)
    for name, (L, _) in runs.items():
        rd = L["robot_dist"].ravel()
        n = len(rd)
        say("  %-8s 최소 %.3f m   중앙 %.3f   %.2f 미만 %5.1f%% 의 스텝  %s"
            % (name, rd.min(), np.median(rd), args.rr_min,
               100.0 * float(np.mean(rd < args.rr_min)),
               "★ 제약 위반" if rd.min() < args.rr_min else "지켜짐"))

    if args.out:
        draw(args.out, sdf, occ, info, ref, runs, say)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write("\n".join(lines) + "\n")


def draw(out_path: str, sdf, occ, info, ref: dict, runs: dict, say) -> None:
    """계획 경로 + 각 러너의 실제 주행 경로를 한 그림에.

    색 = 러너 정체성, 선모양 = 로봇 A/B. 색으로 로봇을 가르면 러너와 로봇이
    같은 채널을 두고 다투게 된다 — 복합 부호화로 채널을 나눈다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.lines import Line2D

    # 한글 라벨. DejaVu 에는 한글 글리프가 없어 네모로 나온다. CJK 폰트를 앞에 세우고,
    # 없으면 라벨을 영문으로 떨어뜨린다 (깨진 글자를 내보내는 것보다 낫다).
    have = {f.name for f in fm.fontManager.ttflist}
    ko = next((n for n in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic",
                           "Malgun Gothic", "AppleGothic") if n in have), None)
    if ko:
        plt.rcParams["font.family"] = [ko, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    T = (dict(title="차체 주행 경로 — 계획 대비 실제", ref="계획 (레퍼런스)",
              ra="로봇 A", rb="로봇 B", start="시작", goal="목표") if ko else
         dict(title="Base paths - plan vs executed", ref="plan (reference)",
              ra="robot A", rb="robot B", start="start", goal="goal"))

    res = info["resolution"]
    ox, oy = info["origin"][0], info["origin"][1]
    ny, nx = occ.shape
    extent = [ox, ox + nx * res, oy, oy + ny * res]

    # 그릴 범위 = 모든 경로의 bbox + 여백
    pts = [ref[s][:, 1:3] for s in ("a", "b") if s in ref]
    for L, _ in runs.values():
        pts += [L["base_a"][:, :2], L["base_b"][:, :2]]
    P = np.vstack(pts)
    pad = 2.0
    xlim = (P[:, 0].min() - pad, P[:, 0].max() + pad)
    ylim = (P[:, 1].min() - pad, P[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 벽은 배경이다 — 옅게, 데이터보다 뒤로
    ax.imshow(np.where(occ > 0, 1.0, np.nan), extent=extent, origin="lower",
              cmap=matplotlib.colors.ListedColormap(["#c4c4c0"]),
              interpolation="nearest", zorder=0)

    styles = {"a": "-", "b": "--"}
    # 레퍼런스 먼저(뒤에), 러너를 그 위에
    for s in ("a", "b"):
        if s in ref:
            ax.plot(ref[s][:, 1], ref[s][:, 2], styles[s], color=HUES[0],
                    lw=2.0, zorder=2, solid_capstyle="round")
    for i, (name, (L, _)) in enumerate(runs.items()):
        c = HUES[1 + i % (len(HUES) - 1)]
        for s in ("a", "b"):
            xy = L["base_%s" % s][:, :2]
            ax.plot(xy[:, 0], xy[:, 1], styles[s], color=c, lw=2.0, zorder=3 + i,
                    solid_capstyle="round")

    # 시작·목표는 레퍼런스 기준 한 번만
    for s in ("a", "b"):
        if s not in ref:
            continue
        ax.plot(*ref[s][0, 1:3], "o", ms=9, mfc="white", mec=HUES[0], mew=2.0, zorder=6)
        ax.plot(*ref[s][-1, 1:3], "s", ms=9, mfc=HUES[0], mec="white", mew=1.5, zorder=6)
    ax.annotate(T["start"], ref["a"][0, 1:3], textcoords="offset points", xytext=(10, 8),
                color=INK_MUTED, fontsize=10)
    ax.annotate(T["goal"], ref["a"][-1, 1:3], textcoords="offset points", xytext=(10, 8),
                color=INK_MUTED, fontsize=10)

    handles = [Line2D([], [], color=HUES[0], lw=2.0, label=T["ref"])]
    for i, name in enumerate(runs):
        handles.append(Line2D([], [], color=HUES[1 + i % (len(HUES) - 1)], lw=2.0,
                              label=name))
    handles += [Line2D([], [], color=INK_MUTED, lw=1.6, ls="-", label=T["ra"]),
                Line2D([], [], color=INK_MUTED, lw=1.6, ls="--", label=T["rb"])]
    leg = ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.95,
                    edgecolor=GRID, fontsize=10, labelcolor=INK)
    leg.get_frame().set_linewidth(0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (map)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("y [m]  (map)", color=INK_MUTED, fontsize=10)
    ax.set_title(T["title"], color=INK, fontsize=13, pad=12)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    say("")
    say("  그림 %s" % out_path)


if __name__ == "__main__":
    main()
