"""2D SLAM 맵(.pgm/.yaml) -> 3D 압출 월드 + **통과 가능성 분석**.

두 가지를 한다
--------------
::

    [분석]  이 맵에서 어느 정도의 여유(margin)까지 실제로 통행 가능한가
    [압출]  벽을 높이 2 m 로 세운 USD 메시 + 정적 콜라이더

분석이 먼저인 이유는, v29 handoff 가 요구한 ``base_margin >= 0.70`` 이 이 맵에서
성립하는지가 **시뮬을 짓기 전에 답이 나오는 질문**이기 때문이다.

필요 여유거리는 margin 이 아니다
--------------------------------
플래너는 사각형을 원판으로 덮어 SDF 와 비교한다
(``two_robot_nlp.cover_rect_with_disks``)::

    r = hypot(half_wid, half_len / n)

    base   0.80 x 0.55, n=3  ->  r = 0.306 m
    object 2.00 x 0.30, n=5  ->  r = 0.250 m

그리고 ``pose_clearance`` 는 ``sdf - r - margin`` 을 본다. 즉 **필요 SDF 여유는
``r + margin``** 이다. margin 만 보고 판단하면 0.306 m 를 통째로 빠뜨린다.

미지영역(205) 문제
------------------
``munji_3f_2025_wide`` 는 **90.96 % 가 픽셀 205(미지)** 다. 그런데 이 맵의 yaml 은
``free_thresh: 0.25`` 이고 205 의 점유도는 0.196 이므로, ROS ``map_server`` 규칙
(``occ < free_thresh -> free``)에 따라 **미지가 자유공간으로 발행된다.**

그 결과 플래너의 ``unknown_is_occupied: True`` 는 **이 맵에서 아무 일도 하지 않는다** —
``/map`` 에 -1 이 하나도 안 실려 오기 때문이다. A* 가 건물 바깥 미지영역을 자유롭게
가로지르는 계획을 낼 수 있다.

그래서 여기서는 **원본 pgm 에서 직접** 분류하고 (thresholded grid 가 아니라),
미지를 장애물로 볼지 고를 수 있게 둔다. 기본값은 ``unknown_is_occupied=True`` —
매핑되지 않은 곳을 로봇이 주행할 수 있다고 가정하면 안 되기 때문이다.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
from scipy import ndimage as ndi

#: 픽셀 -> 점유도. ROS map_server 규칙 (negate=0): occ = (255 - px) / 255
FREE_OCC_MAX = 0.02      # 확실한 자유공간 (픽셀 >= 250)
WALL_OCC_MIN = 0.65      # occupied_thresh


def read_pgm(path: str) -> np.ndarray:
    """P5 (binary) PGM -> (h, w) uint8. 주석 줄을 건너뛴다."""
    raw = open(path, "rb").read()
    pos, fields = 0, []
    while len(fields) < 4:
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b"#":            # 공백을 먼저 건너뛴 뒤에 주석 판정
            pos = raw.index(b"\n", pos) + 1
            continue
        m = re.match(rb"\S+", raw[pos:])
        fields.append(m.group(0))
        pos += m.end()
    pos += 1
    magic, w, h = fields[0].decode(), int(fields[1]), int(fields[2])
    if magic != "P5":
        raise ValueError("P5(binary) PGM 만 지원한다: %s" % magic)
    return np.frombuffer(raw[pos:pos + w * h], dtype=np.uint8).reshape(h, w)


def read_map_yaml(path: str) -> dict:
    """map_server yaml 의 최소 파싱. 주석(#)과 리스트 origin 만 다룬다."""
    out = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("["):
            out[k.strip()] = [float(x) for x in v.strip("[]").split(",")]
        else:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def classify(img: np.ndarray, free_occ_max=FREE_OCC_MAX, wall_occ_min=WALL_OCC_MIN):
    """(free, wall, unknown) 불리언 마스크. **원본 픽셀에서** 직접 나눈다.

    map_server 가 발행한 OccupancyGrid 를 쓰지 않는 이유는 모듈 docstring 참고 —
    이 맵의 free_thresh 가 미지를 자유공간으로 만들어 버린다.
    """
    occ = (255.0 - img.astype(np.float64)) / 255.0
    free = occ <= free_occ_max
    wall = occ >= wall_occ_min
    return free, wall, ~free & ~wall


def clean(mask: np.ndarray, open_px: int = 1, close_px: int = 2) -> np.ndarray:
    """SLAM 잡티 제거. **닫기 먼저, 열기 나중**.

    순서가 중요하다. 열기를 먼저 하면 1 px 두께로 스캔된 얇은 벽이 통째로 지워져
    **없는 문이 생긴다**. 닫기로 벽의 구멍을 먼저 메운 뒤 튀는 점을 지운다.
    """
    out = mask
    if close_px > 0:
        out = ndi.binary_closing(out, np.ones((close_px * 2 + 1,) * 2))
    if open_px > 0:
        out = ndi.binary_opening(out, np.ones((open_px * 2 + 1,) * 2))
    return out


def rectangles(mask: np.ndarray) -> np.ndarray:
    """점유 마스크 -> 축정렬 사각형 목록 (K, 4) = y0, y1, x0, x1 (반열림).

    행별 런을 만든 뒤 위아래로 **같은 구간이면 이어 붙인다**. 최적 분해는 아니지만
    벽 위주 맵에서는 셀 수를 두세 자릿수로 줄여 준다. 그리고 결정적이다.
    """
    h, w = mask.shape
    active: dict[tuple[int, int], int] = {}     # (x0, x1) -> y0
    out = []
    for y in range(h + 1):
        runs = {}
        if y < h:
            row = mask[y]
            idx = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
            for x0, x1 in zip(idx[0::2], idx[1::2]):
                runs[(int(x0), int(x1))] = True
        for key, y0 in list(active.items()):
            if key not in runs:
                out.append((y0, y, key[0], key[1]))
                del active[key]
        for key in runs:
            active.setdefault(key, y)
    return np.array(out, dtype=np.int64).reshape(-1, 4)


def analyse(free: np.ndarray, obst: np.ndarray, res: float, geom: dict, say) -> dict:
    """주행 가능 영역의 여유거리와 **margin 별 통행 가능성**.

    핵심 질문: margin 을 키우면 맵이 어디서 끊기는가. 면적만 보면 안 된다 —
    면적이 40 % 남아도 조각조각 끊겨 있으면 방 사이를 못 간다.
    """
    lab, n = ndi.label(free, structure=np.ones((3, 3)))
    if n == 0:
        raise SystemExit("자유공간이 없다. 임계값을 확인할 것")
    sizes = ndi.sum(free, lab, range(1, n + 1))
    big = lab == (int(np.argmax(sizes)) + 1)
    area = float(big.sum()) * res * res

    clear = ndi.distance_transform_edt(~obst, sampling=res)
    c = clear[big]

    say("")
    say("주행 가능 영역 (매핑된 자유공간 최대 연결성분)")
    say("-" * 76)
    say("  면적           %.1f m^2   (자유공간 %d 성분 중 최대, 전체의 %.1f%%)"
        % (area, n, 100.0 * big.sum() / max(1, free.sum())))
    say("  여유거리       중앙 %.2f   95%% %.2f   최대 %.2f m"
        % (float(np.median(c)), float(np.percentile(c, 95)), float(c.max())))

    say("")
    say("margin 별 통행 가능성 — 필요 여유 = 디스크반경 + margin")
    say("-" * 76)
    say("  %-8s %-10s %10s %8s %12s   %s"
        % ("대상", "margin", "필요여유", "면적%", "최대성분", "판정"))
    rows = []
    for tag, r_disk, margins in (("base", geom["base_r"], (0.05, 0.20, 0.30, 0.50, 0.70)),
                                 ("object", geom["obj_r"], (0.05, 0.20))):
        for mg in margins:
            need = r_disk + mg
            m = big & (clear >= need)
            if not m.any():
                rows.append((tag, mg, need, 0.0, 0.0, 0))
                say("  %-8s %-10.2f %8.3f m %7.1f%% %12s   통행 불가"
                    % (tag, mg, need, 0.0, "-"))
                continue
            l2, n2 = ndi.label(m, structure=np.ones((3, 3)))
            s2 = ndi.sum(m, l2, range(1, n2 + 1))
            frac = 100.0 * m.sum() / big.sum()
            biggest = float(s2.max()) * res * res
            share = biggest / area
            verdict = ("연결 양호" if share > 0.60 else
                       "분절 주의" if share > 0.25 else "**분절 심함**")
            rows.append((tag, mg, need, frac, biggest, n2))
            say("  %-8s %-10.2f %8.3f m %7.1f%% %7.1f m^2(%d)   %s"
                % (tag, mg, need, frac, biggest, n2, verdict))
    say("-" * 76)
    return {"area": area, "clear": clear, "big": big, "rows": rows}


def build_mesh(rects: np.ndarray, res: float, origin, height: float):
    """사각형 목록 -> 삼각형 메시 (points, faceVertexIndices).

    상자마다 8 정점 12 삼각형. 바닥면도 넣는다 — 라이다는 위에서 안 보지만,
    빼면 메시가 열려 있어 콜라이더 생성이 판마다 다르게 나온다.
    """
    ox, oy = float(origin[0]), float(origin[1])
    pts, idx = [], []
    for y0, y1, x0, x1 in rects:
        # 이미지 행 y 는 맵 y 가 **아래에서 위로** 이므로 뒤집어야 한다.
        # (map_server 는 pgm 마지막 행을 origin 쪽으로 놓는다)
        xa, xb = ox + x0 * res, ox + x1 * res
        ya, yb = oy + y0 * res, oy + y1 * res
        b = len(pts)
        for z in (0.0, height):
            pts += [(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)]
        idx += [
            b + 0, b + 2, b + 1, b + 0, b + 3, b + 2,          # 바닥
            b + 4, b + 5, b + 6, b + 4, b + 6, b + 7,          # 천장
            b + 0, b + 1, b + 5, b + 0, b + 5, b + 4,          # 옆면 4
            b + 1, b + 2, b + 6, b + 1, b + 6, b + 5,
            b + 2, b + 3, b + 7, b + 2, b + 7, b + 6,
            b + 3, b + 0, b + 4, b + 3, b + 4, b + 7,
        ]
    return np.asarray(pts, dtype=np.float32), np.asarray(idx, dtype=np.int32)


def write_usd(path: str, pts, idx, say) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/World/map")
    # numpy.float32 는 Gf.Vec3f 생성자가 안 받는다 — 파이썬 float 로 캐스팅
    mesh.CreatePointsAttr([Gf.Vec3f(float(a), float(b), float(c)) for a, b, c in pts])
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in idx])
    mesh.CreateFaceVertexCountsAttr([3] * (len(idx) // 3))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    # 정적 삼각형 메시 콜라이더. approximation=none 이라야 통로 폭이 보존된다 —
    # convexHull 로 근사하면 벽이 부풀어 문이 막힌다.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mc.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
    # PhysxSchema 는 Kit 확장이라 맨 pxr 에는 없다. 없어도 Isaac 이 로드할 때
    # PhysX 기본값을 채우므로 선택적으로만 적용한다.
    try:
        from pxr import PhysxSchema  # noqa: E402
        PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    except ImportError:
        say("  (PhysxSchema 없음 — Isaac 로드 시 PhysX 기본값이 채워진다)")

    stage.GetRootLayer().Save()
    say("  USD 저장 %s" % path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="handoff/map_munji_3f_2025_wide/munji_3f_2025_wide.yaml")
    ap.add_argument("--out", default="twin/world/assets/munji_3f_2025_wide.usd")
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--unknown-occupied", dest="unk", type=int, default=1,
                    help="1 이면 미지영역을 장애물로 (기본). 0 이면 자유공간으로")
    ap.add_argument("--open-px", type=int, default=1)
    ap.add_argument("--close-px", type=int, default=2)
    ap.add_argument("--no-usd", action="store_true", help="분석만 하고 USD 는 안 쓴다")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def say(m: str = ""):
        print(m, flush=True)
        lines.append(m)

    meta = read_map_yaml(args.map)
    img = read_pgm(os.path.join(os.path.dirname(args.map), str(meta["image"])))
    res = float(meta["resolution"])
    origin = meta["origin"]
    h, w = img.shape

    say("=" * 76)
    say("맵 압출 — %s" % os.path.basename(args.map))
    say("=" * 76)
    say("  크기       %d x %d @ %.3f m = %.1f x %.1f m" % (w, h, res, w * res, h * res))
    say("  origin     (%.2f, %.2f)   범위 x [%.1f, %.1f]  y [%.1f, %.1f]"
        % (origin[0], origin[1], origin[0], origin[0] + w * res,
           origin[1], origin[1] + h * res))
    say("  임계값     occupied %.2f  free %.2f  (yaml)"
        % (meta.get("occupied_thresh", 0.65), meta.get("free_thresh", 0.25)))

    free, wall, unk = classify(img)
    say("")
    say("  원본 픽셀 분류")
    say("    자유  %8d (%5.2f%%)   벽  %6d (%.2f%%)   미지  %8d (%5.2f%%)"
        % (free.sum(), 100 * free.mean(), wall.sum(), 100 * wall.mean(),
           unk.sum(), 100 * unk.mean()))
    fth = float(meta.get("free_thresh", 0.25))
    if 0.196 < fth:
        say("    ⚠ free_thresh %.2f > 0.196 이라 map_server 는 미지(205)를 **자유공간으로**"
            % fth)
        say("      발행한다. 즉 /map 에 -1 이 없고, 플래너의 unknown_is_occupied 는")
        say("      이 맵에서 무효다. 여기서는 원본 픽셀로 직접 분류한다.")

    obst_raw = wall | unk if args.unk else wall
    # ★ 정리한 뒤 **원본 벽을 다시 합친다.** 열기(open)는 1 셀 두께 돌출을 지우는데,
    #   SLAM 맵의 벽은 흔히 1~2 셀(0.05~0.10 m)이라 그대로 두면 **실재하는 벽이
    #   사라져 없는 문이 생긴다.** 정리는 미지영역 경계의 잡티에만 적용되어야 한다.
    obst = clean(obst_raw, args.open_px, args.close_px) | wall
    say("")
    say("  장애물 마스크 (미지=%s)  정리 전 %d -> 후 %d 셀 (%+d)"
        % ("장애물" if args.unk else "자유", obst_raw.sum(), obst.sum(),
           int(obst.sum()) - int(obst_raw.sum())))

    # 정리가 통로를 막지 않았는지 확인한다. 닫기(close)는 벽의 스캔 구멍을 메우지만
    # **폭이 close_px*2+1 셀보다 좁은 통로도 같이 막는다.** 그러면 없던 벽이 생긴 채로
    # 시뮬을 돌리게 되고, 정책이 못 지나간 것이 아니라 월드가 막혀 있던 것이 된다.
    def largest_share(mask):
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            return 0.0, 0
        s = ndi.sum(mask, lab, range(1, n + 1))
        return float(s.max()) / float(mask.sum()), n

    sh0, n0 = largest_share(~obst_raw)
    sh1, n1 = largest_share(~obst)
    say("  자유공간 연결성  정리 전 최대성분 %.1f%% (%d 성분) -> 후 %.1f%% (%d 성분)"
        % (100 * sh0, n0, 100 * sh1, n1))
    if sh1 < sh0 - 0.01:
        say("    ⚠ 정리가 자유공간을 %.1f%% 잘라냈다. close_px=%d 는 폭 %.2f m 미만의"
            % (100 * (sh0 - sh1), args.close_px, (args.close_px * 2 + 1) * res))
        say("      통로를 막는다. --close-px 를 낮춰 보고, 그래도 남으면 실제로 그만큼")
        say("      좁은 통로가 있다는 뜻이니 어느 쪽을 믿을지 결정해야 한다.")

    # 정리 뒤의 자유공간으로 분석한다 — 압출될 월드와 같은 것을 재야 한다.
    geom = {"base_r": float(np.hypot(0.55 / 2, (0.80 / 2) / 3)),
            "obj_r": float(np.hypot(0.30 / 2, (2.00 / 2) / 5))}
    say("  디스크 반경  base %.3f m (0.80x0.55, n=3)   object %.3f m (2.00x0.30, n=5)"
        % (geom["base_r"], geom["obj_r"]))
    analyse(~obst, obst, res, geom, say)

    rects = rectangles(obst)
    say("")
    say("  사각형 분해  %d 개  (평균 %.1f 셀)"
        % (len(rects), obst.sum() / max(1, len(rects))))

    if not args.no_usd:
        # pgm 의 행 0 은 맵의 **위쪽**이다. 뒤집어야 origin 규약과 맞는다.
        pts, idx = build_mesh(
            np.stack([h - rects[:, 1], h - rects[:, 0], rects[:, 2], rects[:, 3]], axis=1),
            res, origin, args.height)
        say("  메시         정점 %d  삼각형 %d" % (len(pts), len(idx) // 3))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        write_usd(args.out, pts, idx, say)
    say("=" * 76)

    if args.report:
        with open(args.report, "w") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
