"""GUI 로 볼 때 씬에 겹쳐 그리는 것들.

무엇을 그리나
-------------
::

    계획 EE 경로     로봇별 색, 얇은 선     "따라가야 할 길"
    계획 차체 경로   같은 색, 더 얇게       차체 참조 (정책은 이걸 안 보지만 편차를 눈으로)
    현재 목표        굵은 점                지금 이 순간의 EE 목표
    오차선           목표 <-> 실제 tool0    길어지면 추종이 무너지는 중

숫자로만 보면 "13.7 mm" 가 좋은지 나쁜지 감이 안 온다. 오차선이 거의 안 보이면
잘 붙어 있는 것이고, 눈에 띄게 길어지면 그 구간을 로그에서 찾아보면 된다.

debug_draw 는 GUI 가 있을 때만 의미가 있으므로, 없으면 조용히 아무것도 안 한다.
"""

from __future__ import annotations

import numpy as np

#: 로봇별 색 (r, g, b, a)
COLOR = {"a": (0.20, 0.85, 0.35, 1.0), "b": (0.25, 0.55, 1.00, 1.0)}
COLOR_DIM = {"a": (0.20, 0.85, 0.35, 0.35), "b": (0.25, 0.55, 1.00, 0.35)}
ERR_OK = (1.0, 1.0, 1.0, 0.9)
ERR_BAD = (1.0, 0.25, 0.25, 1.0)


class DebugViz:
    """isaacsim debug_draw 래퍼. 인터페이스가 없으면 무해하게 비활성."""

    def __init__(self, goal_z: float = 0.55, err_warn: float = 0.05):
        self.goal_z, self.err_warn = goal_z, err_warn
        self.draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:  # noqa: BLE001 - GUI 없이 돌 때가 정상 경로다
            self.draw = None
        self._static_done = False

    @property
    def enabled(self) -> bool:
        return self.draw is not None

    # ------------------------------------------------------------------ 정적
    def draw_tracks(self, tracks: dict) -> None:
        """계획 경로. 매 프레임 다시 그릴 필요가 없지만 debug_draw 는 프레임마다
        지워지므로 :meth:`update` 에서 같이 그린다. 여기서는 점 목록만 만들어 둔다."""
        self._ee_seg, self._base_seg, self._colors, self._colors_dim = [], [], [], []
        for s, tr in tracks.items():
            ee = np.column_stack([tr.ee[:, 1], tr.ee[:, 2],
                                  np.full(len(tr.ee), self.goal_z)])
            bs = np.column_stack([tr.base[:, 1], tr.base[:, 2],
                                  np.full(len(tr.base), 0.05)])
            for arr, store, col in ((ee, self._ee_seg, self._colors),
                                    (bs, self._base_seg, self._colors_dim)):
                for i in range(len(arr) - 1):
                    store.append((tuple(arr[i]), tuple(arr[i + 1])))
                    col.append(COLOR[s] if store is self._ee_seg else COLOR_DIM[s])
        self._static_done = True

    # ------------------------------------------------------------------ 매 스텝
    def update(self, targets: dict, ee_now: dict) -> None:
        if not self.enabled:
            return
        d = self.draw
        d.clear_lines()
        d.clear_points()

        if self._static_done:
            starts = [a for a, _ in self._ee_seg] + [a for a, _ in self._base_seg]
            ends = [b for _, b in self._ee_seg] + [b for _, b in self._base_seg]
            cols = list(self._colors) + list(self._colors_dim)
            d.draw_lines(starts, ends, cols, [2.0] * len(self._ee_seg)
                         + [1.0] * len(self._base_seg))

        pts, pcols, psz = [], [], []
        es, ee_, ecol, esz = [], [], [], []
        for s, tgt in targets.items():
            p = tuple(float(v) for v in tgt["pos_w"])
            q = tuple(float(v) for v in ee_now[s])
            pts += [p, q]
            pcols += [COLOR[s], (1.0, 1.0, 0.25, 1.0)]
            psz += [14.0, 10.0]
            err = float(np.linalg.norm(np.asarray(p) - np.asarray(q)))
            es.append(p); ee_.append(q)
            ecol.append(ERR_BAD if err > self.err_warn else ERR_OK)
            esz.append(4.0 if err > self.err_warn else 2.0)
        d.draw_points(pts, pcols, psz)
        d.draw_lines(es, ee_, ecol, esz)

    def clear(self) -> None:
        if self.enabled:
            self.draw.clear_lines()
            self.draw.clear_points()


class BarMarker:
    """두 EE 사이를 잇는 **갈색 막대**. 시각물일 뿐 물리가 아니다.

    무엇이 아닌가
    -------------
    충돌체도 질량도 없다. 두 로봇을 구속하지 않고, 힘도 주고받지 않는다. 그래서
    "정말 이 막대를 들 수 있는가" 는 이걸로 알 수 없다 — 그건 스프링댐퍼(B1)나
    강체 결합(B2)을 붙여야 나온다.

    그럼에도 붙이는 이유는, 숫자로 "막대 길이 오차 12 mm" 를 보는 것과 두 팔 사이에
    실제로 막대가 걸쳐 흔들리는 것을 보는 것이 다르기 때문이다. 파지점이 벌어지거나
    막대가 벽을 스치는 순간이 **눈에 먼저 걸린다.**

    구현
    ----
    원기둥 하나를 매 프레임 두 TCP 사이로 옮기고, 축을 방향에 맞추고, 길이만큼
    z 로 늘린다. 원기둥 기본 축이 +z 이므로 +z -> 방향벡터 최단회전을 쓴다.
    """

    def __init__(self, radius: float = 0.075, color=(0.45, 0.28, 0.12),
                 prim_path: str = "/Visuals/twin_bar"):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "bar": sim_utils.CylinderCfg(
                    radius=radius, height=1.0,      # 높이 1 로 두고 scale 로 늘린다
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color, roughness=0.8),
                ),
            },
        )
        self.mk = VisualizationMarkers(self.cfg)
        self._torch = __import__("torch")

    def update(self, ee_a, ee_b) -> float:
        """두 TCP 위치를 받아 막대를 맞춘다. 현재 길이를 돌려준다."""
        import numpy as np
        a = np.asarray(ee_a, dtype=np.float64)
        b = np.asarray(ee_b, dtype=np.float64)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return L
        u = d / L
        # +z -> u 최단회전 쿼터니언 (wxyz)
        c = float(u[2])
        if c < -1.0 + 1e-9:                       # 정확히 반대 방향
            q = np.array([0.0, 1.0, 0.0, 0.0])
        else:
            v = np.array([-u[1], u[0], 0.0])      # z x u
            q = np.array([1.0 + c, v[0], v[1], v[2]])
            q /= np.linalg.norm(q)
        t = self._torch
        self.mk.visualize(
            translations=t.tensor([(a + b) * 0.5], dtype=t.float32),
            orientations=t.tensor([q], dtype=t.float32),
            scales=t.tensor([[1.0, 1.0, L]], dtype=t.float32),
        )
        return L
