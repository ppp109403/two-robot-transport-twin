"""로봇 실측 기하 — **한 군데서만 정의한다.**

왜 별도 모듈인가
----------------
차체 크기는 플래너 파라미터, 맵 통과 가능성 분석, twin 의 침투 판정 세 군데에서 쓰인다.
셋이 따로 적혀 있으면 하나를 고쳐도 나머지가 옛 값을 쓰고, 그 어긋남은 **에러 없이
"통과 가능" 판정만 바꾼다.** 이 저장소가 이미 여러 번 당한 형태다.

실측 근거 (두 경로가 일치)
--------------------------
::

    에셋 주석   amr_diff_drive.py — amr_base.stl bbox
                x -0.317 ~ +0.3375,  y +-0.201
    Isaac 실측  base_link 월드 AABB (기본자세)
                x -0.317 ~ +0.338,   y -0.201 ~ +0.201

    -> 길이 0.655 m,  폭 0.402 m

플래너가 쓰던 값은 0.80 x 0.55 였다. 폭이 37 % 과대여서 원판 반경이 0.306 m 로
부풀었고 (실측이면 0.233), 사방 7.3 cm 씩 더 먹어 좁은 문이 막혔다.

비대칭에 대하여
---------------
차체는 base_link 기준으로 앞뒤가 다르다 (-0.317 / +0.338). 플래너의 사각형 모델은
대칭이므로 긴 쪽에 맞춰 ``0.68`` 을 쓴다 — 1 cm 정도 더 먹지만 짧은 쪽에 맞추면
뒤쪽이 모델 밖으로 나간다.

팔은 여기 없다
--------------
``base_margin`` 을 실측만큼 같이 줄이면 안 된다. 팔·그리퍼는 플래너 충돌 모델에
**아예 없고**, 기본자세에서 링크 원점의 y 퍼짐이 0.52 m 로 차체 폭보다 넓다.
margin 은 그 몫으로 남겨 두는 것이 안전하다.
"""

from __future__ import annotations

import math

#: 차체 사각형 [m]. Isaac base_link AABB 실측 + 앞뒤 비대칭 보정.
#: 원판 수를 5 로 둔 이유: r = hypot(반폭, 반길이/n) 이라 n 을 올리면 반경이
#: 준다 (3 -> 0.2342, 5 -> 0.2160, 하한은 반폭 0.205). 1.8 cm 를 거의 공짜로
#: 벌 수 있고, 그만큼 base_margin 을 덜 깎아도 좁은 문을 지난다.
BASE_LEN = 0.68
BASE_WID = 0.41
N_BASE_DISKS = 5

#: 참고: 순수 실측값 (대칭 보정 전)
BASE_LEN_MEASURED = 0.655
BASE_WID_MEASURED = 0.402

#: 막대 (운반 물체). 길이는 운용에서 정한다.
OBJ_WID = 0.30
N_OBJ_DISKS = 10


def disk_cover(half_len: float, half_wid: float, n: int) -> tuple[list, float]:
    """``two_robot_nlp.cover_rect_with_disks`` 와 **같은 식**이어야 한다.

    n 개 원판을 긴 축을 따라 놓고, 반경은 각 원판이 덮어야 할 구간의 대각선이다::

        r = hypot(half_wid, half_len / n)
    """
    n = max(1, int(n))
    r = float(math.hypot(half_wid, half_len / n))
    offs = [-half_len + (2 * i + 1) * (half_len / n) for i in range(n)]
    return offs, r


def base_disks(n: int = N_BASE_DISKS):
    """차체 원판 (오프셋 목록, 반경). 침투 판정과 여유 분석이 같이 쓴다."""
    return disk_cover(BASE_LEN / 2.0, BASE_WID / 2.0, n)


def base_radius(n: int = N_BASE_DISKS) -> float:
    return base_disks(n)[1]


def obj_disks(obj_len: float, n: int = N_OBJ_DISKS):
    return disk_cover(obj_len / 2.0, OBJ_WID / 2.0, n)


def ee_clearance(obj_len: float, obj_margin: float, n: int = N_OBJ_DISKS) -> float:
    """계획이 EE 에 보장하는 벽까지 거리.

    끝 원판 중심은 ``half_len*(1 - 1/n)`` 이라 EE(=막대 끝)보다 ``half_len/n`` 안쪽이다.
    그만큼 여유가 깎인다::

        EE 여유 = (r + obj_margin) - half_len/n
    """
    _, r = obj_disks(obj_len, n)
    return r + obj_margin - (obj_len / 2.0) / max(1, n)


if __name__ == "__main__":
    print("차체 %.3f x %.3f (실측 %.3f x %.3f)"
          % (BASE_LEN, BASE_WID, BASE_LEN_MEASURED, BASE_WID_MEASURED))
    for n in (3, 5):
        print("  n=%d -> r %.4f m   (구 설정 0.80x0.55 n=3 은 %.4f)"
              % (n, base_radius(n), disk_cover(0.40, 0.275, 3)[1]))
    for L in (1.5, 2.0):
        print("  막대 %.2f m, n=%d, obj_margin 0.05 -> EE 여유 %.3f m"
              % (L, N_OBJ_DISKS, ee_clearance(L, 0.05)))
