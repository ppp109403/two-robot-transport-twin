"""Isaac(py3.11) <-> ROS2(py3.12) 공유메모리 다리.

왜 rclpy 를 Isaac 안에서 안 쓰나
--------------------------------
::

    Isaac Sim 5.1 / conda ppo_afl   Python 3.11
    ROS 2 Jazzy                     Python 3.12

rclpy 는 배포판 파이썬에 맞춰 컴파일된 확장이라 3.11 로 import 되지 않는다. 그래서
Isaac 프로세스는 ROS 를 모르고, **ROS 쪽 별도 프로세스**가 토픽을 받아 여기에 적는다.

왜 mmap 인가
------------
의존성이 없다 (표준 라이브러리만). ZMQ 나 소켓을 쓰면 두 파이썬 양쪽에 같은 패키지를
깔아야 하는데, 이 저장소는 이미 파이썬이 셋(3.11 ppo_afl / 3.12 jazzy / 3.12 conda ros2)
으로 갈라져 있어서 공유 의존성을 늘리면 그만큼 깨질 자리가 는다.

락 대신 seqlock
---------------
30~100 Hz 로 작은 구조체 하나를 주고받을 뿐이라 뮤텍스가 과하다. 쓰는 쪽은 seq 를
**쓰기 전후로 한 번씩** 올리고 (홀수 = 쓰는 중), 읽는 쪽은 앞뒤 seq 가 같고 짝수일
때만 값을 받는다. 찢어진 프레임을 읽으면 그냥 직전 값을 유지한다 — 한 프레임 늦는 것이
반쪽짜리 자세를 쓰는 것보다 낫다.
"""

from __future__ import annotations

import mmap
import os
import struct

MAGIC = b"TWIN"
VERSION = 3
SIDES = ("a", "b")

#: 기본 경로. /dev/shm 은 tmpfs 라 디스크를 안 친다.
DEFAULT_PATH = "/dev/shm/twin_bridge.bin"

# ---------------------------------------------------------------- 레이아웃
#  헤더        magic(4s) version(I)
#  명령 블록   seq(Q) plan_id(Q) spawn_id(Q) mode(i) _pad(i) t(d)
#              per side: ee_pos(3d) ee_quat(4d) base_pos(3d) base_quat(4d) twist(2d)
#              per side: start(6d) = 계획 t=0 의 차체(x,y,yaw) + EE(x,y,yaw)
#
#  spawn_id 는 RViz 의 **SETUP 버튼**이 눌린 횟수다. 계획이 풀렸다고 바로 스폰하지
#  않는 이유는, 애니메이션으로 계획을 먼저 확인하고 싶기 때문이다.
#  start 에 EE 도 넣는 이유는, EXECUTE 전에는 노드가 셋포인트를 아예 발행하지 않아서
#  Isaac 이 붙잡고 있을 목표를 스스로 알아야 하기 때문이다.
#  측정 블록   seq(Q)
#              per side: ee_pos(3d) ee_quat(4d) base_pos(3d) base_quat(4d)
_HDR = struct.Struct("<4sI")
_CMD_HEAD = struct.Struct("<QQQiid")
_CMD_SIDE = struct.Struct("<16d")          # ee 7 + base 7 + twist 2
_CMD_START = struct.Struct("<6d")
_MEAS_HEAD = struct.Struct("<Q")
_MEAS_SIDE = struct.Struct("<14d")         # ee 7 + base 7

_OFF_HDR = 0
_OFF_CMD = _HDR.size
_CMD_BYTES = _CMD_HEAD.size + 2 * _CMD_SIDE.size + 2 * _CMD_START.size
_OFF_MEAS = _OFF_CMD + _CMD_BYTES
_MEAS_BYTES = _MEAS_HEAD.size + 2 * _MEAS_SIDE.size
SIZE = _OFF_MEAS + _MEAS_BYTES

#: mode 값. ROS 쪽이 계획 발행/실행 상태를 알려 준다.
MODE_IDLE, MODE_HOLD, MODE_RUN = 0, 1, 2


class Bridge:
    """양쪽에서 같은 파일을 열어 쓴다. ``create=True`` 는 한쪽만."""

    def __init__(self, path: str = DEFAULT_PATH, create: bool = False):
        self.path = path
        if create:
            with open(path, "wb") as fh:
                fh.write(b"\0" * SIZE)
        elif not os.path.exists(path):
            raise FileNotFoundError(
                "공유메모리가 없다: %s\n  ROS 쪽 twin_bridge 를 먼저 띄울 것" % path)
        self._f = open(path, "r+b")
        self.m = mmap.mmap(self._f.fileno(), SIZE)
        if create:
            _HDR.pack_into(self.m, _OFF_HDR, MAGIC, VERSION)
        else:
            magic, ver = _HDR.unpack_from(self.m, _OFF_HDR)
            if magic != MAGIC or ver != VERSION:
                raise ValueError("공유메모리 형식이 다르다 (%r v%d). 양쪽 twin 버전을 맞출 것"
                                 % (magic, ver))

    def close(self) -> None:
        self.m.flush()
        self.m.close()
        self._f.close()

    # ------------------------------------------------------------ 명령 (ROS -> Isaac)
    def write_cmd(self, plan_id: int, spawn_id: int, mode: int, t: float,
                  per_side: dict, starts: dict) -> None:
        seq, = struct.unpack_from("<Q", self.m, _OFF_CMD)
        seq += 1
        struct.pack_into("<Q", self.m, _OFF_CMD, seq)          # 홀수 = 쓰는 중
        _CMD_HEAD.pack_into(self.m, _OFF_CMD, seq, plan_id, spawn_id, mode, 0, t)
        off = _OFF_CMD + _CMD_HEAD.size
        for s in SIDES:
            _CMD_SIDE.pack_into(self.m, off, *per_side[s])
            off += _CMD_SIDE.size
        for s in SIDES:
            _CMD_START.pack_into(self.m, off, *starts[s])
            off += _CMD_START.size
        struct.pack_into("<Q", self.m, _OFF_CMD, seq + 1)      # 짝수 = 완료

    def read_cmd(self) -> dict | None:
        for _ in range(4):
            s0, plan_id, spawn_id, mode, _pad, t = _CMD_HEAD.unpack_from(self.m, _OFF_CMD)
            if s0 % 2:                       # 쓰는 중
                continue
            off = _OFF_CMD + _CMD_HEAD.size
            sides, starts = {}, {}
            for s in SIDES:
                sides[s] = _CMD_SIDE.unpack_from(self.m, off)
                off += _CMD_SIDE.size
            for s in SIDES:
                starts[s] = _CMD_START.unpack_from(self.m, off)
                off += _CMD_START.size
            s1, = struct.unpack_from("<Q", self.m, _OFF_CMD)
            if s0 == s1:
                return {"seq": s0, "plan_id": plan_id, "spawn_id": spawn_id,
                        "mode": mode, "t": t, "side": sides, "start": starts}
        return None                          # 계속 찢어지면 호출부가 직전 값을 유지

    # ------------------------------------------------------------ 측정 (Isaac -> ROS)
    def write_meas(self, per_side: dict) -> None:
        seq, = struct.unpack_from("<Q", self.m, _OFF_MEAS)
        seq += 1
        struct.pack_into("<Q", self.m, _OFF_MEAS, seq)
        off = _OFF_MEAS + _MEAS_HEAD.size
        for s in SIDES:
            _MEAS_SIDE.pack_into(self.m, off, *per_side[s])
            off += _MEAS_SIDE.size
        struct.pack_into("<Q", self.m, _OFF_MEAS, seq + 1)

    def read_meas(self) -> dict | None:
        for _ in range(4):
            s0, = _MEAS_HEAD.unpack_from(self.m, _OFF_MEAS)
            if s0 % 2:
                continue
            off = _OFF_MEAS + _MEAS_HEAD.size
            sides = {}
            for s in SIDES:
                sides[s] = _MEAS_SIDE.unpack_from(self.m, off)
                off += _MEAS_SIDE.size
            s1, = _MEAS_HEAD.unpack_from(self.m, _OFF_MEAS)
            if s0 == s1:
                return {"seq": s0, "side": sides}
        return None
