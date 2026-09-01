"""ROS 2 <-> Isaac 다리. **시스템 py3.12 + Jazzy** 에서 돈다.

토픽 이름 주의
--------------
배포된 ``two_robot_nlp_node`` 는 셋포인트를 ``*_pose`` 로 낸다 (``*_cmd`` 가 아니다)::

    /<ns>/ee_pose     PoseStamped   EE 셋포인트 (100 Hz, EXECUTE 중에만)
    /<ns>/base_pose   PoseStamped   차체 셋포인트
    /<ns>/base_twist  TwistStamped  차체 (v, w)
    /<ns>/ee_path     Path (latched) 계획 전체.  **첫 점이 t=0 목표**
    /<ns>/base_path   Path (latched) 〃
    /twin/setup       Bool (latched) RViz 의 SETUP 버튼

그래서 Isaac 이 잰 값은 같은 이름으로 되돌릴 수 없다 (셋포인트와 충돌한다).
``/<ns>/ee_meas`` / ``/<ns>/base_meas`` 로 낸다.

EXECUTE 전에는 셋포인트가 아예 안 나온다
----------------------------------------
노드의 ``_tick_stream`` 은 실행 중이 아니면 그냥 return 한다. 그래서 SETUP 직후
Isaac 이 붙잡고 있을 목표가 토픽으로는 오지 않는다. 대신 latched ``ee_path`` 의
**첫 점**을 시작 목표로 실어 보낸다.

모드 판정도 같은 이유로 **셋포인트가 최근에 왔는지**로 한다 — RViz 버튼은 토픽을
쓰지 않아 실행 상태를 직접 알 방법이 없다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shm import MODE_HOLD, MODE_IDLE, MODE_RUN, SIDES, Bridge  # noqa: E402

LATCHED = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

#: 셋포인트가 이 시간 안에 왔으면 재생 중으로 본다 [s]. 100 Hz 스트림 기준 넉넉하다.
RUN_TIMEOUT = 0.30


def quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TwinBridge(Node):
    def __init__(self, path: str, ns_a: str, ns_b: str, rate: float):
        super().__init__("twin_bridge")
        self.br = Bridge(path, create=True)
        self.ns = {"a": ns_a, "b": ns_b}

        z = [0.0] * 16
        z[3] = 1.0          # ee quat w
        z[10] = 1.0         # base quat w
        self.cmd = {s: list(z) for s in SIDES}
        self.start = {s: [0.0] * 6 for s in SIDES}      # base xyyaw + ee xyyaw
        self.plan_id = 0
        self.spawn_id = 0
        self._last_sp = None                            # 마지막 셋포인트 도착 시각
        self._got = {s: False for s in SIDES}
        self._n_sp = 0

        self.pub_ee, self.pub_base = {}, {}
        for s in SIDES:
            n = self.ns[s]
            self.create_subscription(PoseStamped, "/%s/ee_pose" % n,
                                     lambda m, k=s: self._on_ee(k, m), 50)
            self.create_subscription(PoseStamped, "/%s/base_pose" % n,
                                     lambda m, k=s: self._on_base(k, m), 50)
            self.create_subscription(TwistStamped, "/%s/base_twist" % n,
                                     lambda m, k=s: self._on_twist(k, m), 50)
            self.create_subscription(Path, "/%s/ee_path" % n,
                                     lambda m, k=s: self._on_ee_path(k, m), LATCHED)
            self.create_subscription(Path, "/%s/base_path" % n,
                                     lambda m, k=s: self._on_base_path(k, m), LATCHED)
            self.pub_ee[s] = self.create_publisher(PoseStamped, "/%s/ee_meas" % n, 20)
            self.pub_base[s] = self.create_publisher(PoseStamped, "/%s/base_meas" % n, 20)
        self.create_subscription(Bool, "/twin/setup", self._on_setup, LATCHED)

        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(2.0, self._report)
        self.get_logger().info("공유메모리 %s  |  ns %s, %s" % (path, ns_a, ns_b))

    # ------------------------------------------------------------------ 셋포인트
    def _on_ee(self, s, m: PoseStamped):
        p, q = m.pose.position, m.pose.orientation
        self.cmd[s][0:7] = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
        self._got[s] = True
        self._last_sp = self.get_clock().now()
        self._n_sp += 1

    def _on_base(self, s, m: PoseStamped):
        p, q = m.pose.position, m.pose.orientation
        self.cmd[s][7:14] = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]

    def _on_twist(self, s, m: TwistStamped):
        self.cmd[s][14:16] = [m.twist.linear.x, m.twist.angular.z]

    # ------------------------------------------------------------------ 계획
    def _on_base_path(self, s, m: Path):
        if not m.poses:
            return
        p0 = m.poses[0].pose
        self.start[s][0:3] = [p0.position.x, p0.position.y, quat_to_yaw(p0.orientation)]
        if s == "a":
            self.plan_id += 1
            self.get_logger().info(
                "새 계획 #%d  — RViz 애니메이션 확인 후 SETUP 을 누를 것" % self.plan_id)

    def _on_ee_path(self, s, m: Path):
        if not m.poses:
            return
        p0 = m.poses[0].pose
        self.start[s][3:6] = [p0.position.x, p0.position.y, quat_to_yaw(p0.orientation)]

    def _on_setup(self, m: Bool):
        if not m.data:
            return
        self.spawn_id += 1
        self.get_logger().info(
            "SETUP #%d — Isaac 이 계획 시작 자세로 스폰한다  "
            "A base (%.2f, %.2f, %.1f deg) ee (%.2f, %.2f)"
            % (self.spawn_id, self.start["a"][0], self.start["a"][1],
               math.degrees(self.start["a"][2]), self.start["a"][3], self.start["a"][4]))

    # ------------------------------------------------------------------ 주기
    def _mode(self) -> int:
        if self.plan_id == 0:
            return MODE_IDLE
        if self._last_sp is None:
            return MODE_HOLD
        age = (self.get_clock().now() - self._last_sp).nanoseconds * 1e-9
        return MODE_RUN if age < RUN_TIMEOUT else MODE_HOLD

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        self.br.write_cmd(self.plan_id, self.spawn_id, self._mode(), t,
                          self.cmd, self.start)

        meas = self.br.read_meas()
        if meas is None:
            return
        stamp = self.get_clock().now().to_msg()
        for s in SIDES:
            v = meas["side"][s]
            if all(abs(x) < 1e-12 for x in v[:3]):
                continue                      # Isaac 이 아직 안 적었다
            for pub, off in ((self.pub_ee[s], 0), (self.pub_base[s], 7)):
                msg = PoseStamped()
                msg.header.frame_id = "map"
                msg.header.stamp = stamp
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = v[off:off + 3]
                (msg.pose.orientation.w, msg.pose.orientation.x,
                 msg.pose.orientation.y, msg.pose.orientation.z) = v[off + 3:off + 7]
                pub.publish(msg)

    def _report(self):
        m = {MODE_IDLE: "idle", MODE_HOLD: "hold", MODE_RUN: "RUN"}[self._mode()]
        have = "".join(s.upper() if self._got[s] else "-" for s in SIDES)
        meas = self.br.read_meas()
        alive = "yes" if meas and meas["seq"] > 0 else "no"
        self.get_logger().info(
            "plan #%d  spawn #%d  mode %-4s  셋포인트 %s %d/2s  |  Isaac 측정 %s"
            % (self.plan_id, self.spawn_id, m, have, self._n_sp, alive))
        self._n_sp = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shm", default="/dev/shm/twin_bridge.bin")
    ap.add_argument("--ns-a", dest="ns_a", default="a")
    ap.add_argument("--ns-b", dest="ns_b", default="b")
    ap.add_argument("--rate", type=float, default=100.0)
    args, rest = ap.parse_known_args()

    rclpy.init(args=rest)
    node = TwinBridge(args.shm, args.ns_a, args.ns_b, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.br.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
