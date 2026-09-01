# src/mobile_manipulator_trajectory/object_planner_executor.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy  # ✅ QoS 추가

import time, math, os, csv
from copy import deepcopy
from datetime import datetime
from typing import List

from geometry_msgs.msg import (
    PoseWithCovarianceStamped, PoseStamped, Quaternion, TransformStamped, Point
)
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray

import tf2_ros
from tf2_ros import Buffer
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformException


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MultiTfListener:
    """
    여러 네임스페이스의 TF(/robot1/tf, /robot1/tf_static, /robot2/tf, /robot2/tf_static 등)를
    **올바른 QoS**로 한 Buffer에 넣어주는 리스너.
    - /tf: VOLATILE + BEST_EFFORT
    - /tf_static: TRANSIENT_LOCAL + RELIABLE  ✅ 핵심
    """
    def __init__(self, node: Node, buffer: Buffer, topics: List[str]):
        self._node = node
        self._buffer = buffer
        self._subs = []

        # 표준 TF QoS 프로파일
        qos_tf = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_tf_static = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,  # 1도 가능하나 여유 있게
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # ★ 정적 TF는 반드시 TL
        )

        for t in topics:
            is_static = t.endswith('tf_static')
            qos = qos_tf_static if is_static else qos_tf
            self._subs.append(
                node.create_subscription(TFMessage, t, self._cb_factory(t), qos)
            )

    def _cb_factory(self, topic_name: str):
        is_static = topic_name.endswith('tf_static')

        def _cb(msg: TFMessage):
            authority = topic_name
            if is_static:
                for t in msg.transforms:
                    try:
                        self._buffer.set_transform_static(t, authority)
                    except Exception:
                        pass
            else:
                for t in msg.transforms:
                    try:
                        self._buffer.set_transform(t, authority)
                    except Exception:
                        pass
        return _cb


class ObjectPlannerExecutor(Node):
    def __init__(self):
        super().__init__('object_planner_executor')

        # ---- 기본 파라미터 ----
        self.declare_parameter('model_name', 'rect_object')
        self.declare_parameter('size_l', 1.2)   # 물체 길이
        self.declare_parameter('size_w', 0.8)
        self.declare_parameter('size_h', 0.25)
        self.declare_parameter('speed', 0.3)
        self.declare_parameter('dt', 0.03)
        self.declare_parameter('broadcast_tf', False)
        self.declare_parameter('simulate_motion', True)
        self.declare_parameter('show_heading_markers', False)
        self.declare_parameter('world_frame', 'map')

        # ---- 입력/출력 토픽 이름 ----
        self.declare_parameter('a_topic', '/a/ee_pose')
        self.declare_parameter('b_topic', '/b/ee_pose')
        self.declare_parameter('a_path_topic', '/a/ee_path')
        self.declare_parameter('b_path_topic', '/b/ee_path')
        self.declare_parameter('ds', 0.01)  # 리샘플 간격

        # ---- TF 관련 ----
        # a=robot1, b=robot2
        self.declare_parameter('a_frame', 'robot1/link4')
        self.declare_parameter('b_frame', 'robot2/link4')
        self.declare_parameter('tf_topics', ['/robot1/tf','/robot1/tf_static','/robot2/tf','/robot2/tf_static'])
        self.declare_parameter('tf_poll_hz', 50.0)  # TF 조회 주기

        p = self.get_parameter
        self.model_name = p('model_name').get_parameter_value().string_value
        self.L = p('size_l').get_parameter_value().double_value
        self.W = p('size_w').get_parameter_value().double_value
        self.H = p('size_h').get_parameter_value().double_value
        self.speed = p('speed').get_parameter_value().double_value
        self.dt = p('dt').get_parameter_value().double_value
        self.broadcast_tf = p('broadcast_tf').get_parameter_value().bool_value
        self.simulate_motion = p('simulate_motion').get_parameter_value().bool_value
        self.show_heading_markers = p('show_heading_markers').get_parameter_value().bool_value
        self.world_frame = p('world_frame').get_parameter_value().string_value

        self.a_topic = p('a_topic').get_parameter_value().string_value
        self.b_topic = p('b_topic').get_parameter_value().string_value
        self.a_path_topic = p('a_path_topic').get_parameter_value().string_value
        self.b_path_topic = p('b_path_topic').get_parameter_value().string_value
        self.ds = p('ds').get_parameter_value().double_value

        self.a_frame = p('a_frame').get_parameter_value().string_value
        self.b_frame = p('b_frame').get_parameter_value().string_value

        # 파라미터 API 호환 처리
        tf_topics_param = self.get_parameters(['tf_topics'])[0].get_parameter_value()
        try:
            self.tf_topics = [s for s in tf_topics_param.string_array_value]
        except Exception:
            self.tf_topics = ['/robot1/tf','/robot1/tf_static','/robot2/tf','/robot2/tf_static']

        self.tf_poll_hz = p('tf_poll_hz').get_parameter_value().double_value
        if self.tf_poll_hz <= 0.0:
            self.tf_poll_hz = 50.0

        # ---- 액션 클라이언트 (Nav2 Planner) ----
        self.plan_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')

        # ---- 퍼블리셔 ----
        self.path_pub_front = self.create_publisher(Path, 'planned_path_front', 10)
        self.path_pub_back  = self.create_publisher(Path, 'planned_path_back', 10)
        self.path_pub       = self.create_publisher(Path, 'planned_path', 10)

        self.path_pub_a = self.create_publisher(Path, self.a_path_topic, 10)
        self.path_pub_b = self.create_publisher(Path, self.b_path_topic, 10)

        self.marker_pub = self.create_publisher(Marker, 'object_marker', 1)
        self.pose_pub = self.create_publisher(PoseStamped, 'object_pose', 10)

        # a/b 포즈를 이 노드가 TF에서 뽑아서 퍼블리시
        self.pub_a_pose = self.create_publisher(PoseStamped, self.a_topic, 10)
        self.pub_b_pose = self.create_publisher(PoseStamped, self.b_topic, 10)

        # TF 브로드캐스터(옵션)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.broadcast_tf else None

        # 이 노드도 a/b 토픽을 구독해서 기존 로직 재사용(루프백)
        self.pose_a_topic = None
        self.pose_b_topic = None
        self.sub_a = self.create_subscription(PoseStamped, self.a_topic, self.on_a_pose, 10)
        self.sub_b = self.create_subscription(PoseStamped, self.b_topic, self.on_b_pose, 10)

        # ---- TF Buffer/Listener (멀티 토픽) ----
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.multi_tf_listener = MultiTfListener(self, self.tf_buffer, self.tf_topics)

        # ---- 기타 ----
        self.center_pose_pub = self.create_publisher(PoseArray, 'planned_path_poses', 10)

        # CSV 저장 옵션
        self.declare_parameter('export_csv', False)
        self.declare_parameter('export_dir', '~/ros2_ws/exports')
        self.declare_parameter('export_prefix', 'object_plan')
        self.export_csv = p('export_csv').get_parameter_value().bool_value
        self.export_dir = os.path.expanduser(p('export_dir').get_parameter_value().string_value)
        self.export_prefix = p('export_prefix').get_parameter_value().string_value
        os.makedirs(self.export_dir, exist_ok=True)

        # 상태
        self.initialized = False
        self.last_pose = None
        self.pose_a = None
        self.pose_b = None
        self.goal_front = None
        self.goal_back  = None

        # TF 폴링 타이머: a/b 프레임을 world_frame 기준으로 읽어 a_topic/b_topic에 퍼블리시
        self.tf_timer = self.create_timer(1.0 / self.tf_poll_hz, self._tick_tf)

        # goal/초기화 입력
        self.init_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.on_initialpose, 10
        )
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability  = DurabilityPolicy.VOLATILE
        self.goal_sub_mb   = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.on_goal_center, qos)
        self.goal_sub_nav2 = self.create_subscription(PoseStamped, '/goal_pose',            self.on_goal_center, qos)

        self.get_logger().info(
            'ObjectPlannerExecutor ready (TF→/a/ee_pose, /b/ee_pose 자동 퍼블리시).\n'
            f'  world_frame={self.world_frame}, a_frame={self.a_frame}, b_frame={self.b_frame}\n'
            f'  tf_topics={list(self.tf_topics)}'
        )

    # === TF에서 즉시 PoseStamped 하나를 뽑는 헬퍼 ===
    def _lookup_pose_from_tf(self, frame_id: str, timeout_sec: float = 0.5) -> PoseStamped | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, frame_id, rclpy.time.Time(),
                rclpy.duration.Duration(seconds=timeout_sec)
            )
        except TransformException as e:
            self.get_logger().warn(f'TF lookup 실패: {self.world_frame} -> {frame_id}: {e}')
            return None

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x = tf.transform.translation.x
        ps.pose.position.y = tf.transform.translation.y
        ps.pose.position.z = tf.transform.translation.z
        ps.pose.orientation = tf.transform.rotation
        return ps

    # ===== TF 폴링 → a/b PoseStamped 퍼블리시 =====
    def _tick_tf(self):
        now = rclpy.time.Time()  # 최신 시간 기준 조회
        for which in ('a', 'b'):
            frame = self.a_frame if which == 'a' else self.b_frame
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.world_frame, frame, now, rclpy.duration.Duration(seconds=0.05)
                )
            except TransformException:
                continue

            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = self.world_frame
            ps.pose.position.x = tf.transform.translation.x
            ps.pose.position.y = tf.transform.translation.y
            ps.pose.position.z = tf.transform.translation.z
            ps.pose.orientation = tf.transform.rotation

            if which == 'a':
                self.pub_a_pose.publish(ps)
                self.pose_a_topic = ps
            else:
                self.pub_b_pose.publish(ps)
                self.pose_b_topic = ps

    # ===== CSV 유틸 =====
    def _path_to_rows(self, path: Path):
        rows = []
        for i, ps in enumerate(path.poses):
            x = ps.pose.position.x
            y = ps.pose.position.y
            z = ps.pose.position.z
            yaw = quat_to_yaw(ps.pose.orientation)
            sec = getattr(ps.header.stamp, 'sec', 0)
            nsec = getattr(ps.header.stamp, 'nanosec', 0)
            t = sec + nsec * 1e-9
            rows.append([i, x, y, z, yaw, yaw * 180.0 / math.pi, t])
        return rows

    def _save_csv(self, filepath: str, rows: list[list]):
        with open(filepath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['index', 'x', 'y', 'z', 'yaw_rad', 'yaw_deg', 't_sec'])
            w.writerows(rows)

    def _export_path_csv(self, name: str, path: Path):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.export_dir, f'{self.export_prefix}_{name}_{ts}.csv')
        rows = self._path_to_rows(path)
        self._save_csv(filepath, rows)
        self.get_logger().info(f'[CSV] Saved: {filepath}')

    # ===== 입력(A/B) 수신 =====
    def on_a_pose(self, msg: PoseStamped):
        self.pose_a_topic = msg

    def on_b_pose(self, msg: PoseStamped):
        self.pose_b_topic = msg

    def _update_ab_poses(self) -> bool:
        """
        1) 먼저 현재 토픽값 사용
        2) 하나라도 None이면 TF로 즉시 보충 조회
        """
        if self.pose_a_topic is None:
            ps_a = self._lookup_pose_from_tf(self.a_frame, timeout_sec=0.5)
            if ps_a is not None:
                self.pose_a_topic = ps_a

        if self.pose_b_topic is None:
            ps_b = self._lookup_pose_from_tf(self.b_frame, timeout_sec=0.5)
            if ps_b is not None:
                self.pose_b_topic = ps_b

        self.pose_a = self.pose_a_topic
        self.pose_b = self.pose_b_topic
        ok = (self.pose_a is not None and self.pose_b is not None)
        if not ok:
            self.get_logger().error('A/B 포즈를 확보하지 못했습니다. (TF/프레임/월드프레임 설정 확인)')
        return ok

    def _interp(self, a, b, t):
        return a + (b - a) * t

    # -------------------------------
    # 각도 유틸
    # -------------------------------
    def _wrap_to_pi(self, ang: float) -> float:
        return (ang + math.pi) % (2 * math.pi) - math.pi

    def _unwrap_angles(self, thetas: list[float]) -> list[float]:
        out = []
        prev = None
        for th in thetas:
            if prev is None:
                out.append(th); prev = th; continue
            delta = self._wrap_to_pi(th - prev)
            prev = prev + delta
            out.append(prev)
        return out

    def _shortest_angle_interp(self, th0: float, th1: float, t: float) -> float:
        delta = self._wrap_to_pi(th1 - th0)
        return th0 + delta * t

    # ------------------------------------------------------------
    # SE(2) 혼합 메트릭 리샘플
    # ------------------------------------------------------------
    def _resample_center_path_se2(self, center_path: Path, ds: float = 0.05, w_theta: float | None = None) -> Path:
        n = len(center_path.poses)
        if n < 2:
            return center_path

        xs, ys, th_raw = [], [], []
        for ps in center_path.poses:
            xs.append(ps.pose.position.x)
            ys.append(ps.pose.position.y)
        for ps in center_path.poses:
            th_raw.append(quat_to_yaw(ps.pose.orientation))
        thetas = self._unwrap_angles(th_raw)

        if w_theta is None:
            w_theta = 0.5 * self.L

        u = [0.0]
        for i in range(1, n):
            dx = xs[i] - xs[i-1]; dy = ys[i] - ys[i-1]
            dpos = math.hypot(dx, dy)
            dth  = abs(self._wrap_to_pi(thetas[i] - thetas[i-1]))
            u.append(u[-1] + dpos + w_theta * dth)

        total = u[-1]
        if total < 1e-9:
            return center_path

        m = max(2, int(total / ds) + 1)
        u_new = [i * total / (m - 1) for i in range(m)]

        xs_new, ys_new, th_new = [], [], []
        j = 0
        for ui in u_new:
            while j + 1 < n and u[j+1] < ui:
                j += 1
            if j + 1 >= n:
                xs_new.append(xs[-1]); ys_new.append(ys[-1]); th_new.append(thetas[-1])
                continue
            den = max(1e-12, (u[j+1] - u[j]))
            t = (ui - u[j]) / den
            xs_new.append(self._interp(xs[j], xs[j+1], t))
            ys_new.append(self._interp(ys[j], ys[j+1], t))
            th_new.append(self._shortest_angle_interp(thetas[j], thetas[j+1], t))

        out = Path()
        out.header = center_path.header
        for i in range(m):
            ps = PoseStamped()
            ps.header = center_path.header
            ps.pose.position.x = xs_new[i]
            ps.pose.position.y = ys_new[i]
            ps.pose.position.z = center_path.poses[0].pose.position.z
            ps.pose.orientation = yaw_to_quat(th_new[i])
            out.poses.append(ps)
        return out

    # ---- 기존 x,y 기준 리샘플(백업) ----
    def _resample_center_path(self, center_path: Path, ds: float = 0.05) -> Path:
        n = len(center_path.poses)
        if n < 2:
            return center_path
        xs, ys = [], []
        for ps in center_path.poses:
            xs.append(ps.pose.position.x)
            ys.append(ps.pose.position.y)
        s = [0.0]
        for i in range(1, n):
            s.append(s[-1] + math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]))
        total = s[-1]
        if total < 1e-6:
            return center_path
        m = max(2, int(total / ds) + 1)
        s_new = [i * total / (m - 1) for i in range(m)]
        xs_new, ys_new = [], []
        j = 0
        for si in s_new:
            while j + 1 < n and s[j+1] < si:
                j += 1
            if j + 1 >= n:
                xs_new.append(xs[-1]); ys_new.append(ys[-1]); continue
            t = (si - s[j]) / max(1e-9, (s[j+1] - s[j]))
            xs_new.append(xs[j] + (xs[j+1]-xs[j])*t)
            ys_new.append(ys[j] + (ys[j+1]-ys[j])*t)
        yaws, prev = [], None
        for i in range(len(xs_new)):
            i0 = max(0, i-1); i1 = min(len(xs_new)-1, i+1)
            dx = xs_new[i1] - xs_new[i0]
            dy = ys_new[i1] - ys_new[i0]
            yaw = math.atan2(dy, dx) if (abs(dx)+abs(dy)) > 1e-12 else (yaws[-1] if yaws else 0.0)
            if prev is not None:
                delta = (yaw - prev + math.pi) % (2*math.pi) - math.pi
                yaw = prev + delta
            yaws.append(yaw); prev = yaw
        out = Path(); out.header = center_path.header
        for i in range(len(xs_new)):
            ps = PoseStamped(); ps.header = center_path.header
            ps.pose.position.x = xs_new[i]
            ps.pose.position.y = ys_new[i]
            ps.pose.position.z = center_path.poses[0].pose.position.z
            q = Quaternion(); q.x=0.0; q.y=0.0; q.z=math.sin(yaws[i]*0.5); q.w=math.cos(yaws[i]*0.5)
            ps.pose.orientation = q
            out.poses.append(ps)
        return out

    def _offset_front_back_from_center(self, center_path: Path):
        if center_path is None or len(center_path.poses) == 0:
            return None
        path_front, path_back = Path(), Path()
        path_front.header = center_path.header
        path_back.header  = center_path.header
        for ps in center_path.poses:
            yaw_c = quat_to_yaw(ps.pose.orientation)
            dx = 0.5 * self.L * math.cos(yaw_c)
            dy = 0.5 * self.L * math.sin(yaw_c)
            pf = PoseStamped(); pf.header = ps.header
            pf.pose.position.x = ps.pose.position.x + dx
            pf.pose.position.y = ps.pose.position.y + dy
            pf.pose.position.z = ps.pose.position.z
            pf.pose.orientation = yaw_to_quat(self._wrap_to_pi(yaw_c + math.pi))  # 중심을 향하도록
            pb = PoseStamped(); pb.header = ps.header
            pb.pose.position.x = ps.pose.position.x - dx
            pb.pose.position.y = ps.pose.position.y - dy
            pb.pose.position.z = ps.pose.position.z
            pb.pose.orientation = yaw_to_quat(yaw_c)
            path_front.poses.append(pf)
            path_back.poses.append(pb)
        return path_front, path_back

    # ===== yaw 연속화 =====
    def _unwrap(self, prev: float | None, curr: float) -> float:
        if prev is None:
            return curr
        delta = (curr - prev + math.pi) % (2 * math.pi) - math.pi
        return prev + delta

    # ===== 시각화 =====
    def publish_path_marker(self, path: Path, front: bool):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f'{self.model_name}_path_{"front" if front else "back"}'
        m.id = 100 if front else 101
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        if front:
            m.color.r, m.color.g, m.color.b = 0.95, 0.10, 0.10
        else:
            m.color.r, m.color.g, m.color.b = 0.10, 0.30, 0.95
        m.color.a = 1.0
        m.pose.orientation.w = 1.0
        m.points = [Point(x=ps.pose.position.x, y=ps.pose.position.y, z=ps.pose.position.z)
                    for ps in path.poses]
        self.marker_pub.publish(m)

    def publish_path_heading(self, path: Path, front: bool, step: int = 6, length: float | None = None):
        if length is None:
            length = 0.4 * self.L
        head_len = 0.3 * length
        head_deg = 25.0
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f'{self.model_name}_heading_{"front" if front else "back"}'
        m.id = 110 if front else 111
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.03
        if front:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.95, 0.10, 0.10, 1.0
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.10, 0.30, 0.95, 1.0
        m.pose.orientation.w = 1.0
        m.points = []
        if len(path.poses) == 0:
            self.marker_pub.publish(m); return
        rad = math.radians
        for i in range(0, len(path.poses), max(1, step)):
            ps = path.poses[i]
            x = ps.pose.position.x; y = ps.pose.position.y; z = ps.pose.position.z
            yaw = quat_to_yaw(ps.pose.orientation)
            x2 = x + length * math.cos(yaw); y2 = y + length * math.sin(yaw)
            m.points.append(Point(x=x, y=y, z=z)); m.points.append(Point(x=x2, y=y2, z=z))
            left = yaw + math.pi - rad(head_deg); right = yaw + math.pi + rad(head_deg)
            xL = x2 + head_len * math.cos(left);  yL = y2 + head_len * math.sin(left)
            xR = x2 + head_len * math.cos(right); yR = y2 + head_len * math.sin(right)
            m.points.append(Point(x=x2, y=y2, z=z)); m.points.append(Point(x=xL, y=yL, z=z))
            m.points.append(Point(x=x2, y=y2, z=z)); m.points.append(Point(x=xR, y=yR, z=z))
        # 필요 시 self.marker_pub.publish(m)

    def publish_marker(self, pose):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.model_name
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose = pose
        m.scale.x = float(self.L)
        m.scale.y = float(self.W)
        m.scale.z = float(self.H)
        m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.27, 0.07, 1.0
        self.marker_pub.publish(m)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.world_frame
            t.child_frame_id = f'{self.model_name}_frame'
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(t)

        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        self.pose_pub.publish(ps)

    def publish_goal_marker(self, ps: PoseStamped, idx: int, front: bool):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f'{self.model_name}_goals'
        m.id = idx
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = ps.pose
        m.scale.x = m.scale.y = m.scale.z = 0.15
        if front:
            m.color.r, m.color.g, m.color.b = 0.95, 0.10, 0.10
        else:
            m.color.r, m.color.g, m.color.b = 0.10, 0.30, 0.95
        m.color.a = 1.0
        self.marker_pub.publish(m)

    # ===== A/B → 물체 중심/자세 계산 =====
    def _compose_object_pose_from_ab(self) -> PoseStamped | None:
        if not self._update_ab_poses():
            return None
        ax, ay, az = self.pose_a.pose.position.x, self.pose_a.pose.position.y, self.pose_a.pose.position.z
        bx, by, bz = self.pose_b.pose.position.x, self.pose_b.pose.position.y, self.pose_b.pose.position.z
        cx, cy, cz = (ax + bx) * 0.5, (ay + by) * 0.5, (az + bz) * 0.5
        yaw = math.atan2(by - ay, bx - ax)  # A→B 방향을 물체 +x로 간주
        dist = math.hypot(bx - ax, by - ay)
        if abs(dist - self.L) > 0.15 * self.L:
            self.get_logger().warn(f'A-B 거리({dist:.2f} m)가 물체 길이 L({self.L:.2f} m)와 많이 다릅니다.')
        pose = PoseStamped()
        pose.header.frame_id = self.world_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = cx
        pose.pose.position.y = cy
        pose.pose.position.z = cz
        pose.pose.orientation = yaw_to_quat(yaw)
        return pose

    # ===== /initialpose: 트리거 =====
    def on_initialpose(self, _: PoseWithCovarianceStamped):
        # 초기화 직전에 한번 강제 갱신 시도
        if not self._update_ab_poses():
            time.sleep(0.1)  # 타이머가 퍼블리시할 시간 한 박자 대기
            if not self._update_ab_poses():
                self.get_logger().error('초기화 실패: A/B 포즈를 확보하지 못했습니다.')
                return

        obj_ps = self._compose_object_pose_from_ab()
        if obj_ps is None:
            self.get_logger().error('초기화 실패: A/B 포즈를 확보하지 못했습니다.')
            return

        self.initialized = True
        self.last_pose = obj_ps.pose
        self.publish_marker(obj_ps.pose)

        yaw = quat_to_yaw(obj_ps.pose.orientation)
        dx = (self.L * 0.5) * math.cos(yaw)
        dy = (self.L * 0.5) * math.sin(yaw)

        self.goal_front = PoseStamped()
        self.goal_front.header.frame_id = self.world_frame
        self.goal_front.header.stamp = self.get_clock().now().to_msg()
        self.goal_front.pose.position.x = obj_ps.pose.position.x + dx
        self.goal_front.pose.position.y = obj_ps.pose.position.y + dy
        self.goal_front.pose.position.z = obj_ps.pose.position.z
        self.goal_front.pose.orientation = yaw_to_quat(yaw)

        self.goal_back = PoseStamped()
        self.goal_back.header.frame_id = self.world_frame
        self.goal_back.header.stamp = self.get_clock().now().to_msg()
        self.goal_back.pose.position.x = obj_ps.pose.position.x - dx
        self.goal_back.pose.position.y = obj_ps.pose.position.y - dy
        self.goal_back.pose.position.z = obj_ps.pose.position.z
        self.goal_back.pose.orientation = yaw_to_quat(yaw)

        self.publish_goal_marker(self.goal_front, idx=1, front=True)
        self.publish_goal_marker(self.goal_back,  idx=2, front=False)
        self.get_logger().info('초기화 완료. goal을 지정해 중심 경로를 생성하세요.')

    # ===== goal 입력 → 경로/분배 =====
    def on_goal_center(self, goal_center: PoseStamped):
        if not self.plan_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('planner_server action not available.')
            return
        if self.last_pose is None:
            obj_ps = self._compose_object_pose_from_ab()
            if obj_ps is None:
                self.get_logger().error('현재 중심을 알 수 없습니다. /initialpose 또는 TF/A/B 토픽을 확인하세요.')
                return
            self.last_pose = obj_ps.pose
            self.publish_marker(self.last_pose)

        goal = ComputePathToPose.Goal()
        goal.goal = goal_center

        start = PoseStamped()
        start.header.frame_id = self.world_frame
        start.header.stamp = self.get_clock().now().to_msg()
        start.pose = self.last_pose
        goal.start = start
        goal.use_start = True
        goal.planner_id = 'GridBased'

        send_future = self.plan_client.send_goal_async(goal)

        def _result_cb(fut):
            try:
                gh = fut.result()
            except Exception as e:
                self.get_logger().error(f'[center] Action error: {e}')
                return
            if gh is None or not getattr(gh, "accepted", True):
                self.get_logger().error('[center] Goal이 거부되었습니다.')
                return
            res_future = gh.get_result_async()

            def _res_done(rf):
                try:
                    res = rf.result()
                except Exception as e:
                    self.get_logger().error(f'[center] Result error: {e}')
                    return
                result = getattr(res, 'result', None)
                if result is None or result.path is None or len(result.path.poses) == 0:
                    self.get_logger().warn('[center] 빈 경로가 반환되었습니다.')
                    return

                center_path = result.path
                center_path = self._resample_center_path_se2(center_path, ds=self.ds, w_theta=0.5*self.L)
                self.path_pub.publish(center_path)

                pa = PoseArray()
                pa.header.frame_id = self.world_frame
                pa.header.stamp = self.get_clock().now().to_msg()
                pa.poses = [ps.pose for ps in center_path.poses]
                self.center_pose_pub.publish(pa)

                fb = self._offset_front_back_from_center(center_path)
                if fb is None:
                    self.get_logger().warn('front/back 경로 생성 실패.')
                    return
                path_front, path_back = fb

                self.path_pub_front.publish(path_front)
                self.path_pub_back.publish(path_back)

                if self.show_heading_markers:
                    self.publish_path_heading(path_front, front=True,  step=6, length=0.4*self.L)
                    self.publish_path_heading(path_back,  front=False, step=6, length=0.4*self.L)

                path_a, path_b, mapping = self._select_paths_for_AB(path_front, path_back)
                self.path_pub_a.publish(path_a)
                self.path_pub_b.publish(path_b)
                self.get_logger().info(f'[center] 경로 할당: {mapping}')

                self.goal_front = path_front.poses[-1]
                self.goal_back  = path_back.poses[-1]
                self.publish_goal_marker(self.goal_front, idx=11, front=True)
                self.publish_goal_marker(self.goal_back,  idx=12, front=False)

                if self.simulate_motion:
                    self.follow_path(center_path)

                self.last_pose = center_path.poses[-1].pose
                self.get_logger().info(f"[center] path poses = {len(center_path.poses)}")
                ds_live = self.get_parameter('ds').get_parameter_value().double_value
                self.get_logger().info(f'[resample] ds={ds_live:.4f} m')

                if self.export_csv:
                    self._export_path_csv('center', center_path)
                    self._export_path_csv('ee_A',  path_a)
                    self._export_path_csv('ee_B',  path_b)
                    self._export_path_csv('front', path_front)
                    self._export_path_csv('back',  path_back)

            res_future = gh.get_result_async()
            res_future.add_done_callback(_res_done)

        send_future.add_done_callback(_result_cb)

    # ---- A/B 경로 매핑 ----
    def _select_paths_for_AB(self, path_front: Path, path_back: Path):
        mapping = 'A→front, B→back (fallback)'
        if not self._update_ab_poses():
            return path_front, path_back, mapping
        f0 = path_front.poses[0].pose.position
        b0 = path_back.poses[0].pose.position
        a = self.pose_a.pose.position
        b = self.pose_b.pose.position
        d_af = math.hypot(a.x - f0.x, a.y - f0.y)
        d_ab = math.hypot(a.x - b0.x, a.y - b0.y)
        d_bf = math.hypot(b.x - f0.x, b.y - f0.y)
        d_bb = math.hypot(b.x - b0.x, b.y - b0.y)
        cost_Af_Bb = d_af + d_bb
        cost_Ab_Bf = d_ab + d_bf
        if cost_Af_Bb <= cost_Ab_Bf:
            mapping = f"A→front({d_af:.2f}m), B→back({d_bb:.2f}m)"
            return path_front, path_back, mapping
        else:
            mapping = f"A→back({d_ab:.2f}m), B→front({d_bf:.2f}m)"
            return path_back, path_front, mapping

    # ---- 마커 기반 이동 ----
    def follow_path(self, path: Path):
        if len(path.poses) == 0:
            return
        for ps in path.poses:
            pose = deepcopy(ps.pose)
            self.publish_marker(pose)
            self.last_pose = pose
            time.sleep(float(self.dt))
        self.get_logger().info('Arrived at goal.')


def main():
    rclpy.init()
    node = ObjectPlannerExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
