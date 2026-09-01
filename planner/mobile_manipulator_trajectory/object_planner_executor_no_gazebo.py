# src/mobile_manipulator_trajectory/object_planner_executor.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import time, math
from copy import deepcopy

from geometry_msgs.msg import (
    PoseWithCovarianceStamped, PoseStamped, Quaternion, TransformStamped, Point
)

from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray   # ← 추가


import tf2_ros

import csv, os
from datetime import datetime


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


class ObjectPlannerExecutor(Node):
    def __init__(self):
        super().__init__('object_planner_executor')

        # ---- 기본 파라미터 ----
        self.declare_parameter('model_name', 'rect_object')
        # 사이즈는 런치에서 주입된다고 가정(미주입 시 예외 방지 위해 기본도 선언)
        self.declare_parameter('size_l', 1.2)   # 물체 길이 (front-back 간 거리)
        self.declare_parameter('size_w', 0.8)
        self.declare_parameter('size_h', 0.25)
        self.declare_parameter('speed', 0.3)
        self.declare_parameter('dt', 0.03)
        self.declare_parameter('broadcast_tf', False)
        self.declare_parameter('simulate_motion', True)        # 마커만 따라가며 이동 시뮬
        self.declare_parameter('show_heading_markers', False)   # ← 새 파라미터: 화살표 on/off
        self.declare_parameter('world_frame', 'map')

        # ---- 입력/출력 토픽 이름 ----
        self.declare_parameter('a_topic', '/a/ee_pose')
        self.declare_parameter('b_topic', '/b/ee_pose')
        self.declare_parameter('a_path_topic', '/a/ee_path')
        self.declare_parameter('b_path_topic', '/b/ee_path')
        self.declare_parameter('ds', 0.01)  # ⬅️ 샘플 간격 기본값 1cm

        p = self.get_parameter
        self.model_name = p('model_name').get_parameter_value().string_value
        self.L = p('size_l').get_parameter_value().double_value
        self.W = p('size_w').get_parameter_value().double_value
        self.H = p('size_h').get_parameter_value().double_value
        self.speed = p('speed').get_parameter_value().double_value
        self.dt = p('dt').get_parameter_value().double_value
        self.broadcast_tf = p('broadcast_tf').get_parameter_value().bool_value
        self.simulate_motion = p('simulate_motion').get_parameter_value().bool_value
        self.show_heading_markers = p('show_heading_markers').get_parameter_value().bool_value  # ← 읽기
        self.world_frame = p('world_frame').get_parameter_value().string_value

        self.a_topic = p('a_topic').get_parameter_value().string_value
        self.b_topic = p('b_topic').get_parameter_value().string_value
        self.a_path_topic = p('a_path_topic').get_parameter_value().string_value
        self.b_path_topic = p('b_path_topic').get_parameter_value().string_value
        self.ds = p('ds').get_parameter_value().double_value  # ⬅️ ds 보관

        # ---- 액션 클라이언트 (Nav2 Planner) ----
        self.plan_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')

        # ---- 구독/퍼블리셔 ----
        self.init_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.on_initialpose, 10
        )
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        self.goal_sub_mb = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.on_goal_center, qos)
        self.goal_sub_nav2 = self.create_subscription(PoseStamped, '/goal_pose', self.on_goal_center, qos)

        # 경로 퍼블리셔 (시각화용)
        self.path_pub_front = self.create_publisher(Path, 'planned_path_front', 10)  # 앞(빨강)
        self.path_pub_back  = self.create_publisher(Path, 'planned_path_back', 10)   # 뒤(파랑)
        self.path_pub       = self.create_publisher(Path, 'planned_path', 10)        # 중심 경로

        # 제어용: A/B 로봇에 전달될 EE 경로
        self.path_pub_a = self.create_publisher(Path, self.a_path_topic, 10)
        self.path_pub_b = self.create_publisher(Path, self.b_path_topic, 10)

        # 물체 마커/포즈
        self.marker_pub = self.create_publisher(Marker, 'object_marker', 1)
        self.pose_pub = self.create_publisher(PoseStamped, 'object_pose', 10)

        # TF (출력용 브로드캐스트만 유지)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.broadcast_tf else None

        # 토픽 입력 구독
        self.pose_a_topic = None
        self.pose_b_topic = None
        self.sub_a = self.create_subscription(PoseStamped, self.a_topic, self.on_a_pose, 10)
        self.sub_b = self.create_subscription(PoseStamped, self.b_topic, self.on_b_pose, 10)

        # 상태
        self.initialized = False
        self.last_pose = None      # 물체 현재 포즈(중심)
        self.pose_a = None         # A EE 최신 포즈
        self.pose_b = None         # B EE 최신 포즈
        self.goal_front = None     # front/back 마지막 점(표시용)
        self.goal_back  = None

        self.get_logger().info(
            'ObjectPlannerExecutor ready.\n'
            ' - goal(물체 중심) → Nav2 중심 경로 계산 → 각 점에서 front/back 오프셋 경로 생성\n'
            ' - A/B 경로 할당은 초기 위치에서 가까운 쪽으로 자동 매핑합니다.'
        )



        self.declare_parameter('export_csv', False)                # CSV 저장 on/off
        self.declare_parameter('export_dir', '~/ros2_ws/exports') # 저장 폴더
        self.declare_parameter('export_prefix', 'object_plan')    # 파일명 접두사

        # __init__ 안, p = self.get_parameter 이후 값 읽기
        self.export_csv = p('export_csv').get_parameter_value().bool_value
        self.export_dir = os.path.expanduser(p('export_dir').get_parameter_value().string_value)
        self.export_prefix = p('export_prefix').get_parameter_value().string_value
        os.makedirs(self.export_dir, exist_ok=True)

        self.center_pose_pub = self.create_publisher(PoseArray, 'planned_path_poses', 10)  # ← 추가


        

    def _path_to_rows(self, path: Path):
        """Path -> [index, x, y, z, yaw(rad), yaw(deg), t_sec] 리스트로 변환"""
        rows = []
        for i, ps in enumerate(path.poses):
            x = ps.pose.position.x
            y = ps.pose.position.y
            z = ps.pose.position.z
            yaw = quat_to_yaw(ps.pose.orientation)
            # 포즈별 stamp가 없을 수 있으니 안전하게 처리
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
        """지정한 Path를 {export_dir}/{prefix}_{name}_{timestamp}.csv 로 저장"""
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
        """항상 토픽 입력으로 A/B 포즈 갱신."""
        self.pose_a = self.pose_a_topic
        self.pose_b = self.pose_b_topic
        ok = (self.pose_a is not None and self.pose_b is not None)
        if not ok:
            self.get_logger().warn('A/B EE 포즈를 둘 다 얻지 못했습니다.')
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
    # A안: SE(2) 혼합 메트릭 기반 리샘플 (이동 + 회전 모두 "길이"로 반영)
    # ------------------------------------------------------------
    def _resample_center_path_se2(self, center_path: Path, ds: float = 0.05, w_theta: float | None = None) -> Path:
        """
        SE(2) 혼합 메트릭으로 중심 경로를 등간격 리샘플링.
        u = 이동거리 + w_theta * |Δyaw| 를 누적 길이로 사용.
        yaw은 orientation이 없거나 신뢰하기 어렵다면 (x,y) 기하 방향으로 보정, 이후 unwrap.
        """
        n = len(center_path.poses)
        if n < 2:
            return center_path

        # 1) 원본 x,y 와 yaw(가능하면 orientation, 없으면 기하 방향) 수집
        xs, ys, th_raw = [], [], []
        for i, ps in enumerate(center_path.poses):
            x = ps.pose.position.x
            y = ps.pose.position.y
            xs.append(x); ys.append(y)

        for i, ps in enumerate(center_path.poses):
            # 우선 쿼터니안 yaw
            th_q = quat_to_yaw(ps.pose.orientation)
            # 기본값: 쿼터니안
            th_raw.append(th_q)

        # unwrap으로 연속 yaw 구성
        thetas = self._unwrap_angles(th_raw)

        # 2) 혼합 누적 길이 u 계산 (이동 + 회전*가중치)
        if w_theta is None:
            w_theta = 0.5 * self.L  # 요청: size_l/2

        u = [0.0]
        for i in range(1, n):
            dx = xs[i] - xs[i-1]; dy = ys[i] - ys[i-1]
            dpos = math.hypot(dx, dy)
            dth  = abs(self._wrap_to_pi(thetas[i] - thetas[i-1]))
            u.append(u[-1] + dpos + w_theta * dth)

        total = u[-1]
        if total < 1e-9:
            return center_path

        # 3) u 등간격 리샘플
        m = max(2, int(total / ds) + 1)
        u_new = [i * total / (m - 1) for i in range(m)]

        # 4) 1D 보간 (x,y 선형, theta는 shortest-angle 보간)
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

        # 5) Path 생성 (θ→quat)
        out = Path()
        out.header = center_path.header
        for i in range(m):
            ps = PoseStamped()
            ps.header = center_path.header
            ps.pose.position.x = xs_new[i]
            ps.pose.position.y = ys_new[i]
            ps.pose.position.z = center_path.poses[0].pose.position.z
            th = th_new[i]
            ps.pose.orientation = yaw_to_quat(th)
            out.poses.append(ps)
        return out


    # def _resample_center_path_se2(self, center_path: Path, ds: float = 0.05, w_theta: float | None = None) -> Path:
    #     """
    #     SE(2) 혼합 메트릭으로 중심 경로를 등간격 리샘플링.
    #     u = 이동거리 + w_theta * |Δyaw| 를 누적 길이로 사용.
    #     yaw은 orientation이 없거나 신뢰하기 어렵다면 (x,y) 기하 방향으로 보정, 이후 unwrap.
    #     """
    #     n = len(center_path.poses)
    #     if n < 2:
    #         return center_path

    #     # 1) 원본 x,y 와 yaw(가능하면 orientation, 없으면 기하 방향) 수집
    #     xs, ys, th_raw = [], [], []
    #     for i, ps in enumerate(center_path.poses):
    #         x = ps.pose.position.x
    #         y = ps.pose.position.y
    #         xs.append(x); ys.append(y)

    #     for i, ps in enumerate(center_path.poses):
    #         # 우선 쿼터니안 yaw
    #         th_q = quat_to_yaw(ps.pose.orientation)
    #         # orientation이 비거나(사실상 0,0,0,1) 기하가 더 신뢰되는 경우, 구간 방향으로 대체
    #         use_geom = False
    #         if abs(ps.pose.orientation.x) + abs(ps.pose.orientation.y) + abs(ps.pose.orientation.z) < 1e-9:
    #             use_geom = True

    #         if use_geom:
    #             if i + 1 < n:
    #                 dx = xs[i+1] - xs[i]; dy = ys[i+1] - ys[i]
    #                 if abs(dx) + abs(dy) > 1e-12:
    #                     th_raw.append(math.atan2(dy, dx))
    #                     continue
    #             if i - 1 >= 0:
    #                 dx = xs[i] - xs[i-1]; dy = ys[i] - ys[i-1]
    #                 if abs(dx) + abs(dy) > 1e-12:
    #                     th_raw.append(math.atan2(dy, dx))
    #                     continue
    #         # 기본값: 쿼터니안
    #         th_raw.append(th_q)

    #     # unwrap으로 연속 yaw 구성
    #     thetas = self._unwrap_angles(th_raw)

    #     # 2) 혼합 누적 길이 u 계산 (이동 + 회전*가중치)
    #     if w_theta is None:
    #         w_theta = 0.5 * self.L  # 요청: size_l/2

    #     u = [0.0]
    #     for i in range(1, n):
    #         dx = xs[i] - xs[i-1]; dy = ys[i] - ys[i-1]
    #         dpos = math.hypot(dx, dy)
    #         dth  = abs(self._wrap_to_pi(thetas[i] - thetas[i-1]))
    #         u.append(u[-1] + dpos + w_theta * dth)

    #     total = u[-1]
    #     if total < 1e-9:
    #         return center_path

    #     # 3) u 등간격 리샘플
    #     m = max(2, int(total / ds) + 1)
    #     u_new = [i * total / (m - 1) for i in range(m)]

    #     # 4) 1D 보간 (x,y 선형, theta는 shortest-angle 보간)
    #     xs_new, ys_new, th_new = [], [], []
    #     j = 0
    #     for k, ui in enumerate(u_new):
    #         # 마지막은 무조건 끝점 사용(경계 안정화)
    #         if k == len(u_new) - 1:
    #             xs_new.append(xs[-1]); ys_new.append(ys[-1]); th_new.append(thetas[-1])
    #             continue

    #         # 경계 안전화(+epsilon)로 구간 탐색
    #         while j + 1 < n and u[j+1] <= ui + 1e-12:
    #             j += 1
    #         if j + 1 >= n:
    #             xs_new.append(xs[-1]); ys_new.append(ys[-1]); th_new.append(thetas[-1])
    #             continue

    #         den = max(1e-12, (u[j+1] - u[j]))
    #         t = (ui - u[j]) / den

    #         xs_new.append(self._interp(xs[j], xs[j+1], t))
    #         ys_new.append(self._interp(ys[j], ys[j+1], t))
    #         # 참조 yaw는 unwrap 가정 하에 선형 보간
    #         th_new.append(self._shortest_angle_interp(thetas[j], thetas[j+1], t))

    #     # --- 하이브리드 yaw 최종 계산 ---
    #     # th_new = self._recompute_yaw_geometric_with_fallback(xs_new, ys_new, th_new, eps_xy=1e-6)


    #     # 5) Path 생성 (θ→quat)
    #     out = Path()
    #     out.header = center_path.header
    #     for i in range(m):
    #         ps = PoseStamped()
    #         ps.header = center_path.header
    #         ps.pose.position.x = xs_new[i]
    #         ps.pose.position.y = ys_new[i]
    #         ps.pose.position.z = center_path.poses[0].pose.position.z
    #         th = th_new[i]
    #         ps.pose.orientation = yaw_to_quat(th)
    #         out.poses.append(ps)
    #     return out

    # --- 0) 내부 헬퍼: 하이브리드 yaw 재계산 (이동 없으면 th_ref 사용) ---
    def _recompute_yaw_geometric_with_fallback(self, xs_new, ys_new, th_ref, eps_xy=1e-6):
        m = len(xs_new)
        yaws = []
        prev = None
        for i in range(m):
            # 중앙차분(가능 시), 아니면 전/후방 차분
            if 0 < i < m-1:
                dx = xs_new[i+1] - xs_new[i]
                dy = ys_new[i+1] - ys_new[i]
            else:
                dx = dy = 0.0

            if abs(dx) + abs(dy) > eps_xy:
                yaw = math.atan2(dy, dx)   # 기하 yaw
            else:
                yaw = th_ref[i]            # 순수 회전 구간: 참조 yaw

            # unwrap 연속성 유지
            if prev is not None:
                delta = (yaw - prev + math.pi) % (2*math.pi) - math.pi
                yaw = prev + delta
            yaws.append(yaw)
            prev = yaw
        return yaws


    # ---- (기존 x,y 기준 리샘플; 남겨둠: 필요 시 비교/백업용) ----
    def _resample_center_path(self, center_path: Path, ds: float = 0.05) -> Path:
        """중심 경로를 거리 등간격 ds(m)로 리샘플 + yaw 연속화해서 PoseStamped.orientation 채움."""
        n = len(center_path.poses)
        if n < 2:
            return center_path

        # 원 좌표/길이 적분
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

        # 등간격 샘플 지점
        m = max(2, int(total / ds) + 1)
        s_new = [i * total / (m - 1) for i in range(m)]

        # 선형 보간으로 (x,y) 구하기
        xs_new, ys_new = [], []
        j = 0
        for si in s_new:
            while j + 1 < n and s[j+1] < si:
                j += 1
            if j + 1 >= n:
                xs_new.append(xs[-1]); ys_new.append(ys[-1]); continue
            t = (si - s[j]) / max(1e-9, (s[j+1] - s[j]))
            xs_new.append(self._interp(xs[j], xs[j+1], t) if hasattr(self, "_interp") else xs[j] + (xs[j+1]-xs[j])*t)
            ys_new.append(self._interp(ys[j], ys[j+1], t) if hasattr(self, "_interp") else ys[j] + (ys[j+1]-ys[j])*t)

        # 중앙 차분으로 yaw 계산 + unwrap
        yaws = []
        prev_yaw = None
        m = len(xs_new)
        for i in range(m):
            i0 = max(0, i-1); i1 = min(m-1, i+1)
            dx = xs_new[i1] - xs_new[i0]
            dy = ys_new[i1] - ys_new[i0]
            yaw = math.atan2(dy, dx) if (abs(dx)+abs(dy)) > 1e-12 else (yaws[-1] if yaws else 0.0)
            # unwrap
            if prev_yaw is not None:
                delta = (yaw - prev_yaw + math.pi) % (2*math.pi) - math.pi
                yaw = prev_yaw + delta
            yaws.append(yaw); prev_yaw = yaw

        # Path 생성
        out = Path()
        out.header = center_path.header
        for i in range(m):
            ps = PoseStamped()
            ps.header = center_path.header
            ps.pose.position.x = xs_new[i]
            ps.pose.position.y = ys_new[i]
            ps.pose.position.z = center_path.poses[0].pose.position.z
            # yaw -> quat
            q = Quaternion()
            q.x = 0.0; q.y = 0.0
            q.z = math.sin(yaws[i]*0.5); q.w = math.cos(yaws[i]*0.5)
            ps.pose.orientation = q
            out.poses.append(ps)
        return out

    def _offset_front_back_from_center(self, center_path: Path) -> tuple[Path, Path] | None:
        """center_path의 orientation을 신뢰하고, ±L/2 오프셋으로 front/back 생성.
           ※ front(+dx,+dy) 포인트의 orientation은 '중심을 향하도록' yaw+π 로 설정."""
        if center_path is None or len(center_path.poses) == 0:
            return None
        path_front, path_back = Path(), Path()
        path_front.header = center_path.header
        path_back.header  = center_path.header

        for ps in center_path.poses:
            yaw_c = quat_to_yaw(ps.pose.orientation)
            dx = 0.5 * self.L * math.cos(yaw_c)
            dy = 0.5 * self.L * math.sin(yaw_c)

            # front: +offset, but heading faces center → yaw + π
            pf = PoseStamped(); pf.header = ps.header
            pf.pose.position.x = ps.pose.position.x + dx
            pf.pose.position.y = ps.pose.position.y + dy
            pf.pose.position.z = ps.pose.position.z
            yaw_pf = self._wrap_to_pi(yaw_c + math.pi)
            pf.pose.orientation = yaw_to_quat(yaw_pf)

            # back: -offset, heading = center yaw
            pb = PoseStamped(); pb.header = ps.header
            pb.pose.position.x = ps.pose.position.x - dx
            pb.pose.position.y = ps.pose.position.y - dy
            pb.pose.position.z = ps.pose.position.z
            pb.pose.orientation = yaw_to_quat(yaw_c)

            path_front.poses.append(pf)
            path_back.poses.append(pb)

        return path_front, path_back

    # ===== 유틸: yaw 연속화(unwrap) =====
    def _unwrap(self, prev: float | None, curr: float) -> float:
        if prev is None:
            return curr
        delta = (curr - prev + math.pi) % (2 * math.pi) - math.pi
        return prev + delta

    # ===== 물체 마커/경로 마커 =====
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
        """
        경로 orientation을 시각화: 일정 간격(step)마다 방향선 + 화살촉을 그림.
        - step: 샘플링 간격
        - length: 화살 길이 (기본: 물체 길이의 40%)
        """
        if length is None:
            length = 0.4 * self.L
        head_len = 0.3 * length     # 화살촉 길이
        head_deg = 25.0             # 화살촉 각도(deg)

        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f'{self.model_name}_heading_{"front" if front else "back"}'
        m.id = 110 if front else 111
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.03  # 선 두께

        if front:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.95, 0.10, 0.10, 1.0  # 빨강
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.10, 0.30, 0.95, 1.0  # 파랑

        m.pose.orientation.w = 1.0
        m.points = []

        if len(path.poses) == 0:
            self.marker_pub.publish(m)
            return

        rad = math.radians
        for i in range(0, len(path.poses), max(1, step)):
            ps = path.poses[i]
            x = ps.pose.position.x
            y = ps.pose.position.y
            z = ps.pose.position.z
            yaw = quat_to_yaw(ps.pose.orientation)

            # 본 화살(몸통)
            x2 = x + length * math.cos(yaw)
            y2 = y + length * math.sin(yaw)
            m.points.append(Point(x=x, y=y, z=z))
            m.points.append(Point(x=x2, y=y2, z=z))

            # 화살촉 (V자 두 날)
            left = yaw + math.pi - rad(head_deg)
            right = yaw + math.pi + rad(head_deg)
            xL = x2 + head_len * math.cos(left)
            yL = y2 + head_len * math.sin(left)
            xR = x2 + head_len * math.cos(right)
            yR = y2 + head_len * math.sin(right)
            # 왼쪽 날
            m.points.append(Point(x=x2, y=y2, z=z))
            m.points.append(Point(x=xL, y=yL, z=z))
            # 오른쪽 날
            m.points.append(Point(x=x2, y=y2, z=z))
            m.points.append(Point(x=xR, y=yR, z=z))

        # self.marker_pub.publish(m)

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
        yaw = math.atan2(by - ay, bx - ax)  # A→B 방향을 물체 +x(길이)로 간주

        # 유효성 체크: A-B 거리 ~ L
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
        obj_ps = self._compose_object_pose_from_ab()
        if obj_ps is None:
            self.get_logger().error('초기화 실패: A/B 포즈를 확보하지 못했습니다.')
            return

        self.initialized = True
        self.last_pose = obj_ps.pose
        self.publish_marker(obj_ps.pose)

        # front/back 마지막 점(표시용) 갱신 (두 방향 모두 중심 yaw 유지)
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

    # ===== 단일 goal 입력: '물체 중심' 경로 계산 후 front/back 경로 생성 =====
    def on_goal_center(self, goal_center: PoseStamped):
        """goal_center는 '물체 중심' 목표. 현재 중심(self.last_pose)→goal_center로 경로 계산 후
           각 경로점마다 ±L/2 오프셋하여 front/back 경로를 만든다.
           A/B 경로 할당은 초기 위치에서 가까운 조합을 선택한다."""
        if not self.plan_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('planner_server action not available.')
            return

        # 시작 포즈 준비: last_pose가 없으면 A/B로부터 추정
        if self.last_pose is None:
            obj_ps = self._compose_object_pose_from_ab()
            if obj_ps is None:
                self.get_logger().error('현재 중심을 알 수 없습니다. /initialpose 또는 A/B 토픽을 확인하세요.')
                return
            self.last_pose = obj_ps.pose
            self.publish_marker(self.last_pose)

        # ComputePathToPose: 중심 경로 요청
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
                # center_path = self._resample_center_path(center_path, ds=0.05)
                self.path_pub.publish(center_path)

                ## 경로 center pose
                pa = PoseArray()
                pa.header.frame_id = self.world_frame
                pa.header.stamp = self.get_clock().now().to_msg()
                pa.poses = [ps.pose for ps in center_path.poses]
                self.center_pose_pub.publish(pa)


                # ✅ 연속 yaw를 신뢰하여 오프셋 (front는 중심을 향하도록 yaw+π)
                fb = self._offset_front_back_from_center(center_path)
                if fb is None:
                    self.get_logger().warn('front/back 경로 생성 실패.')
                    return
                path_front, path_back = fb

                # 시각화 퍼블리시
                self.path_pub_front.publish(path_front)
                self.path_pub_back.publish(path_back)
                # self.publish_path_marker(path_front, front=True)
                # self.publish_path_marker(path_back, front=False)

                # 방향 화살표(토글)
                if self.show_heading_markers:
                    self.publish_path_heading(path_front, front=True,  step=6, length=0.4*self.L)
                    self.publish_path_heading(path_back,  front=False, step=6, length=0.4*self.L)

                # === A/B 할당: 초기 위치에서 가까운 조합 선택 ===
                path_a, path_b, mapping = self._select_paths_for_AB(path_front, path_back)
                self.path_pub_a.publish(path_a)
                self.path_pub_b.publish(path_b)
                self.get_logger().info(f'[center] 경로 할당: {mapping}')

                # 최종 front/back 마지막 점 마커 갱신(지오메트릭 의미 유지)
                self.goal_front = path_front.poses[-1]
                self.goal_back  = path_back.poses[-1]
                self.publish_goal_marker(self.goal_front, idx=11, front=True)
                self.publish_goal_marker(self.goal_back,  idx=12, front=False)

                # 시뮬 이동(옵션): 중심 경로 따라가며 큐브 이동
                if self.simulate_motion:
                    self.follow_path(center_path)

                # 현재 위치 갱신
                self.last_pose = center_path.poses[-1].pose
                self.get_logger().info(f"[center] path poses = {len(center_path.poses)} / front/back 생성 및 매핑 완료.")



                if self.export_csv:
                    self._export_path_csv('center', center_path)  # 중심 경로
                    self._export_path_csv('ee_A',  path_a)        # A EE 경로
                    self._export_path_csv('ee_B',  path_b)        # B EE 경로
                    # 필요하면 front/back도 저장
                    self._export_path_csv('front', path_front)
                    self._export_path_csv('back',  path_back)


            res_future.add_done_callback(_res_done)

        send_future.add_done_callback(_result_cb)

    # ---- 중심 경로를 front/back 두 경로로 분리 (yaw 연속화 적용) ----
    def _split_center_path_to_front_back(self, center_path: Path):
        if center_path is None or len(center_path.poses) == 0:
            return None

        n = len(center_path.poses)
        path_front = Path()
        path_back  = Path()
        path_front.header = center_path.header
        path_back.header  = center_path.header

        prev_yaw = None
        for i, ps in enumerate(center_path.poses):
            # 기본 yaw: 포즈의 쿼터니안
            yaw = quat_to_yaw(ps.pose.orientation)
            # 플래너가 orientation을 비워둔 경우(거의 [0,0,0,1]) → 세그먼트 진행 방향으로 보정
            if abs(ps.pose.orientation.x) + abs(ps.pose.orientation.y) + abs(ps.pose.orientation.z) < 1e-9:
                if i + 1 < n:
                    nx, ny = center_path.poses[i+1].pose.position.x, center_path.poses[i+1].pose.position.y
                    cx, cy = ps.pose.position.x, ps.pose.position.y
                    if (nx - cx) != 0.0 or (ny - cy) != 0.0:
                        yaw = math.atan2(ny - cy, nx - cx)
                elif prev_yaw is not None:
                    yaw = prev_yaw

            # ✅ 연속화로 π 점프 방지
            yaw = self._unwrap(prev_yaw, yaw)
            prev_yaw = yaw

            dx = 0.5 * self.L * math.cos(yaw)
            dy = 0.5 * self.L * math.sin(yaw)

            # front: +offset, 요구사항 반영 → 중심을 향하도록 yaw+π
            pf = PoseStamped()
            pf.header = ps.header
            pf.pose.position.x = ps.pose.position.x + dx
            pf.pose.position.y = ps.pose.position.y + dy
            pf.pose.position.z = ps.pose.position.z
            pf.pose.orientation = yaw_to_quat(self._wrap_to_pi(yaw + math.pi))

            # back: -offset, 중심 yaw 유지
            pb = PoseStamped()
            pb.header = ps.header
            pb.pose.position.x = ps.pose.position.x - dx
            pb.pose.position.y = ps.pose.position.y - dy
            pb.pose.position.z = ps.pose.position.z
            pb.pose.orientation = yaw_to_quat(yaw)

            path_front.poses.append(pf)
            path_back.poses.append(pb)

        return path_front, path_back

    # ---- A/B 경로 매핑(초기 위치에서 가까운 조합 선택) ----
    def _select_paths_for_AB(self, path_front: Path, path_back: Path):
        """A/B 현재 위치와 front/back 시작점 거리로 조합 선택.
        (A->front + B->back) vs (A->back + B->front) 중 합이 작은 쪽."""
        # 기본값: 실패 시 A->front, B->back
        mapping = 'A→front, B→back (fallback)'
        if not self._update_ab_poses():
            return path_front, path_back, mapping

        # 시작점
        f0 = path_front.poses[0].pose.position
        b0 = path_back.poses[0].pose.position
        # A/B 현재
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

    # ---- 마커 기반 이동(옵션: 중심 경로 따라 이동) ----
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
