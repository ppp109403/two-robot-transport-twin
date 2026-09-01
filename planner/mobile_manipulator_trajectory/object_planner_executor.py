# src/mobile_manipulator_trajectory/object_planner_executor.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from gazebo_msgs.srv import SpawnEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from builtin_interfaces.msg import Time as RosTime  # (선택) 미사용이면 제거해도 OK
from visualization_msgs.msg import Marker

from math import atan2
import textwrap, time, math
from copy import deepcopy
from geometry_msgs.msg import Quaternion

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

class ObjectPlannerExecutor(Node):
    def __init__(self):
        super().__init__('object_planner_executor')
        # --- params ---
        self.declare_parameter('model_name', 'rect_object')
        self.declare_parameter('model_sdf_path', '')
        self.declare_parameter('size_l', 1.0)
        self.declare_parameter('size_w', 0.6)
        self.declare_parameter('size_h', 0.2)
        self.declare_parameter('speed', 0.3)
        self.declare_parameter('dt', 0.03)
        self.declare_parameter('spawn_service', '/spawn_entity')
        # 월드 플러그인에 namespace를 준 경우 기본값을 /gazebo/set_entity_state 로 둠
        self.declare_parameter('set_state_service', '/gazebo/set_entity_state')

        p = self.get_parameter
        self.model_name = p('model_name').get_parameter_value().string_value
        self.model_sdf_path = p('model_sdf_path').get_parameter_value().string_value
        self.L = p('size_l').get_parameter_value().double_value
        self.W = p('size_w').get_parameter_value().double_value
        self.H = p('size_h').get_parameter_value().double_value
        self.speed = p('speed').get_parameter_value().double_value
        self.dt = p('dt').get_parameter_value().double_value
        self.spawn_srv_name = p('spawn_service').get_parameter_value().string_value
        self.set_state_srv_name = p('set_state_service').get_parameter_value().string_value  # => /gazebo/set_entity_state 또는 /set_entity_state

        # --- services & action ---
        self.spawn_cli = self.create_client(SpawnEntity, self.spawn_srv_name)
        self.set_state_cli = self.create_client(SetEntityState, self.set_state_srv_name)
        self.plan_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')

        # 현재 이름으로 서버가 없으면 자동 폴백(/gazebo/set_entity_state ↔ /set_entity_state)
        self._resolve_set_state_service()

        # --- topics ---
        self.init_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_initialpose, 10)
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        self.goal_sub_mb = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.on_goal, qos)
        self.goal_sub_nav2 = self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, qos)
        self.get_logger().info("Subscribed goal topics: /move_base_simple/goal and /goal_pose")
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.marker_pub = self.create_publisher(Marker, 'object_marker', 1)


        self.spawned = False
        self.last_pose = None


        # self.current_pose = None
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # self.tf_timer = self.create_timer(0.05, self._broadcast_tf)  # 20Hz


        self.get_logger().info('ObjectPlannerExecutor up. In RViz: 2D Pose Estimate (spawn+start) → 2D Nav Goal (plan+move).')


    def publish_marker(self, pose):
        m = Marker()
        m.header.frame_id = 'map'           # RViz Fixed Frame과 일치 (네 환경은 map)
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'rect_object'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose = pose                       # 위치/자세 그대로
        m.scale.x = float(self.L)
        m.scale.y = float(self.W)
        m.scale.z = float(self.H)
        # 갈색 (박스 SDF와 맞춤)
        m.color.r = 0.55; m.color.g = 0.27; m.color.b = 0.07; m.color.a = 1.0
        # 필요시 약간의 발광 효과 느낌은 emissive 대신 밝은 색/라이팅으로 대체
        m.lifetime.sec = 0                  # 0이면 계속 유지
        self.marker_pub.publish(m)

    # --- 서비스 이름 자동 확인/폴백 ---
    def _resolve_set_state_service(self):
        # 현재 설정으로 먼저 시도
        if self.set_state_cli.wait_for_service(timeout_sec=0.2):
            self.get_logger().info(f'Using set_state service: {self.set_state_srv_name}')
            return
        # 후보들 순차 확인
        for name in ('/gazebo/set_entity_state', '/set_entity_state'):
            if name == self.set_state_srv_name:
                continue
            candidate = self.create_client(SetEntityState, name)
            if candidate.wait_for_service(timeout_sec=0.2):
                self.get_logger().warn(f'{self.set_state_srv_name} not found. Fallback → {name}')
                self.set_state_srv_name = name
                self.set_state_cli = candidate
                return
        self.get_logger().warn('set_entity_state server not ready yet; will retry before teleport.')

    # ---- helper: make SDF box from L, W, H (갈색 머티리얼) ----
    def make_box_sdf(self, L=None, W=None, H=None) -> str:
        L = L if L is not None else self.L
        W = W if W is not None else self.W
        H = H if H is not None else self.H
        t = 0.05
        return textwrap.dedent(f"""\
            <?xml version="1.0"?>
            <sdf version="1.6">
            <model name="{self.model_name}">
            <static>false</static>
            <link name="link">
                <inertial><mass>5.0</mass></inertial>

                <!-- 본체: 갈색 -->
                <collision name="collision">
                    <geometry><box><size>{L} {W} {H}</size></box></geometry>
                </collision>
                <visual name="body_visual">
                    <geometry><box><size>{L} {W} {H}</size></box></geometry>
                    <material>
                        <ambient>0.30 0.15 0.05 1.0</ambient>
                        <diffuse>0.55 0.27 0.07 1.0</diffuse>
                        <specular>0.10 0.10 0.10 1.0</specular>
                        <emissive>0.05 0.02 0.00 1.0</emissive>
                    </material>
                </visual>
                <!-- 앞면(+X): 빨강 -->
                <visual name="front_panel">
                    <pose>{L/2:.6f} 0 0  0 0 0</pose>
                    <geometry><box><size>{t} {W} {H}</size></box></geometry>
                    <material>
                        <ambient>0.8 0.0 0.0 1.0</ambient>
                        <diffuse>1.0 0.0 0.0 1.0</diffuse>
                        <specular>0.3 0.2 0.2 1.0</specular>
                        <emissive>0.2 0.0 0.0 1.0</emissive>
                    </material>
                </visual>

                <!-- 뒷면(-X): 파랑 -->
                <visual name="rear_panel">
                    <pose>-{L/2:.6f} 0 0  0 0 0</pose>
                    <geometry><box><size>{t} {W} {H}</size></box></geometry>
                    <material>
                        <ambient>0.0 0.0 0.8 1.0</ambient>
                        <diffuse>0.0 0.0 1.0 1.0</diffuse>
                        <specular>0.2 0.2 0.3 1.0</specular>
                        <emissive>0.0 0.0 0.2 1.0</emissive>
                    </material>
                </visual>
            </link>
        </model>
        </sdf>
        """).strip()

    def wait_for(self, client, name, timeout=5.0):
        """서비스 발견을 위해 짧게 스핀 돌리며 기다림"""
        end = time.time() + timeout
        # 그래프 이벤트 처리를 위해 약간 스핀
        for _ in range(2):
            rclpy.spin_once(self, timeout_sec=0.1)
        while time.time() < end:
            if client.wait_for_service(timeout_sec=0.1):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(f"Timeout waiting for {name}")
        return False

    # ---- /initialpose ----
    def on_initialpose(self, msg):
        pose = msg.pose.pose

        if not self.wait_for(self.spawn_cli, self.spawn_srv_name, timeout=5.0):
            return

        req = SpawnEntity.Request()
        req.name = self.model_name
        req.xml = self.make_box_sdf() if not self.model_sdf_path else open(self.model_sdf_path).read()
        req.robot_namespace = self.get_namespace() or ""
        req.reference_frame = "world"
        req.initial_pose = pose

        fut = self.spawn_cli.call_async(req)
        def _done(_):
            res = fut.result()
            if res and getattr(res, 'success', True):
                self.spawned = True
                self.last_pose = pose
                self.get_logger().info(f"Spawned {req.name}.")
                # set_state 준비 확인
                if not self.set_state_cli.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn(f"{self.set_state_srv_name} not ready yet (will be needed for teleport/move)")
            else:
                self.get_logger().error(f"Spawn failed: {getattr(res, 'status_message', fut.exception())}")
        fut.add_done_callback(_done)

    # ---- /move_base_simple/goal → plan with SMAC (ComputePathToPose) ----
    def on_goal(self, goal_msg: PoseStamped):
        if self.last_pose is None:
            self.get_logger().warn('No initial pose yet. Click 2D Pose Estimate first.')
            return

        if not self.plan_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('planner_server action not available.')
            return

        goal = ComputePathToPose.Goal()
        goal.goal = goal_msg
        # start pose는 현재 위치(마지막 텔레포트 위치)로 생성
        start = PoseStamped()
        start.header.frame_id = 'map'
        start.header.stamp = self.get_clock().now().to_msg()
        start.pose = self.last_pose
        goal.start = start
        goal.use_start = True
        goal.planner_id = 'GridBased'     # plugin key

        send_future = self.plan_client.send_goal_async(goal)

        def _result_cb(fut):
            try:
                res = fut.result()
            except Exception as e:
                self.get_logger().error(f'Action error: {e}')
                return
            result = getattr(res, 'result', None)
            if result is None or result.path is None or len(result.path.poses) == 0:
                self.get_logger().warn('Planner returned empty path.')
                return
            path = result.path
            self.path_pub.publish(path)
            self.get_logger().info(f'Got path with {len(path.poses)} poses.')
            self.follow_path(path)

        def _goal_response_cb(fut):
            gh = fut.result()
            if gh is None or not getattr(gh, "accepted", True):
                self.get_logger().error('Failed to send goal to planner_server (rejected).')
                return
            self.get_logger().info('Goal accepted, waiting for result...')
            res_future = gh.get_result_async()
            res_future.add_done_callback(_result_cb)

        send_future.add_done_callback(_goal_response_cb)
        return  # 콜백 종료 후 executor가 나머지 이벤트 처리

    # ---- teleport helper ----
    def teleport(self, pose):
        # 서버 준비 재확인 + 필요시 폴백
        if not self.set_state_cli.wait_for_service(timeout_sec=1.0):
            self._resolve_set_state_service()
            if not self.set_state_cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error(f"{self.set_state_srv_name} not ready; skip teleport")
                return
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.model_name
        st.pose = pose
        st.reference_frame = 'world'
        req.state = st
        future = self.set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.publish_marker(pose)

        # 응답이 비어있을 수 있으니, 에러 없으면 성공으로 간주

    # ---- simple follower (경로 접선 방향으로 yaw 맞춤, blocking) ----
    def follow_path(self, path: Path):
        if len(path.poses) == 0:
            return

        N = len(path.poses)
        for i, ps in enumerate(path.poses):
            # 위치는 그대로, 방향(yaw)은 경로 접선으로 맞춤
            pose = deepcopy(ps.pose)

            # if N >= 2:
            #     if i < N - 1:
            #         dx = path.poses[i+1].pose.position.x - ps.pose.position.x
            #         dy = path.poses[i+1].pose.position.y - ps.pose.position.y
            #     else:
            #         # 마지막 점은 이전 점 기준
            #         dx = ps.pose.position.x - path.poses[i-1].pose.position.x
            #         dy = ps.pose.position.y - path.poses[i-1].pose.position.y
            #     yaw = math.atan2(dy, dx)

            #     # 마지막 점은 목표 orientation을 유지하고 싶다면 아래처럼:
            #     if i == N - 1:
            #         pose.orientation = ps.pose.orientation
            #     else:
            #         pose.orientation = yaw_to_quat(yaw)
            # else:
            #     # 점이 1개뿐이면 yaw=0
            #     pose.orientation = yaw_to_quat(0.0)

            # (선택) 바닥 튐 방지: pose.position.z = max(pose.position.z, 0.02)

            self.teleport(pose)
            self.last_pose = pose
            time.sleep(float(self.dt))  # rclpy.sleep() 없음 → time.sleep()

        self.get_logger().info('Arrived at goal.')

def main():
    rclpy.init()
    node = ObjectPlannerExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
