#!/usr/bin/env python3
import math, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.srv import GetPlan   # ✅ Humble은 이거!
from nav_msgs.msg import Path
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0))

def yaw_from_quat(z, w):
    return 2.0 * math.atan2(z, w)

def lerp(a, b, s): return a + s*(b - a)

class RvizPathExecutor(Node):
    def __init__(self):
        super().__init__('rviz_path_executor')
        self.model_name = self.declare_parameter('model_name', 'rect_object').get_parameter_value().string_value
        self.speed = self.declare_parameter('speed', 0.3).get_parameter_value().double_value
        self.dt = self.declare_parameter('dt', 0.03).get_parameter_value().double_value
        self.service_name = self.declare_parameter('set_state_service', '/gazebo/set_entity_state').get_parameter_value().string_value

        # RViz 입력
        self.sub_init = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_init, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)

        # ✅ Humble에서는 GetPlan 사용
        self.plan_cli = self.create_client(GetPlan, '/planner_server/get_plan')
        self.gz_cli   = self.create_client(SetEntityState, self.service_name)

        self.start_pose = None
        self.goal_pose = None
        self.get_logger().info("RVizPathExecutor ready. Set start & goal in RViz.")

    def on_init(self, msg):
        self.start_pose = PoseStamped()
        self.start_pose.header = msg.header
        self.start_pose.pose = msg.pose.pose
        self.get_logger().info("Start pose set.")

    def on_goal(self, msg):
        self.goal_pose = msg
        self.get_logger().info("Goal pose set.")
        self.plan_and_execute()

    def plan_and_execute(self):
        if self.start_pose is None or self.goal_pose is None:
            self.get_logger().warn("Need start & goal.")
            return
        req = GetPlan.Request()
        req.start = self.start_pose
        req.goal = self.goal_pose

        future = self.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None or len(resp.plan.poses) < 2:
            self.get_logger().error("Planning failed.")
            return

        self.follow(resp.plan)

    def follow(self, path: Path):
        poses = path.poses
        for i in range(len(poses)-1):
            p0, p1 = poses[i].pose, poses[i+1].pose
            dx, dy = p1.position.x - p0.position.x, p1.position.y - p0.position.y
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6: continue
            steps = max(2, int(seg_len / (self.speed*self.dt)))
            yaw = math.atan2(dy, dx)
            qx,qy,qz,qw = quat_from_yaw(yaw)
            for s in range(steps):
                s_ratio = float(s+1)/steps
                x = lerp(p0.position.x, p1.position.x, s_ratio)
                y = lerp(p0.position.y, p1.position.y, s_ratio)
                self.set_entity(x, y, qx,qy,qz,qw)
                time.sleep(self.dt)
        self.get_logger().info("Path playback done.")

    def set_entity(self, x,y,qx,qy,qz,qw):
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.model_name
        st.pose.position.x = x
        st.pose.position.y = y
        st.pose.position.z = 0.0
        st.pose.orientation.x = qx
        st.pose.orientation.y = qy
        st.pose.orientation.z = qz
        st.pose.orientation.w = qw
        req.state = st
        f = self.gz_cli.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=1.0)

def main():
    rclpy.init()
    node = RvizPathExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
