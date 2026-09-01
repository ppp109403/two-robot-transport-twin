"""관제 PC -> 로봇 2대 도메인 브릿지.

넘기는 토픽과 도메인 배치는 config/domain_bridge.yaml 에 있다.
  10 : 관제 PC (이 PC)   11 : 로봇 A   12 : 로봇 B

사용::

    ros2 launch mobile_manipulator_trajectory domain_bridge.launch.py

    # 다른 설정 파일로
    ros2 launch mobile_manipulator_trajectory domain_bridge.launch.py \\
        config:=/path/to/other.yaml

선행 조건::

    sudo apt install ros-jazzy-domain-bridge

★ 이 프로세스는 ROS_DOMAIN_ID 환경변수를 쓰지 않는다. yaml 의 from_domain /
  to_domain 만 본다. 그래서 어느 터미널에서 띄우든 동작이 같다.

★ 로봇 PC 는 ROS_DOMAIN_ID 만 설정하면 된다 (11 또는 12).
  브릿지 설치도, 추가 설정도 필요 없다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('mobile_manipulator_trajectory'),
        'config', 'domain_bridge.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        Node(
            package='domain_bridge',
            executable='domain_bridge',
            name='mobile_manipulator_domain_bridge',
            output='screen',
            arguments=[LaunchConfiguration('config')],
        ),
    ])
