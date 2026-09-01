from setuptools import setup
import os
from glob import glob

package_name = 'mobile_manipulator_trajectory'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),  # ✅ 이 줄 추가
        # *.yaml 도 함께 설치한다 (domain_bridge.yaml). 빠뜨리면
        # domain_bridge.launch.py 가 share 에서 설정 파일을 못 찾는다.
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz') + glob('config/*.yaml')),
        ('share/' + package_name + '/params',  glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Trajectory execution for mobile manipulator object in Gazebo',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
        'rviz_path_executor = mobile_manipulator_trajectory.rviz_path_executor:main',
        'object_planner_executor = mobile_manipulator_trajectory.object_planner_executor:main',
        'object_planner_executor_no_gazebo = mobile_manipulator_trajectory.object_planner_executor_no_gazebo:main',
        'object_planner_executor_no_gazebo_ee_sub = mobile_manipulator_trajectory.object_planner_executor_no_gazebo_ee_sub:main',
        'two_robot_nlp_planner = mobile_manipulator_trajectory.two_robot_nlp_node:main',
        ],
    },
)