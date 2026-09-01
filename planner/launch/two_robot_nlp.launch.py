import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def make_nodes(context):
    pkg = get_package_share_directory('mobile_manipulator_trajectory')
    lc = LaunchConfiguration

    map_yaml = lc('map_yaml').perform(context)

    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': False}],
    )

    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_map', output='screen',
        parameters=[{'use_sim_time': False, 'autostart': True,
                     'node_names': ['map_server']}],
    )

    planner = Node(
        package='mobile_manipulator_trajectory', executable='two_robot_nlp_planner',
        name='two_robot_nlp_planner', output='screen',
        parameters=[{
            'map_frame': 'map',
            'csv_dir': lc('csv_dir').perform(context),
            'ns_a': lc('ns_a').perform(context),
            'ns_b': lc('ns_b').perform(context),
            'stream_rate': float(lc('stream_rate').perform(context)),
            'auto_start': lc('auto_start').perform(context).lower() == 'true',

            # ---- AMR footprint (rectangle, covered by n_base_disks circles)
            'base_len': float(lc('base_len').perform(context)),
            'base_wid': float(lc('base_wid').perform(context)),
            'n_base_disks': int(lc('n_base_disks').perform(context)),
            'base_margin': float(lc('base_margin').perform(context)),

            # ---- carried object (rectangle; obj_len is also the grasp spacing)
            'obj_len': float(lc('obj_len').perform(context)),
            'obj_wid': float(lc('obj_wid').perform(context)),
            'n_obj_disks': int(lc('n_obj_disks').perform(context)),
            'obj_margin': float(lc('obj_margin').perform(context)),

            # ---- the only thing standing in for the arm
            'reach_min': float(lc('reach_min').perform(context)),
            'reach_max': float(lc('reach_max').perform(context)),
            'ee_forward_min': float(lc('ee_forward_min').perform(context)),
            'map_inflation': float(lc('map_inflation').perform(context)),

            # ---- limits (hard ceilings enforced as NLP constraints)
            'v_max': float(lc('v_max').perform(context)),
            'v_min': float(lc('v_min').perform(context)),
            'w_max': float(lc('w_max').perform(context)),
            'a_max': float(lc('a_max').perform(context)),
            'alpha_max': float(lc('alpha_max').perform(context)),
            'obj_v_max': float(lc('obj_v_max').perform(context)),
            'obj_w_max': float(lc('obj_w_max').perform(context)),
            'ee_v_max': float(lc('ee_v_max').perform(context)),
            'robot_robot_min': float(lc('robot_robot_min').perform(context)),

            # ---- horizon / seed
            'dt': float(lc('dt').perform(context)),
            'v_nom': float(lc('v_nom').perform(context)),
            'n_settle': 12,
            'max_steps': int(lc('max_steps').perform(context)),
            'astar_clearance': float(lc('astar_clearance').perform(context)),
            'astar_prefer_clearance': 0.90,
            'astar_clearance_weight': 0.60,
            'sdf_crop_margin': 3.0,
            'sdf_downsample': int(lc('sdf_downsample').perform(context)),
            'sdf_max_cells': int(lc('sdf_max_cells').perform(context)),
            'horizon_slack': float(lc('horizon_slack').perform(context)),
            'sdf_smooth_cells': 1.0,
            'unknown_is_occupied': True,
            'object_symmetric': True,
            'pin_base_start': False,

            # ---- solver
            'w_base_ref': float(lc('w_base_ref').perform(context)),
            'w_base_yaw_ref': float(lc('w_base_yaw_ref').perform(context)),
            'w_base_vel': float(lc('w_base_vel').perform(context)),
            'collision_substeps': int(lc('collision_substeps').perform(context)),
            'max_iter': int(lc('max_iter').perform(context)),
            'ipopt_print_level': int(lc('ipopt_print_level').perform(context)),
            'hessian': 'exact',
            'sdf_interp': lc('sdf_interp').perform(context),
        }],
    )

    static_tf_map_base = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='static_tf_map_base',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', os.path.join(pkg, 'config', 'two_robot_nlp.rviz')],
        condition=IfCondition(lc('use_rviz')),
    )

    return [map_server, lifecycle_mgr, planner, static_tf_map_base, rviz]


def generate_launch_description():
    # 맵은 afl_nav 패키지가 share 로 설치한다 ('amr' 에서 이름이 바뀐 것으로 보인다).
    # DeclareLaunchArgument 의 default_value 는 커맨드라인으로 덮어써도 **먼저 평가**되므로,
    # 없는 패키지를 참조하면 map_yaml:= 를 넘겨도 launch 가 그 전에 죽는다.
    default_map = os.path.join(
        get_package_share_directory('afl_nav'),
        'map', 'munji_3f_2025_wide', 'munji_3f_2025_wide.yaml')

    args = [
        DeclareLaunchArgument('map_yaml', default_value=default_map),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('csv_dir', default_value=''),
        DeclareLaunchArgument('ns_a', default_value='a'),
        DeclareLaunchArgument('ns_b', default_value='b'),
        DeclareLaunchArgument('stream_rate', default_value='100.0'),
        DeclareLaunchArgument('auto_start', default_value='false'),

        DeclareLaunchArgument('base_len', default_value='0.80'),
        DeclareLaunchArgument('base_wid', default_value='0.55'),
        DeclareLaunchArgument('n_base_disks', default_value='3'),
        DeclareLaunchArgument('base_margin', default_value='0.05'),

        DeclareLaunchArgument('obj_len', default_value='2.00'),
        DeclareLaunchArgument('obj_wid', default_value='0.30'),
        DeclareLaunchArgument('n_obj_disks', default_value='5'),
        DeclareLaunchArgument('obj_margin', default_value='0.05'),

        DeclareLaunchArgument('reach_min', default_value='0.25'),
        DeclareLaunchArgument('reach_max', default_value='0.75'),
        DeclareLaunchArgument('ee_forward_min', default_value='0.10'),
        DeclareLaunchArgument('map_inflation', default_value='0.05'),
        DeclareLaunchArgument('robot_robot_min', default_value='0.90'),

        # ---- speed / acceleration ceilings ----
        # v_min is the *reverse* limit and matters as much as v_max: in a
        # single-file carry the leading robot faces the bar, so it drives
        # backwards for the whole trajectory.
        DeclareLaunchArgument('v_max', default_value='0.60'),
        DeclareLaunchArgument('v_min', default_value='-0.50'),
        DeclareLaunchArgument('w_max', default_value='1.00'),
        DeclareLaunchArgument('a_max', default_value='0.80'),
        DeclareLaunchArgument('alpha_max', default_value='2.00'),
        DeclareLaunchArgument('obj_v_max', default_value='0.60'),
        DeclareLaunchArgument('obj_w_max', default_value='0.60'),
        # what the RL tracker actually has to follow -- object rotation adds
        # (obj_len/2)*obj_w on top of the object centre speed
        DeclareLaunchArgument('ee_v_max', default_value='0.70'),

        DeclareLaunchArgument('dt', default_value='0.15'),
        DeclareLaunchArgument('v_nom', default_value='0.45'),
        DeclareLaunchArgument('max_steps', default_value='700'),
        DeclareLaunchArgument('astar_clearance', default_value='0.35'),
        DeclareLaunchArgument('sdf_downsample', default_value='1'),
        DeclareLaunchArgument('sdf_max_cells', default_value='400000'),
        DeclareLaunchArgument('horizon_slack', default_value='1.25'),

        DeclareLaunchArgument('w_base_ref', default_value='3.0'),
        DeclareLaunchArgument('w_base_yaw_ref', default_value='0.0'),
        DeclareLaunchArgument('w_base_vel', default_value='8.0'),
        DeclareLaunchArgument('collision_substeps', default_value='-1'),
        DeclareLaunchArgument('max_iter', default_value='3000'),
        DeclareLaunchArgument('ipopt_print_level', default_value='3'),
        DeclareLaunchArgument('sdf_interp', default_value='bspline'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=make_nodes)])
