from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import os
from nav2_common.launch import RewrittenYaml
from launch.substitutions import TextSubstitution  # 그대로 사용

def make_nodes(context):
    # ---- args ----
    world    = LaunchConfiguration('world').perform(context)      # 선언만 유지(미사용)
    map_yaml = LaunchConfiguration('map_yaml').perform(context)
    size_l   = float(LaunchConfiguration('size_l').perform(context))
    size_w   = float(LaunchConfiguration('size_w').perform(context))
    size_h   = float(LaunchConfiguration('size_h').perform(context))
    ds = float(LaunchConfiguration('ds').perform(context)) 
    amr_size_w = 0.0 # amr 가로 길이
    amr_size_l = 0.0 


    # ---- footprint (문자열) ----
    half_l, half_w = size_l/2.0, size_w/2.0
    half_width_turn = amr_size_l                  # full length = 2*amr_size_l
    half_length_turn  = half_l + amr_size_w         # 요청대로

    footprint_str = (
        f"[[{-half_l:.3f}, {-half_w:.3f}], "
        f"[{ half_l:.3f}, {-half_w:.3f}], "
        f"[{ half_l:.3f}, { half_w:.3f}], "
        f"[{-half_l:.3f}, { half_w:.3f}]]"
    )

    turning_footprint = (
        f"[[{-half_length_turn:.3f}, {-half_width_turn:.3f}], "
        f"[{ half_length_turn:.3f}, {-half_width_turn:.3f}], "
        f"[{ half_length_turn:.3f}, { half_width_turn:.3f}], "
        f"[{-half_length_turn:.3f}, { half_width_turn:.3f}]]"
    )

    # ---- Map Server ----
    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': True}],
        arguments=['--ros-args', '--log-level', 'map_server:=debug']
    )

    # ---- Planner Server (Smac Lattice) ----
    params_file = os.path.join(
        get_package_share_directory('mobile_manipulator_trajectory'),
        'params', 'nav2_params.yaml'
    )

    lattice_path = os.path.join(
        get_package_share_directory('nav2_smac_planner'),
        'sample_primitives', '5cm_resolution',
        '0.5m_turning_radius', 'ackermann', 'output.json'
    )
    
    planner_yaml = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites={
            'planner_server.ros__parameters.GridBased.lattice_filepath': TextSubstitution(text=lattice_path),
            'global_costmap.global_costmap.ros__parameters.footprint': TextSubstitution(text=footprint_str),
            'planner_server.ros__parameters.GridBased.turning_footprint': TextSubstitution(text=turning_footprint),
            'planner_server.ros__parameters.use_sim_time': TextSubstitution(text='false'),
            'global_costmap.global_costmap.ros__parameters.use_sim_time': TextSubstitution(text='false'),
        },
        convert_types=True,
    )

    # 순서 중요: costmap_base 정의 후 planner_server 생성
    planner_server = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen',
        parameters=[planner_yaml]  # 필요시 costmap_base도 함께 적용: [planner_yaml, costmap_base]
    )

    # ---- Lifecycle (map + planner 관리) ----
    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_all', output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server', 'planner_server']
        }]
    )

    # ---- TF ----
    static_tf_world_map = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='static_tf_world_map',
        arguments=['0','0','0','0','0','0','world','map']
    )
    static_tf_map_base = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='static_tf_map_base',
        arguments=['0','0','0','0','0','0','map','base_link']
    )

    # ---- Object Executor ----
    executor = Node(
        package='mobile_manipulator_trajectory', executable='object_planner_executor_no_gazebo',
        name='object_planner_executor_no_gazebo', output='screen',
        parameters=[{
            'model_name': 'rect_object',
            'model_sdf_path': '',
            'size_l': size_l, 'size_w': size_w, 'size_h': size_h, 'ds': ds,
            'speed': 0.3, 'dt': 0.03,
            'set_state_service': '/set_entity_state',
            'spawn_service': '/spawn_entity'
        }]
    )

    # ---- RViz ----
    rviz = Node(
        package='rviz2', executable='rviz2',
        name='rviz2', output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    # Gazebo 제외: gazebo 노드는 반환 리스트에서 제거
    return [
        map_server, planner_server, lifecycle_mgr,
        executor, static_tf_world_map, static_tf_map_base, rviz
    ]

def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')
    pkg_path = os.path.join(get_package_share_directory('mobile_manipulator_trajectory'))
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_path, 'config', 'rviz_nav.rviz')
    )

    return LaunchDescription([
        # world 인자는 유지(미사용)
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(
                get_package_share_directory('mobile_manipulator'),
                'worlds', 'munji_room.world'
            )
        ),
        DeclareLaunchArgument(
            'map_yaml',
            default_value=os.path.join(
                get_package_share_directory('amr'),
                'map', 'munji_3f_2025_wide', 'munji_3f_2025_wide.yaml'
            )
        ),
        DeclareLaunchArgument('size_l', default_value='1.6'),
        DeclareLaunchArgument('size_w', default_value='0.8'),
        DeclareLaunchArgument('size_h', default_value='0.25'),
        DeclareLaunchArgument('ds', default_value='0.01'),  # 샘플 간격[m]
        use_rviz_arg,
        rviz_config_arg,
        OpaqueFunction(function=make_nodes),
    ])