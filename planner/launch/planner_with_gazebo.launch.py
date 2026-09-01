from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import os
from nav2_common.launch import RewrittenYaml
from launch.substitutions import TextSubstitution  # ← 추가


def make_nodes(context):
    # ---- args ----
    world    = LaunchConfiguration('world').perform(context)
    map_yaml = LaunchConfiguration('map_yaml').perform(context)
    size_l   = float(LaunchConfiguration('size_l').perform(context))
    size_w   = float(LaunchConfiguration('size_w').perform(context))
    size_h   = float(LaunchConfiguration('size_h').perform(context))
    ds = float(LaunchConfiguration('ds').perform(context)) 
    # ---- footprint (문자열) ----
    half_l, half_w = size_l/2.0, size_w/2.0
    footprint_str = (
        f"[[{-half_l:.3f}, {-half_w:.3f}], "
        f"[{ half_l:.3f}, {-half_w:.3f}], "
        f"[{ half_l:.3f}, { half_w:.3f}], "
        f"[{-half_l:.3f}, { half_w:.3f}]]"
    )

    # ---- Gazebo ----
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world}.items()
    )

    # ---- Map Server ----
    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': True}],
        arguments=['--ros-args', '--log-level', 'map_server:=debug']
    )

    # ---- Global Costmap 파라미터 (planner_server 내부 코스트맵에 적용) ----
    costmap_base = {
        'global_costmap': {
            'global_costmap': {            # ★ 내부 노드명으로 한 번 더 감싸기
            'ros__parameters': {
                'use_sim_time': True,
                'global_frame': 'map',
                'robot_base_frame': 'base_link',
                'update_frequency': 5.0,
                'publish_frequency': 1.0,
                'resolution': 0.05,
                'track_unknown_space': True,
                'plugins': ['static_layer', 'inflation_layer'],
                'static_layer': {
                'plugin': 'nav2_costmap_2d::StaticLayer',
                'map_subscribe_transient_local': True
                },
                'inflation_layer': {
                'plugin': 'nav2_costmap_2d::InflationLayer',
                'inflation_radius': 0.25
                },
                'footprint': footprint_str
            }
            }
        }
    }

    # ---- Planner Server (Smac Lattice) ----
    # planner_params = {
    #     'use_sim_time': True,
    #     'planner_plugins': ['GridBased'],
    #     'GridBased': {
    #         'plugin': 'nav2_smac_planner/SmacPlannerLattice',
    #         'allow_unknown': True,
    #         'tolerance': 0.25,
    #         'max_planning_time': 4.01515,
    #         'cache_obstacle_heuristic': True,
    #         'allow_reverse_expansion': True,
    #         'smooth_path': True,
    #         'viz_expansions': True
    #     }
    # }

    params_file = os.path.join(
        get_package_share_directory('mobile_manipulator_trajectory'),
        'params', 'nav2_params.yaml'
    )


    planner_yaml = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites={
            # ← 값은 "문자열" 또는 TextSubstitution(text='...') 여야 함
            'global_costmap.global_costmap.ros__parameters.footprint': TextSubstitution(text=footprint_str),
            'planner_server.ros__parameters.use_sim_time': TextSubstitution(text='true'),
            'global_costmap.global_costmap.ros__parameters.use_sim_time': TextSubstitution(text='true'),
        },
        convert_types=True,  # 문자열을 실제 bool/list/float로 변환
    )

    # ★★ 순서 중요: costmap_base 정의 후 planner_server 생성
    planner_server = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen',
        parameters=[planner_yaml]
    )

    # ---- Lifecycle (map + planner 관리) ----
    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_all', output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server', 'planner_server']  # global_costmap 제거
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
        package='mobile_manipulator_trajectory', executable='object_planner_executor',
        name='object_planner_executor', output='screen',
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

    return [gazebo, map_server, planner_server,
            lifecycle_mgr, executor,
            static_tf_world_map, static_tf_map_base, rviz]


def generate_launch_description():
    use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')
    pkg_path = os.path.join(get_package_share_directory('mobile_manipulator_trajectory'))
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_path, 'config', 'rviz_nav.rviz')
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(
                get_package_share_directory('mobile_manipulator'),
                'worlds', 'munji_3f_wide.world'
            )
        ),
        DeclareLaunchArgument(
            'map_yaml',
            default_value=os.path.join(
                get_package_share_directory('amr'),
                'map', 'munji_3f_2025_wide', 'munji_3f_2025_wide.yaml'
            )
        ),
        DeclareLaunchArgument('size_l', default_value='1.2'),
        DeclareLaunchArgument('size_w', default_value='0.8'),
        DeclareLaunchArgument('size_h', default_value='0.25'),
        DeclareLaunchArgument('ds', default_value='0.01'),  # 샘플 간격[m]

        use_rviz_arg,
        rviz_config_arg,
        OpaqueFunction(function=make_nodes),
    ])
