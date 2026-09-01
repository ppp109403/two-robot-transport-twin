
# 1. Gazebo visual --> manipulator trajectory (smac lattice)

<img width="1776" height="986" alt="Screenshot from 2025-08-25 01-48-35" src="https://github.com/user-attachments/assets/910869ea-4c25-4520-866d-9a7a10088f1b" />

### dependancy  
mobile_manipulator  → munji world (gazebo)
amr → munji map

### command
ros2 launch mobile_manipulator_trajectory planner_with_gazebo.launch.py   world:=/home/amr2/ros2_ws/src/mobile_manipulator/worlds/munji_3f_wide.world   map_yaml:=/home/amr2/ros2_ws/src/amr/map/munji_3f_2025_wide/munji_3f_2025_wide.yaml   size_l:=2.0 size_w:=0.6 size_h:=0.25

* size_l, size_w, size_h  -->  size

### 다른 PC
ros2 launch mobile_manipulator_trajectory planner_with_gazebo.launch.py          world:=<your_workspace>/src/mobile_manipulator/worlds/munji_3f_wide.world    
map_yaml:=<your_workspace>/src/amr/map/munji_3f_2025_wide/munji_3f_2025_wide.yaml   size_l:=1.0 size_w:=0.6 size_h:=0.25

# 관제용
## 2. rviz2 --> manipulator trajectory ( trajecory EE of two amr )
ros2 launch mobile_manipulator_trajectory planner_only.launch.py  size_l:=0.8 size_w:=0.6 ds:=0.01 

0. 물체 크기 설정 : size_l : 물체가 잡는 양 끝단 사이 거리 / size_w : 다른 변 길이 / ds : 경로 분할 간격
1. 물체 양 끝단 점 토픽 sub ( 기존에는 로봇팔 ee점을 받으려 함 : /a/ee_pose , /b/ee_pose
   실험용 토픽
   

ros2 topic pub /a/ee_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 0.2, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

ros2 topic pub /b/ee_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 1.3, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

--> 거리가 1.0이 됨 ( size_l 과 일치하게 )

2. 초기화 완료. goal을 지정해 중심 경로를 생성하세요. --> 커맨드 창에 뜨면 실행 가능
3. goal pose 입력시 경로 생성

* Marker를 켜면 박스가 나와서 예상 경로를 볼 수 있음
* planned_path_poses : 물체 중심 pose
* mobile_manipulator_trajectory/object_planner_executor_no_gazebo.py line 795 - center_path = self._resample_center_path_se2(center_path, ds=0.05, w_theta=0.5*self.L)
* ds, w_theta 설정시 회전 속도 고려 --> 값이 커질수록 촘촘하게 자름
* ds : 선운동, w_theta: 회전운동
##


-----
2번 ee 가져와서 업데이트 --> 아직 오류있음

``` ros2 launch mobile_manipulator_trajectory planner_only_ee_sub.launch.py   size_l:=0.8 size_w:=0.6 ds:=0.01  ```

관제용 / 관제 --> EE_pos,orientation --> amr1,2,...
