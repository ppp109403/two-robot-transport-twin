"""Two-robot cooperative transport planner on a static /map.

    /map           OccupancyGrid   static obstacles
    /initialpose   PoseWithCov     object start pose (RViz "2D Pose Estimate")
    /goal_pose     PoseStamped     object goal pose  (RViz "2D Goal Pose") -> solves
    /plan_execute  Bool            true starts streaming setpoints

Out, per robot (namespaces from ns_a / ns_b, default "a" and "b"):

    /a/ee_path     Path            whole EE trajectory, timing in per-pose stamps
    /a/base_path   Path            whole base trajectory
    /a/ee_pose     PoseStamped     live EE setpoint, interpolated to stream_rate
    /a/base_pose   PoseStamped     live base setpoint
    /a/base_twist  TwistStamped    live base v, w

Both EE *and* base references are published on purpose.  The EE paths are what
the RL policy tracks; the base paths are what makes the collision guarantee
mean anything downstream, because a policy that only sees the EE target is free
to put the base wherever it likes.

Playback is open loop: setpoints advance on wall-clock time and nothing here
watches where the robots actually are.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import (Point, PoseStamped, PoseWithCovarianceStamped,
                               Quaternion, TwistStamped)
from interactive_markers import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    InteractiveMarkerFeedback, Marker, MarkerArray)

from mobile_manipulator_trajectory import grid_astar as ga
from mobile_manipulator_trajectory.sdf_utils import SignedDistanceField
from mobile_manipulator_trajectory.two_robot_nlp import (
    Geometry, Limits, SolverOpts, Weights, pose_clearance, solve_transport, verify)

LATCHED = QoSProfile(depth=1,
                     history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quat_to_yaw(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TwoRobotNlpPlanner(Node):
    def __init__(self):
        super().__init__('two_robot_nlp_planner')

        d = self.declare_parameter
        # --- frames / io
        d('map_frame', 'map')
        d('csv_dir', '')
        # Namespaces for the two robots. Set these to the real robot
        # namespaces so each side subscribes only to its own topics.
        d('ns_a', 'a')
        d('ns_b', 'b')
        # --- execution streaming
        d('stream_rate', 100.0)     # setpoint rate for the RL tracker
        d('auto_start', False)
        # --- geometry
        d('base_len', 0.80); d('base_wid', 0.55)
        d('n_base_disks', 3); d('base_margin', 0.05)
        d('obj_len', 2.00); d('obj_wid', 0.30)
        d('n_obj_disks', 5); d('obj_margin', 0.05)
        d('reach_min', 0.25); d('reach_max', 0.75)
        d('ee_forward_min', 0.10)
        # 2D 라이다 맵이 벽을 실제보다 두껍게 그리는 몫 [m]. 요구 여유에서 뺀다.
        # 맵을 침식하지 않는 이유는 two_robot_nlp.Geometry.map_inflation 주석 참고.
        d('map_inflation', 0.05)
        # --- limits
        d('v_max', 0.60); d('v_min', -0.50); d('w_max', 1.00)
        d('a_max', 0.80); d('alpha_max', 2.00)
        d('obj_v_max', 0.60); d('obj_w_max', 0.60); d('ee_v_max', 0.70)
        d('robot_robot_min', 0.90)
        # --- weights
        d('w_ref', 20.0); d('w_yaw_ref', 5.0)
        d('w_goal_pos', 2000.0); d('w_goal_yaw', 60.0)
        d('w_u', 0.05); d('w_du', 0.50)
        d('w_obj_vel', 1.0); d('w_obj_acc', 20.0)
        d('w_obj_yaw_rate', 20.0); d('w_obj_yaw_acc', 200.0)
        d('w_term_vel', 50.0); d('w_collision', 1.0e6)
        # 편대 유지 — 차체가 자기 파지점 뒤에 일렬로 서 있게 한다.
        d('w_base_ref', 3.0); d('w_base_yaw_ref', 0.0); d('w_base_vel', 8.0)
        # --- horizon / seed
        d('dt', 0.15)
        d('v_nom', 0.45)
        d('horizon_slack', 1.25)
        d('n_settle', 12)
        d('max_steps', 700)
        d('astar_clearance', 0.35)
        d('astar_prefer_clearance', 0.90)
        d('astar_clearance_weight', 0.60)
        d('sdf_crop_margin', 3.0)
        d('sdf_downsample', 1)
        d('sdf_max_cells', 400_000)
        d('sdf_smooth_cells', 1.0)
        d('unknown_is_occupied', True)
        d('object_symmetric', True)
        d('pin_base_start', False)
        # Safety interlock.  false = refuse to stream a plan that failed
        # verify() (non-converged, colliding, or discontinuous).
        d('allow_unsafe', False)
        # --- visualisation
        d('animate', True)
        d('animate_rate', 30.0)
        d('animate_speed', 1.0)
        d('ghost_count', 14)
        # --- solver
        # -1 = 한계값에서 유도 (two_robot_nlp.required_substeps).
        #      0 = knot 만, n = knot 사이 n 개.  자세한 것은 그 함수 주석.
        d('collision_substeps', -1)
        d('max_iter', 3000); d('ipopt_print_level', 3)
        d('hessian', 'exact'); d('sdf_interp', 'bspline')

        self.map_frame = self.get_parameter('map_frame').value

        self._map_msg = None
        self._sdf_full = None
        self._start_obj = None
        self._busy = False
        self._lock = threading.Lock()

        self.create_subscription(OccupancyGrid, '/map', self.on_map, LATCHED)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 self.on_initialpose, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)

        na = self.get_parameter('ns_a').value
        nb = self.get_parameter('ns_b').value
        self.pub_obj = self.create_publisher(Path, '/object_path', LATCHED)
        self.pub_b1 = self.create_publisher(Path, f'/{na}/base_path', LATCHED)
        self.pub_b2 = self.create_publisher(Path, f'/{nb}/base_path', LATCHED)
        self.pub_e1 = self.create_publisher(Path, f'/{na}/ee_path', LATCHED)
        self.pub_e2 = self.create_publisher(Path, f'/{nb}/ee_path', LATCHED)
        self.pub_ref = self.create_publisher(Path, '/seed_path', LATCHED)

        # Live setpoints: what the RL policy and the base controller actually
        # consume.  The plan is sampled at dt (~6.7 Hz); these are interpolated
        # up to stream_rate so the tracker never sees a staircase.
        self.pub_ee_t = {
            'a': self.create_publisher(PoseStamped, f'/{na}/ee_pose', 10),
            'b': self.create_publisher(PoseStamped, f'/{nb}/ee_pose', 10)}
        self.pub_base_t = {
            'a': self.create_publisher(PoseStamped, f'/{na}/base_pose', 10),
            'b': self.create_publisher(PoseStamped, f'/{nb}/base_pose', 10)}
        self.pub_base_tw = {
            'a': self.create_publisher(TwistStamped, f'/{na}/base_twist', 10),
            'b': self.create_publisher(TwistStamped, f'/{nb}/base_twist', 10)}
        self.pub_done = self.create_publisher(Bool, '/plan_execution_done', LATCHED)
        # Isaac Sim twin handshake.  Latched so a simulator that starts late still
        # sees the last SETUP.  Nothing else in this node reads it.
        self.pub_setup = self.create_publisher(Bool, '/twin/setup', LATCHED)
        self._setup_count = 0
        self.create_subscription(Bool, '/plan_execute', self.on_execute, 10)
        self.pub_mk = self.create_publisher(MarkerArray, '/two_robot_plan/markers', LATCHED)
        self.pub_anim = self.create_publisher(MarkerArray, '/two_robot_plan/animation', 1)
        self.pub_sdf = self.create_publisher(OccupancyGrid, '/sdf_viz', LATCHED)

        self._anim_plan = None
        self._plan_ok = False
        self._anim_k = 0.0
        rate = float(self.get_parameter('animate_rate').value)
        if bool(self.get_parameter('animate').value) and rate > 0.0:
            self.create_timer(1.0 / rate, self._tick_animation)

        self._exec_t0 = None
        srate = float(self.get_parameter('stream_rate').value)
        if srate > 0.0:
            self.create_timer(1.0 / srate, self._tick_stream)

        # Clickable button in RViz.  An interactive marker is used rather than a
        # custom panel plugin so this needs no C++ and no RViz rebuild -- the
        # stock InteractiveMarkers display picks it up.
        self._im_pos = np.array([0.0, 0.0])
        self._last_click = None
        self._im_server = InteractiveMarkerServer(self, 'plan_control')
        self._menu = MenuHandler()
        self._menu.insert('SETUP sim (spawn robots)', callback=self._on_menu_setup)
        self._menu.insert('Execute from start', callback=self._on_menu_execute)
        self._menu.insert('Stop', callback=self._on_menu_stop)
        self._build_button()

        self.get_logger().info(
            'ready. set the object start with RViz "2D Pose Estimate", '
            'then the goal with "2D Goal Pose".')

    # ------------------------------------------------------------- config
    def _geometry(self) -> Geometry:
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        return Geometry(base_len=p('base_len'), base_wid=p('base_wid'),
                        n_base_disks=int(p('n_base_disks')), base_margin=p('base_margin'),
                        obj_len=p('obj_len'), obj_wid=p('obj_wid'),
                        n_obj_disks=int(p('n_obj_disks')), obj_margin=p('obj_margin'),
                        reach_min=p('reach_min'), reach_max=p('reach_max'),
                        ee_forward_min=p('ee_forward_min'),
                        map_inflation=p('map_inflation'))

    def _limits(self) -> Limits:
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        return Limits(v_max=p('v_max'), v_min=p('v_min'), w_max=p('w_max'),
                      a_max=p('a_max'), alpha_max=p('alpha_max'),
                      obj_v_max=p('obj_v_max'), obj_w_max=p('obj_w_max'),
                      ee_v_max=p('ee_v_max'),
                      robot_robot_min=p('robot_robot_min'))

    def _weights(self) -> Weights:
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        return Weights(ref=p('w_ref'), yaw_ref=p('w_yaw_ref'),
                       goal_pos=p('w_goal_pos'), goal_yaw=p('w_goal_yaw'),
                       u=p('w_u'), du=p('w_du'), obj_vel=p('w_obj_vel'),
                       obj_acc=p('w_obj_acc'), obj_yaw_rate=p('w_obj_yaw_rate'),
                       obj_yaw_acc=p('w_obj_yaw_acc'), term_vel=p('w_term_vel'),
                       base_ref=p('w_base_ref'), base_yaw_ref=p('w_base_yaw_ref'),
                       base_vel=p('w_base_vel'), collision=p('w_collision'))

    def _solver_opts(self) -> SolverOpts:
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        return SolverOpts(max_iter=int(p('max_iter')),
                          print_level=int(p('ipopt_print_level')),
                          hessian=p('hessian'), sdf_interp=p('sdf_interp'),
                          collision_substeps=int(p('collision_substeps')))

    # ------------------------------------------------------------ callbacks
    def on_map(self, msg: OccupancyGrid):
        self._map_msg = msg
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        t0 = time.time()
        self._sdf_full = SignedDistanceField.from_occupancy(
            data,
            origin=(info.origin.position.x, info.origin.position.y),
            res=info.resolution,
            unknown_is_occupied=bool(self.get_parameter('unknown_is_occupied').value),
            smooth_sigma_cells=float(self.get_parameter('sdf_smooth_cells').value))
        self.get_logger().info(
            f'/map {info.width}x{info.height} @ {info.resolution:.3f} m -> SDF in '
            f'{time.time() - t0:.2f}s  (range '
            f'{self._sdf_full.sdf.min():.2f}..{self._sdf_full.sdf.max():.2f} m)')
        self._publish_sdf_viz()

    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        self._start_obj = np.array([msg.pose.pose.position.x,
                                    msg.pose.pose.position.y,
                                    quat_to_yaw(msg.pose.pose.orientation)])
        self.get_logger().info(
            f'object start = ({self._start_obj[0]:.2f}, {self._start_obj[1]:.2f}, '
            f'{math.degrees(self._start_obj[2]):.1f} deg)')

    def on_goal(self, msg: PoseStamped):
        if self._sdf_full is None:
            self.get_logger().error('no /map yet.')
            return
        if self._start_obj is None:
            self.get_logger().error(
                'no object start pose. publish /initialpose first '
                '(RViz "2D Pose Estimate").')
            return
        with self._lock:
            if self._busy:
                self.get_logger().warn('a solve is already running; ignoring goal.')
                return
            self._busy = True
        goal = np.array([msg.pose.position.x, msg.pose.position.y,
                         quat_to_yaw(msg.pose.orientation)])
        threading.Thread(target=self._run, args=(goal,), daemon=True).start()

    # ------------------------------------------------------------- pipeline
    def _run(self, goal_obj):
        try:
            self._plan(goal_obj)
        except Exception as exc:  # noqa: BLE001 - keep the node alive
            self.get_logger().error(f'planning failed: {exc}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            with self._lock:
                self._busy = False

    def _plan(self, goal_obj):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        geom, lim, wts, opts = (self._geometry(), self._limits(),
                                self._weights(), self._solver_opts())
        log = self.get_logger().info
        start_obj = self._start_obj.copy()

        log(f'goal = ({goal_obj[0]:.2f}, {goal_obj[1]:.2f}, '
            f'{math.degrees(goal_obj[2]):.1f} deg)')

        # ---- 1. global seed ------------------------------------------------
        t0 = time.time()
        raw = ga.astar(self._sdf_full, start_obj[:2], goal_obj[:2],
                       clearance=float(p('astar_clearance')),
                       prefer_clearance=float(p('astar_prefer_clearance')),
                       clearance_weight=float(p('astar_clearance_weight')))
        if raw is None:
            self.get_logger().error(
                f'A* found no path with clearance {p("astar_clearance")} m. '
                'lower astar_clearance or check start/goal.')
            return
        seed = ga.shortcut(raw, self._sdf_full, float(p('astar_clearance')))
        length = ga.path_length(seed)
        log(f'A*: {len(raw)} -> {len(seed)} pts, {length:.2f} m, {time.time() - t0:.2f}s')

        # ---- 2. horizon ----------------------------------------------------
        dt = float(p('dt'))
        n_settle = int(p('n_settle'))
        # Slack matters: the seed length underestimates how far the formation
        # actually travels once it has to turn the bar, and a horizon sized
        # exactly to the seed simply stops short of the goal.
        n_move = int(math.ceil(float(p('horizon_slack')) * length
                               / max(1e-6, float(p('v_nom')) * dt)))
        n_cap = int(p('max_steps')) - n_settle
        if n_move > n_cap:
            need = length / (n_cap * dt)
            self.get_logger().warn(
                f'path is {length:.1f} m but max_steps caps the horizon at '
                f'{n_cap} steps ({n_cap * dt:.0f} s), which asks the object to '
                f'average {need:.2f} m/s against obj_v_max={lim.obj_v_max:.2f}. '
                'raise max_steps or dt, or the plan will stop short.')
        n_move = int(np.clip(n_move, 20, n_cap))
        ref_move = ga.resample_by_arclength(seed, n_move + 1)
        ref_move[0] = start_obj[:2]
        ref_move[-1] = goal_obj[:2]

        # ---- 3. object yaw seed -------------------------------------------
        # The bar is 180-degree symmetric, so pick whichever representative of
        # the goal yaw is closest to the path tangent instead of forcing a
        # gratuitous half turn at the end.
        period = math.pi if bool(p('object_symmetric')) else 2.0 * math.pi
        th = ga.tangent_yaw(ref_move)
        th = th + period * round((start_obj[2] - th[0]) / period)
        th = th + (start_obj[2] - th[0])
        goal_yaw_eff = goal_obj[2] + period * round((th[-1] - goal_obj[2]) / period)
        n_blend = max(5, n_move // 4)
        ramp = np.linspace(0.0, 1.0, n_blend)
        th[-n_blend:] = (1 - ramp) * th[-n_blend:] + ramp * goal_yaw_eff

        ref_xy = np.vstack([ref_move, np.repeat(goal_obj[:2][None, :], n_settle, axis=0)])
        yaw_guess = np.concatenate([th, np.full(n_settle, goal_yaw_eff)])
        goal_eff = np.array([goal_obj[0], goal_obj[1], goal_yaw_eff])
        log(f'horizon: N={len(ref_xy) - 1} ({n_move} moving + {n_settle} settling), '
            f'{(len(ref_xy) - 1) * dt:.1f} s')

        # ---- 4. crop the SDF around the seed -------------------------------
        stand = geom.obj_len / 2 + geom.reach_max + max(geom.base_len, geom.base_wid)
        sdf = self._sdf_full.crop(ref_xy, margin=float(p('sdf_crop_margin')) + stand)
        ds = int(p('sdf_downsample'))
        # The b-spline interpolant is fitted over every cell of the cropped
        # grid, so a long path across a big map has to be coarsened or the
        # NLP never even gets built.  Min-pooling keeps it conservative.
        budget = int(p('sdf_max_cells'))
        if budget > 0 and sdf.nx * sdf.ny // (ds * ds) > budget:
            ds = int(math.ceil(math.sqrt(sdf.nx * sdf.ny / budget)))
            self.get_logger().warn(
                f'cropped SDF is {sdf.nx}x{sdf.ny} cells; downsampling by {ds}x '
                f'to stay under sdf_max_cells={budget}. resolution becomes '
                f'{sdf.res * ds:.3f} m -- check that narrow doorways survive.')
        sdf = sdf.downsample(ds)
        log(f'SDF crop: {sdf.nx}x{sdf.ny} @ {sdf.res:.3f} m')

        # Fail fast on an unreachable boundary condition.  Both object poses are
        # hard constraints, so if the bar does not fit where it was clicked the
        # NLP is infeasible at that step and IPOPT will only say so after
        # burning a full solve.
        bad = False
        for tag, pose in (('start', start_obj), ('goal', goal_eff)):
            slack = pose_clearance(sdf, pose, geom.obj_len / 2, geom.obj_wid / 2,
                                   geom.n_obj_disks, geom.obj_margin)
            if slack < 0.0:
                self.get_logger().error(
                    f'{tag} object pose is infeasible by {-slack:.3f} m: a '
                    f'{geom.obj_len:.2f} x {geom.obj_wid:.2f} m bar does not fit '
                    f'there with a {geom.obj_margin:.2f} m margin. this pose is a '
                    'hard constraint -- move it, rotate it, or lower obj_margin. '
                    'no solver setting will fix it.')
                bad = True
            else:
                log(f'  {tag} object clearance {slack:+.3f} m')
        if bad:
            return

        # ---- 5. start base poses ------------------------------------------
        start_bases = None
        if bool(p('pin_base_start')):
            reach_nom = 0.5 * (geom.reach_min + geom.reach_max)
            u0 = np.array([math.cos(start_obj[2]), math.sin(start_obj[2])])
            off = geom.obj_len / 2 + reach_nom
            start_bases = (np.array([*(start_obj[:2] - off * u0), start_obj[2]]),
                           np.array([*(start_obj[:2] + off * u0), start_obj[2] + math.pi]))

        # ---- 6. solve ------------------------------------------------------
        self._publish_path(self.pub_ref, ref_xy, yaw_guess)
        t0 = time.time()
        res = solve_transport(sdf, ref_xy, yaw_guess, start_obj, start_bases,
                              goal_eff, dt, geom, lim, wts, opts, log=log)
        log(f'IPOPT finished in {time.time() - t0:.1f}s '
            f'(success={res["success"]}, {res["iters"]} iterations, N={res["N"]})')

        # ---- 7. verify against the raw grid --------------------------------
        # limits 를 넘겨야 연속성 판정이 산다 (한 스텝 이동량 vs v_max*dt)
        rep = verify(sdf, res, geom, lim)
        for tag in ('base1', 'base2', 'object'):
            r = rep[tag]
            log(f'  {tag:7s} min clearance {r["min_clearance"]:+.3f} m '
                f'(margin {r["required_margin"]:.2f}, worst at k={r["at_step"]})')
        log(f'  reach1 {rep["ee1_reach"]["min"]:.3f}..{rep["ee1_reach"]["max"]:.3f}  '
            f'reach2 {rep["ee2_reach"]["min"]:.3f}..{rep["ee2_reach"]["max"]:.3f}  '
            f'(allowed {geom.reach_min}..{geom.reach_max})')
        log(f'  bar length {rep["bar_length"]["min"]:.4f}..'
            f'{rep["bar_length"]["max"]:.4f} (nominal {geom.obj_len})')
        goal_err = float(np.linalg.norm(res['obj_xy'][-1] - goal_obj[:2]))
        log(f'  terminal object position error {goal_err:.3f} m')
        log(f'  object vs seed: mean {res["ref_dev_mean"]:.3f} m, '
            f'max {res["ref_dev_max"]:.3f} m  (w_ref={wts.ref:g})')
        if res['collision_slack_max'] > 1e-4:
            self.get_logger().warn(
                f'  collision slack used: max {res["collision_slack_max"]:.3f} m at '
                f'{res["collision_slack_n"]} samples.  the plan is being asked to go '
                'somewhere it does not fit -- raising w_ref will NOT help, the geometry '
                'has to change (margins, bar length, or a different route).')
        else:
            log('  collision slack 0 -- the shape of this plan is a weight trade-off, '
                'not a fit problem.')

        log(f'  dynamics residual {rep["worst_dynamics_residual"]:.2e} '
            f'(base1 pos {rep["base1_dynamics"]["max_pos_residual"]:.2e}, '
            f'base2 pos {rep["base2_dynamics"]["max_pos_residual"]:.2e})')
        bk = rep['between_knots']
        log(f'  between-knot clearance {bk["min_clearance"]:+.3f} m '
            f'({bk["worst_of"]} at k={bk["at_step"]}, {bk["samples_per_gap"]} samples/gap)')
        cont = rep['continuity']
        allow = cont.get('allowed_step_m', {})
        log(f'  max step  base1 {cont["base1"]["max_step_m"]:.4f}  '
            f'base2 {cont["base2"]["max_step_m"]:.4f}  '
            f'object {cont["object"]["max_step_m"]:.4f} m'
            + (f'  (allowed base {allow["base"]:.4f}, object {allow["object"]:.4f})'
               if allow else ''))

        if not res['success']:
            self.get_logger().error(
                'IPOPT did NOT converge -- this is the last iterate, not a solution. '
                'do not execute it. try a different reach_min/reach_max, '
                'raise max_iter, or set hessian:=limited-memory.')
        if not rep['dynamically_feasible']:
            self.get_logger().error(
                f'NOT dynamically feasible: residual '
                f'{rep["worst_dynamics_residual"]:.2e} m/rad per step. the base '
                'trajectory cannot be driven by a unicycle -- do not execute.')
        if not rep['collision_free']:
            self.get_logger().error(
                f'NOT collision free: worst clearance {rep["worst_clearance"]:+.3f} m.')
        if not rep['continuity']['continuous']:
            c = rep['continuity']
            self.get_logger().error(
                'NOT continuous: the base jumps up to '
                f'{max(c["base1"]["max_step_m"], c["base2"]["max_step_m"]):.3f} m in one '
                f'step against an allowed {c["allowed_step_m"]["base"]:.3f} m. '
                'this is a teleport, not a trajectory -- it will step over walls.')
        # ★ Interlock.  Until this was a flag it was only a log line, and a log
        #   line does not stop a click.  A non-converged iterate teleports the
        #   base through walls; publishing it for inspection is fine, streaming
        #   it is not.
        self._plan_ok = bool(rep['executable'] and res['success'])
        if self._plan_ok:
            log(f'VERIFIED executable: collision free with '
                f'{rep["worst_clearance"]:+.3f} m clearance beyond the disk radius, '
                f'dynamics residual {rep["worst_dynamics_residual"]:.2e}.')
        else:
            self.get_logger().warn(
                'publishing for inspection only -- EXECUTE is BLOCKED for this plan. '
                'override with ros2 param set /two_robot_nlp_planner allow_unsafe true')

        # ---- 8. publish ----------------------------------------------------
        self._publish_path(self.pub_obj, res['obj_xy'], res['obj_yaw'], dt)
        self._publish_path(self.pub_b1, res['base1_xy'], res['base1_yaw'], dt)
        self._publish_path(self.pub_b2, res['base2_xy'], res['base2_yaw'], dt)
        self._publish_path(self.pub_e1, res['ee1_xy'], res['ee1_yaw'], dt)
        self._publish_path(self.pub_e2, res['ee2_xy'], res['ee2_yaw'], dt)
        self._publish_markers(res, geom)
        self._anim_plan = (res, geom)
        self._anim_k = 0.0
        self._export_csv(res)

        # Park the button just clear of the plan's bounding box so it does not
        # sit on top of the robots in a top-down view.
        allxy = np.vstack([res['obj_xy'], res['base1_xy'], res['base2_xy']])
        self._im_pos = np.array([0.5 * (allxy[:, 0].min() + allxy[:, 0].max()),
                                 allxy[:, 1].max() + 3.0])
        self._refresh_button()

        na, nb = p('ns_a'), p('ns_b')
        log(f'published /{na}/ee_path /{nb}/ee_path /{na}/base_path /{nb}/base_path')
        log(f'setpoint stream: /{na}|{nb}/ee_pose, /{na}|{nb}/base_pose, '
            f'/{na}|{nb}/base_twist @ {p("stream_rate"):.0f} Hz')
        if bool(p('auto_start')):
            go = Bool(); go.data = True
            self.on_execute(go)
        else:
            log('to stream setpoints:  ros2 topic pub -1 /plan_execute '
                'std_msgs/msg/Bool "{data: true}"')

    # -------------------------------------------------------------- output
    def _publish_path(self, pub, xy, yaw, dt=None):
        """nav_msgs/Path carries no schedule of its own, so the plan's timing
        goes into each pose's own header.stamp: stamp[k] = t0 + k*dt.  RViz
        ignores it; a consumer can recover the trajectory from it."""
        msg = Path()
        msg.header.frame_id = self.map_frame
        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()
        base_ns = now.nanoseconds
        for k, ((x, y), th) in enumerate(zip(np.asarray(xy), np.asarray(yaw))):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            if dt is None:
                ps.header.stamp = msg.header.stamp
            else:
                t_ns = base_ns + int(k * dt * 1e9)
                ps.header.stamp.sec = int(t_ns // 1_000_000_000)
                ps.header.stamp.nanosec = int(t_ns % 1_000_000_000)
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation = yaw_to_quat(float(th))
            msg.poses.append(ps)
        pub.publish(msg)

    # ------------------------------------------------------- RViz button
    def _build_button(self):
        """Green plate when idle, red when streaming.  Left-click toggles.

        A second plate above it is the **SETUP** button for the Isaac Sim twin.
        The intended order is: look at the animation, press SETUP to spawn the
        simulated robots on the plan's start pose and let their arms settle onto
        the first grasp, then press EXECUTE.  SETUP changes nothing here -- it
        only publishes /twin/setup, so the planner behaves exactly as before if
        no simulator is listening.
        """
        self._build_setup_button()
        running = self._exec_t0 is not None
        if self._anim_plan is None:
            label, rgb = 'NO PLAN YET', (0.45, 0.45, 0.45)
        elif not self._plan_ok and not bool(self.get_parameter('allow_unsafe').value):
            # 검증 실패한 계획은 눌러도 안 나간다.  버튼이 초록이면 눌러도 된다는
            # 뜻이어야 하므로 상태를 색으로 드러낸다.
            label, rgb = 'PLAN UNSAFE - BLOCKED', (0.55, 0.15, 0.15)
        elif running:
            label, rgb = 'STOP', (0.85, 0.15, 0.15)
        else:
            label, rgb = 'EXECUTE  >>', (0.10, 0.70, 0.20)

        im = InteractiveMarker()
        im.header.frame_id = self.map_frame
        im.name = 'plan_button'
        im.description = ''
        im.scale = 2.0
        im.pose.position.x = float(self._im_pos[0])
        im.pose.position.y = float(self._im_pos[1])
        im.pose.position.z = 2.5
        im.pose.orientation.w = 1.0

        plate = Marker()
        plate.type = Marker.CUBE
        plate.scale.x, plate.scale.y, plate.scale.z = 3.0, 1.1, 0.15
        plate.color.r, plate.color.g, plate.color.b = rgb
        plate.color.a = 0.9

        text = Marker()
        text.type = Marker.TEXT_VIEW_FACING
        text.text = label
        text.scale.z = 0.6
        text.pose.position.z = 0.35
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0

        ctrl = InteractiveMarkerControl()
        ctrl.interaction_mode = InteractiveMarkerControl.BUTTON
        ctrl.always_visible = True
        ctrl.name = 'button'
        ctrl.markers.extend([plate, text])
        im.controls.append(ctrl)

        menu_ctrl = InteractiveMarkerControl()
        menu_ctrl.interaction_mode = InteractiveMarkerControl.MENU
        menu_ctrl.always_visible = True
        menu_ctrl.name = 'menu'
        im.controls.append(menu_ctrl)

        self._im_server.insert(im, feedback_callback=self._on_button)
        self._menu.apply(self._im_server, im.name)
        self._im_server.applyChanges()

    def _build_setup_button(self):
        """SETUP plate, stacked above EXECUTE.  Signals the Isaac Sim twin only."""
        if self._anim_plan is None:
            label, rgb = 'SETUP SIM  (no plan yet)', (0.45, 0.45, 0.45)
        elif self._setup_count:
            label, rgb = 'SETUP SIM  (sent x%d)' % self._setup_count, (0.55, 0.35, 0.80)
        else:
            label, rgb = 'SETUP SIM  (spawn robots)', (0.65, 0.45, 0.90)

        im = InteractiveMarker()
        im.header.frame_id = self.map_frame
        im.name = 'setup_button'
        im.description = ''
        im.scale = 2.0
        im.pose.position.x = float(self._im_pos[0])
        im.pose.position.y = float(self._im_pos[1]) + 1.4      # EXECUTE 위에 쌓는다
        im.pose.position.z = 2.5
        im.pose.orientation.w = 1.0

        plate = Marker()
        plate.type = Marker.CUBE
        plate.scale.x, plate.scale.y, plate.scale.z = 3.0, 1.1, 0.15
        plate.color.r, plate.color.g, plate.color.b = rgb
        plate.color.a = 0.9

        text = Marker()
        text.type = Marker.TEXT_VIEW_FACING
        text.text = label
        text.scale.z = 0.45
        text.pose.position.z = 0.35
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0

        ctrl = InteractiveMarkerControl()
        ctrl.interaction_mode = InteractiveMarkerControl.BUTTON
        ctrl.always_visible = True
        ctrl.name = 'button'
        ctrl.markers.extend([plate, text])
        im.controls.append(ctrl)

        self._im_server.insert(im, feedback_callback=self._on_button)
        self._im_server.applyChanges()

    def _send_setup(self):
        """Tell the simulator to spawn the robots on the plan's start pose.

        This does **not** touch the planner's own state.  The point is to look at
        the RViz animation first, and only then commit the simulated robots --
        spawning them the moment a plan is solved would drop them wherever the
        last click happened to be.
        """
        if self._anim_plan is None:
            self.get_logger().warn('no plan yet -- set a goal first.')
            return
        self._setup_count += 1
        msg = Bool()
        msg.data = True
        self.pub_setup.publish(msg)
        smp = self._sample_step(0.0)
        na, nb = self.get_parameter('ns_a').value, self.get_parameter('ns_b').value
        self.get_logger().info('SETUP -> /twin/setup #%d (simulator spawns at plan t=0)'
                               % self._setup_count)
        for side, ns in (('a', na), ('b', nb)):
            self.get_logger().info(
                '  %s: base (%.2f, %.2f, %.1f deg)  ee (%.2f, %.2f)'
                % (ns, smp['base'][side][0], smp['base'][side][1],
                   math.degrees(smp['base_yaw'][side]),
                   smp['ee'][side][0], smp['ee'][side][1]))
        self._refresh_button()

    def _on_menu_setup(self, _fb):
        self._send_setup()

    def _refresh_button(self):
        self._build_button()

    def _on_button(self, fb: InteractiveMarkerFeedback):
        if fb.event_type != InteractiveMarkerFeedback.BUTTON_CLICK:
            return
        # Debounce: a duplicated feedback message would otherwise toggle twice
        # and silently leave execution in the opposite state to the label.
        now = self.get_clock().now()
        if self._last_click is not None:
            if (now - self._last_click).nanoseconds * 1e-9 < 0.4:
                return
        self._last_click = now
        if fb.marker_name == 'setup_button':
            self._send_setup()
        else:
            self._toggle_execution()

    def _on_menu_execute(self, _fb):
        self._start_execution()

    def _on_menu_stop(self, _fb):
        self._stop_execution()

    def _toggle_execution(self):
        if self._exec_t0 is None:
            self._start_execution()
        else:
            self._stop_execution()

    def _start_execution(self):
        if self._anim_plan is None:
            self.get_logger().warn('nothing to execute yet -- set a goal first.')
            return
        if not self._plan_ok and not bool(self.get_parameter('allow_unsafe').value):
            self.get_logger().error(
                'EXECUTE blocked: this plan did not verify (see the errors above). '
                'streaming it would drive the base through walls. set '
                'allow_unsafe:=true only if you know why you want the last iterate.')
            return
        self._exec_t0 = self.get_clock().now()
        self._anim_k = 0.0          # animation restarts with the stream
        done = Bool(); done.data = False
        self.pub_done.publish(done)
        res = self._anim_plan[0]
        self.get_logger().info(
            f'EXECUTE: streaming setpoints from t=0 for {res["N"] * res["dt"]:.1f} s')
        self._refresh_button()

    def _stop_execution(self):
        if self._exec_t0 is None:
            return
        self._exec_t0 = None
        self.get_logger().info('STOP: execution halted, setpoints held.')
        self._refresh_button()

    # ---------------------------------------------------------- execution
    def on_execute(self, msg: Bool):
        """Same action as the RViz button, for scripted triggering."""
        if msg.data:
            self._start_execution()
        else:
            self._stop_execution()

    def _sample(self, t: float):
        """Sample the plan at wall-clock offset t seconds."""
        res, _ = self._anim_plan
        return self._sample_step(t / res['dt'])

    def _sample_step(self, s: float):
        """Linear interpolation of the plan at fractional step index s.

        All yaw series come out of the NLP unwrapped and continuous, so plain
        lerp is correct here -- no angle wrapping needed.

        Both the setpoint stream and the RViz animation go through this one
        function, so what you watch is exactly what is being published.
        """
        res, geom = self._anim_plan
        N = res['N']
        s = min(max(s, 0.0), float(N))
        k = int(min(s, N - 1e-9))
        a = s - k
        k1 = min(k + 1, N)

        def lerp(arr):
            return (1.0 - a) * arr[k] + a * arr[k1]

        # Interpolate the *object* pose and rederive the grasp points, rather
        # than interpolating the two EEs independently -- lerping two points of
        # a rotating rigid body shortens the chord between them.
        po = lerp(res['obj_xy'])
        tho = lerp(res['obj_yaw'])
        half = 0.5 * geom.obj_len * np.array([math.cos(tho), math.sin(tho)])

        return {
            's': s,
            't': s * res['dt'],
            'obj': po, 'obj_yaw': tho,
            'ee': {'a': po - half, 'b': po + half},
            'ee_yaw': {'a': tho, 'b': tho + math.pi},
            'base': {'a': lerp(res['base1_xy']), 'b': lerp(res['base2_xy'])},
            'base_yaw': {'a': lerp(res['base1_yaw']), 'b': lerp(res['base2_yaw'])},
            'v': {'a': lerp(res['v1']), 'b': lerp(res['v2'])},
            'w': {'a': lerp(res['w1']), 'b': lerp(res['w2'])},
            'finished': s >= N,
        }

    def _tick_stream(self):
        if self._anim_plan is None or self._exec_t0 is None:
            return
        t = (self.get_clock().now() - self._exec_t0).nanoseconds * 1e-9
        smp = self._sample(t)
        stamp = self.get_clock().now().to_msg()

        for side in ('a', 'b'):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.header.stamp = stamp
            ps.pose.position.x = float(smp['ee'][side][0])
            ps.pose.position.y = float(smp['ee'][side][1])
            ps.pose.orientation = yaw_to_quat(float(smp['ee_yaw'][side]))
            self.pub_ee_t[side].publish(ps)

            bp = PoseStamped()
            bp.header.frame_id = self.map_frame
            bp.header.stamp = stamp
            bp.pose.position.x = float(smp['base'][side][0])
            bp.pose.position.y = float(smp['base'][side][1])
            bp.pose.orientation = yaw_to_quat(float(smp['base_yaw'][side]))
            self.pub_base_t[side].publish(bp)

            tw = TwistStamped()
            tw.header.frame_id = self.map_frame
            tw.header.stamp = stamp
            tw.twist.linear.x = float(smp['v'][side])
            tw.twist.angular.z = float(smp['w'][side])
            self.pub_base_tw[side].publish(tw)

        if smp['finished']:
            self._exec_t0 = None
            done = Bool(); done.data = True
            self.pub_done.publish(done)
            self.get_logger().info('plan finished; holding the final setpoint.')
            self._refresh_button()

    C_BASE1 = (0.10, 0.75, 0.25)
    C_BASE2 = (0.15, 0.40, 0.95)
    C_OBJ = (0.65, 0.38, 0.16)

    def _mk(self, ns, mid, mtype):
        mk = Marker()
        mk.header.frame_id = self.map_frame
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = ns
        mk.id = mid
        mk.type = mtype
        mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        return mk

    def _outline(self, ns, mid, xy, yaw, half_l, half_w, rgb, alpha, width=0.025):
        mk = self._mk(ns, mid, Marker.LINE_STRIP)
        mk.scale.x = width
        mk.color.r, mk.color.g, mk.color.b = rgb
        mk.color.a = alpha
        c, s = math.cos(yaw), math.sin(yaw)
        for lx, ly in ((half_l, half_w), (half_l, -half_w),
                       (-half_l, -half_w), (-half_l, half_w), (half_l, half_w)):
            pt = Point()
            pt.x = float(xy[0] + c * lx - s * ly)
            pt.y = float(xy[1] + s * lx + c * ly)
            mk.points.append(pt)
        return mk

    def _box(self, ns, mid, xy, yaw, length, width, height, rgb, alpha, z=0.0):
        mk = self._mk(ns, mid, Marker.CUBE)
        mk.pose.position.x = float(xy[0])
        mk.pose.position.y = float(xy[1])
        mk.pose.position.z = float(z + height / 2)
        mk.pose.orientation = yaw_to_quat(float(yaw))
        mk.scale.x, mk.scale.y, mk.scale.z = float(length), float(width), float(height)
        mk.color.r, mk.color.g, mk.color.b = rgb
        mk.color.a = alpha
        return mk

    def _sphere(self, ns, mid, xy, diameter, rgb, alpha, z=0.0):
        mk = self._mk(ns, mid, Marker.SPHERE)
        mk.pose.position.x = float(xy[0])
        mk.pose.position.y = float(xy[1])
        mk.pose.position.z = float(z)
        mk.scale.x = mk.scale.y = mk.scale.z = float(diameter)
        mk.color.r, mk.color.g, mk.color.b = rgb
        mk.color.a = alpha
        return mk

    def _publish_markers(self, res, geom: Geometry):
        """Static ghosts: the swept corridor of the whole formation."""
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        n = res['N'] + 1
        count = max(2, int(self.get_parameter('ghost_count').value))
        idx = np.unique(np.linspace(0, n - 1, count).astype(int))
        for i, k in enumerate(idx):
            # fade from faint at the start to solid at the goal
            a = 0.25 + 0.55 * (i / max(1, len(idx) - 1))
            arr.markers.append(self._outline('ghost_base1', i, res['base1_xy'][k],
                                             res['base1_yaw'][k], geom.base_len / 2,
                                             geom.base_wid / 2, self.C_BASE1, a))
            arr.markers.append(self._outline('ghost_base2', i, res['base2_xy'][k],
                                             res['base2_yaw'][k], geom.base_len / 2,
                                             geom.base_wid / 2, self.C_BASE2, a))
            arr.markers.append(self._outline('ghost_object', i, res['obj_xy'][k],
                                             res['obj_yaw'][k], geom.obj_len / 2,
                                             geom.obj_wid / 2, self.C_OBJ, a, width=0.035))
        self.pub_mk.publish(arr)

    def _tick_animation(self):
        """Replay the formation so a doorway pass is legible.

        While executing this is slaved to the *same* wall clock and the same
        sampler as the setpoint stream, so the robots you see are the poses
        being published.  Free-running preview only applies when idle -- and it
        is tick-counted, which would slowly drift against wall time.
        """
        if self._anim_plan is None:
            return
        res, geom = self._anim_plan
        n = res['N'] + 1

        live = self._exec_t0 is not None
        if live:
            t = (self.get_clock().now() - self._exec_t0).nanoseconds * 1e-9
            self._anim_k = min(max(t / res['dt'], 0.0), float(res['N']))
        else:
            rate = float(self.get_parameter('animate_rate').value)
            speed = float(self.get_parameter('animate_speed').value)
            self._anim_k += speed / max(1e-6, rate * res['dt'])
            if self._anim_k >= n:
                self._anim_k = 0.0
        smp = self._sample_step(self._anim_k)
        b1, b2 = smp['base']['a'], smp['base']['b']
        e1, e2 = smp['ee']['a'], smp['ee']['b']

        arr = MarkerArray()
        arr.markers.append(self._box('anim', 0, b1, smp['base_yaw']['a'],
                                     geom.base_len, geom.base_wid, 0.30, self.C_BASE1, 0.95))
        arr.markers.append(self._box('anim', 1, b2, smp['base_yaw']['b'],
                                     geom.base_len, geom.base_wid, 0.30, self.C_BASE2, 0.95))
        arr.markers.append(self._box('anim', 2, smp['obj'], smp['obj_yaw'],
                                     geom.obj_len, geom.obj_wid, 0.12, self.C_OBJ, 0.95,
                                     z=0.45))
        arr.markers.append(self._sphere('anim', 3, e1, 0.16, (1.0, 1.0, 0.2), 1.0, z=0.51))
        arr.markers.append(self._sphere('anim', 4, e2, 0.16, (1.0, 1.0, 0.2), 1.0, z=0.51))
        # the arm is not modelled, only the reach constraint -- draw it as the
        # straight base->EE segment so you can see the annulus being used
        for i, (b, e) in enumerate(((b1, e1), (b2, e2))):
            mk = self._mk('anim', 5 + i, Marker.LINE_STRIP)
            mk.scale.x = 0.05
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.9, 0.9, 0.9, 0.9
            for pt_xy, pz in ((b, 0.30), (e, 0.51)):
                pt = Point()
                pt.x, pt.y, pt.z = float(pt_xy[0]), float(pt_xy[1]), pz
                mk.points.append(pt)
            arr.markers.append(mk)

        txt = self._mk('anim', 7, Marker.TEXT_VIEW_FACING)
        txt.pose.position.x = float(smp['obj'][0])
        txt.pose.position.y = float(smp['obj'][1]) + 1.2
        txt.pose.position.z = 1.2
        txt.scale.z = 0.4
        txt.color.a = 1.0
        if live:
            txt.color.r, txt.color.g, txt.color.b = 1.0, 0.3, 0.3
            tag = 'LIVE (publishing)'
        else:
            txt.color.r = txt.color.g = txt.color.b = 1.0
            tag = 'preview'
        txt.text = f'{tag}   t = {smp["t"]:.1f} / {res["N"] * res["dt"]:.1f} s'
        arr.markers.append(txt)
        self.pub_anim.publish(arr)

    def _publish_sdf_viz(self):
        """Show what the optimizer actually sees: clearance in metres, 0..2 m."""
        sdf = self._sdf_full
        if sdf is None:
            return
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = sdf.res
        msg.info.width = sdf.nx
        msg.info.height = sdf.ny
        msg.info.origin.position.x = float(sdf.origin[0])
        msg.info.origin.position.y = float(sdf.origin[1])
        msg.info.origin.orientation.w = 1.0
        v = np.clip(sdf.sdf, 0.0, 2.0) / 2.0            # (nx, ny)
        cost = (100.0 * (1.0 - v)).astype(np.int8)
        msg.data = cost.T.ravel(order='C').tolist()      # back to row-major (y, x)
        self.pub_sdf.publish(msg)

    def _export_csv(self, res):
        out_dir = self.get_parameter('csv_dir').value
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        rows = {
            'object': (res['obj_xy'], res['obj_yaw']),
            'base_A': (res['base1_xy'], res['base1_yaw']),
            'base_B': (res['base2_xy'], res['base2_yaw']),
            'ee_A': (res['ee1_xy'], res['ee1_yaw']),
            'ee_B': (res['ee2_xy'], res['ee2_yaw']),
        }
        for name, (xy, yaw) in rows.items():
            path = os.path.join(out_dir, f'{name}.csv')
            with open(path, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['t', 'x', 'y', 'yaw'])
                for t, (x, y), th in zip(res['t'], xy, yaw):
                    w.writerow([f'{t:.4f}', f'{x:.5f}', f'{y:.5f}', f'{th:.6f}'])
        self.get_logger().info(f'CSV written to {out_dir}')


def main():
    rclpy.init()
    node = TwoRobotNlpPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # launch sends SIGINT and rclpy may already have shut the context down;
        # calling it twice raises and makes launch report "process has died"
        # on a perfectly clean exit.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
