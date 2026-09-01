"""Single-shot trajectory optimization for two mobile bases carrying one object.

Formulation
-----------
Decision variables per step k = 0..N:

    base i    p_i = (x, y), psi_i, v_i, w_i          unicycle, acceleration input
    object    p_o = (x, y), th_o                     free planar pose

The object pose is an *independent* variable rather than the output of an arm
kinematic chain.  Two consequences, both of them the point of the exercise:

  * the two grasp points are ee_1 = p_o - (L/2)*u, ee_2 = p_o + (L/2)*u with
    u = (cos th_o, sin th_o), so ||ee_1 - ee_2|| == L holds *exactly* by
    construction.  No consensus residual, no rigid-body constraint to enforce.

  * the arm is modelled by nothing but a reach annulus,
    reach_min <= ||ee_i - p_i|| <= reach_max, plus "the EE is in front of the
    base".  That is deliberately loose -- the RL policy resolves the rest.

Collision uses circle covers of both rectangles queried against a signed
distance field, which makes every constraint a *hard* inequality rather than a
penalty.  That is the whole reason for moving off iLQR: "no collision" becomes
something the solver certifies instead of something a weight has to win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np


# --------------------------------------------------------------------- config
@dataclass
class Geometry:
    base_len: float = 0.80          # AMR footprint, along its heading
    base_wid: float = 0.55
    n_base_disks: int = 3
    base_margin: float = 0.05       # extra clearance on top of the disk radius

    obj_len: float = 2.00           # grasp point to grasp point == rectangle length
    obj_wid: float = 0.30
    n_obj_disks: int = 5
    obj_margin: float = 0.05

    reach_min: float = 0.25         # ||ee - base_centre|| bounds
    reach_max: float = 0.6
    ee_forward_min: float = 0.10    # EE must be this far ahead of the base centre

    map_inflation: float = 0.0
    """How much the *map* over-states every obstacle [m].  Subtracted from every
    required clearance.

    A 2D lidar SLAM map draws walls thicker than they are: the beam endpoint
    cell is marked occupied, and on a 0.05 m grid that quantisation alone costs
    up to one cell per surface.  Measured on munji_3f_2025_wide, wall local
    thickness is a median of 0.100 m -- exactly two cells -- for what are real
    interior walls.

    **Do not implement this by eroding the map.**  Eroding the wall class by one
    cell on this map deletes 91.9 % of the wall cells (two-cell walls simply
    vanish) and opens passages that do not exist.  Subtracting from the required
    clearance is safe by construction: passing through a wall needs sdf < 0,
    while the requirement stays at ``r + margin - map_inflation`` > 0, so no
    amount of this can tunnel.  It only lets the robot hug a surface closer.
    """


@dataclass
class Limits:
    v_max: float = 0.60
    v_min: float = -0.50            # one robot leads the bar and therefore
                                    # drives in reverse the whole way; do not
                                    # throttle it by accident
    w_max: float = 1.00
    a_max: float = 0.80
    alpha_max: float = 2.00
    obj_v_max: float = 0.60
    obj_w_max: float = 0.60
    ee_v_max: float = 0.70          # what the RL tracker actually has to follow:
                                    # object rotation adds (obj_len/2)*w on top
                                    # of the centre speed
    robot_robot_min: float = 0.90   # centre-to-centre


@dataclass
class Weights:
    ref: float = 20.0               # follow the A* seed position
    """How hard the object holds the planned route.

    Raised from 2.0 after measuring: the base terms added below (base_ref 3,
    base_vel 8) are paid for by the object drifting off its seed, because the
    object is carried by the bases.  At 2.0 it wandered a mean 0.111 m / max
    0.239 m from the seed; at 20.0 that is 0.022 / 0.078 m and the base path
    grows only 2 % (6.37 -> 6.48 m).  Past ~20 the remaining error is the A*
    seed's own polyline quality, not something worth chasing.
    """
    yaw_ref: float = 5.0            # ... and, weakly, its tangent heading.
                                    # Without this the bar wanders tens of
                                    # degrees for no benefit and burns the
                                    # horizon; it is soft enough that the bar
                                    # still turns freely to clear a doorway.
    goal_pos: float = 2000.0
    goal_yaw: float = 60.0
    u: float = 0.05
    du: float = 0.50
    obj_vel: float = 1.0
    obj_acc: float = 20.0
    obj_yaw_rate: float = 20.0
    obj_yaw_acc: float = 200.0
    term_vel: float = 50.0
    collision: float = 1.0e6
    """Penalty per metre of collision slack.  This has to be an *exact* penalty,
    not a trade-off, and 5000 was not.

    The slack exists so the solver always has somewhere to go instead of
    entering IPOPT's restoration phase (which is where teleporting iterates come
    from).  It must never be worth spending.  Measured failure at 5000: a plan
    bought 0.40 m of slack -- driving the base through 13 wall cells -- and
    IPOPT reported "Optimal Solution Found", because::

        slack 0.4 m over ~30 samples  ~ 5000 * (30*0.4 + 30*0.16) = 84,000
        missing the goal by 6 m       = w_goal_pos * 36 = 2,000 * 36 = 72,000

    i.e. cutting through the wall was the cheaper option.  The weight has to
    dominate the largest competing term, which is ``goal_pos`` times the square
    of the biggest miss the map allows (~50 m -> 5e6).  1e6 per metre with the
    L1 term makes any real violation cost more than every alternative combined.

    verify() still checks the truth against the raw grid and the node still
    blocks a plan that leans on the slack -- this weight is about not *producing*
    such plans, not about catching them.
    """

    # ---- formation: keep the two AMRs single file behind their grasps -----
    #
    # Everything above is about the *object*.  Nothing was about where the bases
    # go, so within the reach annulus (0.15..0.60 m is a 0.45 m wide band) and
    # with ee_forward_min letting the EE sit behind the base, the bases had a
    # very large null space and wandered in it.  Penalising acceleration does
    # not help: a long meandering path at constant speed is nearly free.
    #
    # The seed already has the shape we want -- base_guess_from_object puts one
    # base trailing its grasp and the other leading, both facing in, which is
    # exactly the single-file posture that fits through a door.  These terms ask
    # the solution to stay near that posture instead of only starting there.
    # They are soft on purpose: a base still swings wide when an obstacle makes
    # it necessary, it just no longer does so for free.
    base_ref: float = 3.0
    """Pull each base toward its standoff pose along the bar axis.

    This is what produces the single-file posture.  Measured effect on the
    lateral offset of the base from the bar axis: 0.415 -> 0.045 m.
    """
    base_yaw_ref: float = 0.0
    """Keep each base facing its own grasp point.  **Default off, and that is
    the measured answer, not a guess.**

    A differential drive cannot move sideways, so asking it to hold both a
    position and an unrelated heading makes it turn-drive-turn its way along.
    Measured on one plan: |w| median 0.0076 -> 0.132 rad/s (saturating the
    0.60 limit), sign reversals 27 -> 25 on much larger amplitudes, and base
    path 8.3 -> 13.2 m.  The natural heading of a diff drive is its own
    direction of travel; let it be.
    """
    base_vel: float = 8.0
    """Penalise base travel, not just acceleration.  This is what stops the
    'wander at constant speed, pay almost nothing' solutions."""


@dataclass
class SolverOpts:
    max_iter: int = 3000
    tol: float = 1e-4
    acceptable_tol: float = 1e-3
    print_level: int = 5
    hessian: str = 'exact'          # or 'limited-memory'
    sdf_interp: str = 'bspline'     # or 'linear'
    collision_substeps: int = -1
    """Extra collision samples **between** consecutive knots.

    ``-1`` (default) derives the answer from the limits -- see
    :func:`required_substeps`.  ``0`` forces knot-only, ``n`` forces n samples.

    Why it is not simply "always on": it roughly doubles the collision
    constraint count and cost 58 -> 479 s on a 564 step plan.  Whether that buys
    anything is decidable, not a matter of taste, so let the code decide.
    """
    extra: dict = field(default_factory=dict)


def required_substeps(geom: Geometry, lim: Limits, dt: float) -> tuple:
    """Do the SDF inequalities need samples between the knots?  -> (n, why)

    The signed distance field is 1-Lipschitz (it is a Euclidean distance, and
    the Gaussian smoothing is a convolution with a probability density, which
    cannot raise the Lipschitz constant).  A point on the segment between two
    knots is at most **half a step** from each end, so::

        sdf(mid) >= sdf(knot) - step/2 >= (r + margin) - step/2

    which stays positive as long as ``margin > step/2``.  The step a cover disk
    takes is translation plus the arc its offset sweeps::

        step = v_max*dt + |offset|_max * w_max * dt

    So tunnelling is not a matter of luck at these speeds -- it is impossible,
    and the check is a two-line inequality.  Raise the speeds or drop the
    margins far enough and it stops holding, which is exactly when the extra
    samples should switch themselves on.
    """
    base_off, _ = cover_rect_with_disks(geom.base_len / 2, geom.base_wid / 2,
                                        geom.n_base_disks)
    obj_off, _ = cover_rect_with_disks(geom.obj_len / 2, geom.obj_wid / 2,
                                       geom.n_obj_disks)
    v = max(abs(lim.v_max), abs(lim.v_min))
    base_step = v * dt + max(abs(o) for o in base_off) * lim.w_max * dt
    obj_step = lim.obj_v_max * dt + max(abs(o) for o in obj_off) * lim.obj_w_max * dt
    need_base = base_step / 2.0
    need_obj = obj_step / 2.0
    ok = geom.base_margin > need_base and geom.obj_margin > need_obj
    why = ('base step %.4f m needs margin > %.4f (have %.3f);  '
           'object step %.4f m needs margin > %.4f (have %.3f)'
           % (base_step, need_base, geom.base_margin,
              obj_step, need_obj, geom.obj_margin))
    if ok:
        return 0, 'knot-only is provably enough -- ' + why
    # One midpoint halves the gap, two thirds it, ...  pick the smallest n that
    # brings the worst case under the margin.
    n = 1
    while n < 8:
        if (geom.base_margin > need_base / (n + 1)
                and geom.obj_margin > need_obj / (n + 1)):
            break
        n += 1
    return n, 'knot-only would NOT be safe, using %d between-knot samples -- %s' % (n, why)


# ------------------------------------------------------------------ geometry
def cover_rect_with_disks(half_len: float, half_wid: float, n: int):
    """Exact circle cover of a 2*half_len x 2*half_wid rectangle.

    n disks of radius sqrt(half_wid^2 + (half_len/n)^2) spaced along the long
    axis.  n=1 degenerates to the circumscribed circle; n=3 already cuts the
    lateral inflation of an 0.8x0.55 AMR from 0.21 m down to 0.03 m.
    """
    n = max(1, int(n))
    r = float(np.hypot(half_wid, half_len / n))
    offs = -half_len + (2 * np.arange(n) + 1) * (half_len / n)
    return offs.astype(float), r


def pose_clearance(sdf, pose, half_len: float, half_wid: float, n_disks: int,
                   margin: float) -> float:
    """Slack of one rectangle pose against the SDF; negative means infeasible.

    Worth checking before solving: the start and goal object poses enter the
    NLP as hard constraints, so if the bar does not fit where it was clicked
    the problem is infeasible at step 0 and no solver setting can rescue it.
    """
    offs, r = cover_rect_with_disks(half_len, half_wid, n_disks)
    yaw = float(pose[2])
    u = np.array([np.cos(yaw), np.sin(yaw)])
    pts = np.asarray(pose[:2], dtype=np.float64)[None, :] + offs[:, None] * u[None, :]
    return float(np.min(sdf.value(pts)) - r - margin)


def base_guess_from_object(ref_xy, yaw, stand_off: float):
    """Single-file seed: base 1 trails ee_1, base 2 leads ee_2, both facing in.

    ``stand_off`` is measured from the *object centre*, i.e. obj_len/2 + reach.
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    u = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    return ref_xy - stand_off * u, ref_xy + stand_off * u


def _finite_diff_states(p, psi, dt):
    """v, w guesses consistent with a positional guess."""
    d = np.diff(p, axis=0)
    v = np.sum(d * np.stack([np.cos(psi[:-1]), np.sin(psi[:-1])], axis=1), axis=1) / dt
    v = np.concatenate([v, v[-1:]])
    w = np.diff(psi) / dt
    w = np.concatenate([w, w[-1:]])
    return v, w


# --------------------------------------------------------------------- solve
def solve_transport(sdf, ref_xy, yaw_guess, start_obj, start_bases,
                    goal_obj, dt: float,
                    geom: Geometry, lim: Limits, wts: Weights, opts: SolverOpts,
                    log=print):
    """Build and solve the NLP.

    ref_xy      (N+1, 2)  object-centre seed from A*
    yaw_guess   (N+1,)    object-yaw seed, unwrapped
    start_obj   (3,)      hard initial object pose (x, y, yaw)
    start_bases ((3,), (3,)) hard initial base poses, or None to let the solver
                          place them (still subject to reach and collision)
    goal_obj    (3,)      terminal object pose target (soft)
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
    yaw_guess = np.asarray(yaw_guess, dtype=np.float64).reshape(-1)
    N = len(ref_xy) - 1
    reach_nom = 0.5 * (geom.reach_min + geom.reach_max)

    base_off, base_r = cover_rect_with_disks(geom.base_len / 2, geom.base_wid / 2,
                                             geom.n_base_disks)
    obj_off, obj_r = cover_rect_with_disks(geom.obj_len / 2, geom.obj_wid / 2,
                                           geom.n_obj_disks)
    log(f'[nlp] N={N} dt={dt:.3f}  base disks: n={len(base_off)} r={base_r:.3f} '
        f'off={np.round(base_off, 3).tolist()}')
    log(f'[nlp] object disks: n={len(obj_off)} r={obj_r:.3f} '
        f'off={np.round(obj_off, 3).tolist()}')

    sdf_fn = sdf.casadi_interpolant('sdf', opts.sdf_interp)

    opti = ca.Opti()
    P1 = opti.variable(2, N + 1); PSI1 = opti.variable(1, N + 1)
    V1 = opti.variable(1, N + 1); W1 = opti.variable(1, N + 1)
    P2 = opti.variable(2, N + 1); PSI2 = opti.variable(1, N + 1)
    V2 = opti.variable(1, N + 1); W2 = opti.variable(1, N + 1)
    PO = opti.variable(2, N + 1); THO = opti.variable(1, N + 1)
    A1 = opti.variable(1, N); AL1 = opti.variable(1, N)
    A2 = opti.variable(1, N); AL2 = opti.variable(1, N)

    # ---- unicycle dynamics (explicit midpoint) --------------------------
    def add_dynamics(P, PSI, V, W, A, AL):
        v_m = V[:, :N] + 0.5 * dt * A
        psi_m = PSI[:, :N] + 0.5 * dt * W[:, :N]
        opti.subject_to(P[:, 1:] == P[:, :N] + dt * ca.vertcat(v_m * ca.cos(psi_m),
                                                              v_m * ca.sin(psi_m)))
        opti.subject_to(PSI[:, 1:] == PSI[:, :N] + dt * (W[:, :N] + 0.5 * dt * AL))
        opti.subject_to(V[:, 1:] == V[:, :N] + dt * A)
        opti.subject_to(W[:, 1:] == W[:, :N] + dt * AL)

    add_dynamics(P1, PSI1, V1, W1, A1, AL1)
    add_dynamics(P2, PSI2, V2, W2, A2, AL2)

    # ---- explicit step bound --------------------------------------------
    # The dynamics above are *equalities*.  An interior point method is free to
    # violate an equality while iterating, and when it does the position
    # variables are joined by nothing at all -- which is exactly how a
    # non-converged iterate ends up teleporting the base across a wall.
    #
    # This inequality says the same thing the dynamics imply, but as a bound the
    # solver has to respect on its way there.  It is redundant at the solution
    # and cheap (2N scalars), and it is what stops "the last iterate" from being
    # nonsense rather than merely inexact.
    v_step = max(abs(lim.v_max), abs(lim.v_min)) * dt
    for P in (P1, P2):
        dP_ = P[:, 1:] - P[:, :N]
        opti.subject_to(ca.vec(dP_[0, :] ** 2 + dP_[1, :] ** 2) <= v_step ** 2)
    for PSI in (PSI1, PSI2):
        opti.subject_to(opti.bounded(-lim.w_max * dt,
                                     ca.vec(PSI[:, 1:] - PSI[:, :N]),
                                     lim.w_max * dt))

    # ---- object rate limits + grasp points ------------------------------
    dPO = PO[:, 1:] - PO[:, :N]
    opti.subject_to(ca.vec(dPO[0, :] ** 2 + dPO[1, :] ** 2) <= (lim.obj_v_max * dt) ** 2)
    dTHO = THO[:, 1:] - THO[:, :N]
    opti.subject_to(opti.bounded(-lim.obj_w_max * dt, ca.vec(dTHO), lim.obj_w_max * dt))

    u_obj = ca.vertcat(ca.cos(THO), ca.sin(THO))            # 2 x (N+1)
    EE1 = PO - (geom.obj_len / 2) * u_obj
    EE2 = PO + (geom.obj_len / 2) * u_obj

    # ---- reach: the only thing standing in for the arm ------------------
    for EE, P, PSI in ((EE1, P1, PSI1), (EE2, P2, PSI2)):
        dEE = EE[:, 1:] - EE[:, :N]
        opti.subject_to(ca.vec(dEE[0, :] ** 2 + dEE[1, :] ** 2) <= (lim.ee_v_max * dt) ** 2)
        d = EE - P
        opti.subject_to(opti.bounded(geom.reach_min ** 2,
                                     ca.vec(d[0, :] ** 2 + d[1, :] ** 2),
                                     geom.reach_max ** 2))
        if geom.ee_forward_min > -1e3:
            fwd = ca.cos(PSI) * d[0, :] + ca.sin(PSI) * d[1, :]
            opti.subject_to(ca.vec(fwd) >= geom.ee_forward_min)

    # ---- collision: hard SDF inequalities on every cover disk -----------
    c1 = ca.vertcat(ca.cos(PSI1), ca.sin(PSI1))
    c2 = ca.vertcat(ca.cos(PSI2), ca.sin(PSI2))
    blocks, radii = [], []
    infl = max(0.0, float(getattr(geom, 'map_inflation', 0.0)))
    req_base = max(1e-3, base_r + geom.base_margin - infl)
    req_obj = max(1e-3, obj_r + geom.obj_margin - infl)
    if infl > 0.0:
        log(f'[nlp] map_inflation {infl:.3f} m -> required clearance '
            f'base {base_r + geom.base_margin:.3f} -> {req_base:.3f}, '
            f'object {obj_r + geom.obj_margin:.3f} -> {req_obj:.3f}')
    for off in base_off:
        blocks.append(P1 + float(off) * c1); radii.append(req_base)
        blocks.append(P2 + float(off) * c2); radii.append(req_base)
    for off in obj_off:
        blocks.append(PO + float(off) * u_obj); radii.append(req_obj)

    Q = ca.horzcat(*blocks)                                  # 2 x M
    M = Q.shape[1]
    D = sdf_fn.map(M)(Q)                                     # 1 x M
    r_req = np.repeat(np.asarray(radii), N + 1)

    # ---- soft, but expensive ---------------------------------------------
    # Hard collision inequalities make the problem infeasible whenever the seed
    # is even slightly bad, and IPOPT answers infeasibility with its restoration
    # phase, which wanders far from anything physical.  That wandering *is* the
    # teleporting plan.
    #
    # A slack that costs `wts.collision` per metre keeps the problem feasible so
    # the solver always has somewhere to go, and the penalty is heavy enough that
    # any real solution drives it to zero.  verify() still checks the true
    # clearance against the raw grid, so a plan that leans on the slack is caught
    # and blocked -- softening the constraint does not soften the guarantee.
    Sc = opti.variable(1, M)
    opti.subject_to(ca.vec(Sc) >= 0)
    opti.subject_to(ca.vec(D) >= r_req.reshape(-1, 1) - ca.vec(Sc))
    opti.set_initial(Sc, 0.0)

    # ---- collision between knots ----------------------------------------
    # The inequalities above pin only the N+1 grid points.  The straight segment
    # joining two clear knots is unconstrained, so a thin wall can in principle
    # be stepped over.  Sample the midpoints as well.  The midpoint pose is the
    # average of its neighbours, which for a unicycle with these step sizes sits
    # inside the true arc -- i.e. conservative.
    n_sub = int(getattr(opts, 'collision_substeps', -1))
    if n_sub < 0:
        n_sub, why = required_substeps(geom, lim, dt)
        log(f'[nlp] collision substeps: auto -> {n_sub}.  {why}')
    n_sub = max(0, n_sub)
    if n_sub > 0:
        mid_blocks, mid_radii = [], []
        for P, PSI, off_list, rad in ((P1, PSI1, base_off, req_base),
                                      (P2, PSI2, base_off, req_base)):
            for f in [(i + 1.0) / (n_sub + 1.0) for i in range(n_sub)]:
                Pm = (1.0 - f) * P[:, :N] + f * P[:, 1:]
                PSIm = (1.0 - f) * PSI[:, :N] + f * PSI[:, 1:]
                cm = ca.vertcat(ca.cos(PSIm), ca.sin(PSIm))
                for off in off_list:
                    mid_blocks.append(Pm + float(off) * cm)
                    mid_radii.append(rad)
        for f in [(i + 1.0) / (n_sub + 1.0) for i in range(n_sub)]:
            POm = (1.0 - f) * PO[:, :N] + f * PO[:, 1:]
            THOm = (1.0 - f) * THO[:, :N] + f * THO[:, 1:]
            um = ca.vertcat(ca.cos(THOm), ca.sin(THOm))
            for off in obj_off:
                mid_blocks.append(POm + float(off) * um)
                mid_radii.append(req_obj)
        Qm = ca.horzcat(*mid_blocks)
        Mm = Qm.shape[1]
        Dm = sdf_fn.map(Mm)(Qm)
        Sm = opti.variable(1, Mm)
        opti.subject_to(ca.vec(Sm) >= 0)
        opti.subject_to(ca.vec(Dm) >= np.repeat(np.asarray(mid_radii), N).reshape(-1, 1)
                        - ca.vec(Sm))
        opti.set_initial(Sm, 0.0)
        log(f'[nlp] collision: {M} knot samples + {Qm.shape[1]} between-knot samples '
            f'(substeps={n_sub})')

    # ---- keep the two AMRs off each other -------------------------------
    dP = P1 - P2
    opti.subject_to(ca.vec(dP[0, :] ** 2 + dP[1, :] ** 2) >= lim.robot_robot_min ** 2)

    # ---- box limits ------------------------------------------------------
    for V, W, A, AL in ((V1, W1, A1, AL1), (V2, W2, A2, AL2)):
        opti.subject_to(opti.bounded(lim.v_min, ca.vec(V), lim.v_max))
        opti.subject_to(opti.bounded(-lim.w_max, ca.vec(W), lim.w_max))
        opti.subject_to(opti.bounded(-lim.a_max, ca.vec(A), lim.a_max))
        opti.subject_to(opti.bounded(-lim.alpha_max, ca.vec(AL), lim.alpha_max))

    # stay inside the interpolant's support, otherwise the b-spline extrapolates
    lo, hi = sdf.bounds(inset=2.5 * sdf.res)
    for P in (P1, P2, PO):
        opti.subject_to(opti.bounded(lo[0], ca.vec(P[0, :]), hi[0]))
        opti.subject_to(opti.bounded(lo[1], ca.vec(P[1, :]), hi[1]))

    # ---- boundary conditions --------------------------------------------
    start_obj = np.asarray(start_obj, dtype=np.float64).reshape(3)
    goal_obj = np.asarray(goal_obj, dtype=np.float64).reshape(3)

    opti.subject_to(PO[:, 0] == start_obj[:2].reshape(2, 1))
    opti.subject_to(THO[0, 0] == start_obj[2])
    opti.subject_to(V1[0, 0] == 0.0); opti.subject_to(W1[0, 0] == 0.0)
    opti.subject_to(V2[0, 0] == 0.0); opti.subject_to(W2[0, 0] == 0.0)

    if start_bases is None:
        # Only the object start is known; let the solver choose where the bases
        # stand, which is often what you actually want out of this.
        u0 = np.array([np.cos(start_obj[2]), np.sin(start_obj[2])])
        sb1 = np.concatenate([start_obj[:2] - (geom.obj_len / 2 + reach_nom) * u0,
                              [start_obj[2]]])
        sb2 = np.concatenate([start_obj[:2] + (geom.obj_len / 2 + reach_nom) * u0,
                              [start_obj[2] + np.pi]])
    else:
        sb1 = np.asarray(start_bases[0], dtype=np.float64).reshape(3)
        sb2 = np.asarray(start_bases[1], dtype=np.float64).reshape(3)
        opti.subject_to(P1[:, 0] == sb1[:2].reshape(2, 1))
        opti.subject_to(PSI1[0, 0] == sb1[2])
        opti.subject_to(P2[:, 0] == sb2[:2].reshape(2, 1))
        opti.subject_to(PSI2[0, 0] == sb2[2])

    # ---- objective -------------------------------------------------------
    ref = ca.DM(ref_xy.T)                                    # 2 x (N+1)
    J = wts.ref * ca.sumsqr(PO - ref)
    J += wts.yaw_ref * ca.sumsqr(THO - ca.DM(yaw_guess.reshape(1, -1)))
    J += wts.goal_pos * ca.sumsqr(PO[:, -1] - ca.DM(goal_obj[:2].reshape(2, 1)))
    J += wts.goal_yaw * (THO[0, -1] - goal_obj[2]) ** 2
    J += wts.u * (ca.sumsqr(A1) + ca.sumsqr(AL1) + ca.sumsqr(A2) + ca.sumsqr(AL2))
    if N > 1:
        J += wts.du * (ca.sumsqr(ca.diff(A1, 1, 1)) + ca.sumsqr(ca.diff(AL1, 1, 1))
                       + ca.sumsqr(ca.diff(A2, 1, 1)) + ca.sumsqr(ca.diff(AL2, 1, 1)))
        J += wts.obj_acc * ca.sumsqr(ca.diff(dPO, 1, 1))
        J += wts.obj_yaw_acc * ca.sumsqr(ca.diff(dTHO, 1, 1))
    J += wts.obj_vel * ca.sumsqr(dPO)
    J += wts.obj_yaw_rate * ca.sumsqr(dTHO)
    J += wts.term_vel * (V1[0, -1] ** 2 + W1[0, -1] ** 2 + V2[0, -1] ** 2 + W2[0, -1] ** 2)
    J += wts.collision * (ca.sum2(Sc) + ca.sumsqr(Sc))
    if n_sub > 0:
        J += wts.collision * (ca.sum2(Sm) + ca.sumsqr(Sm))

    # ---- formation shape -------------------------------------------------
    # Desired standoff is expressed against the *current* object pose, not a
    # fixed path, so this is a shape cost rather than a second trajectory to
    # track: "stay behind your own grasp point, wherever the bar is".
    stand = geom.obj_len / 2 + reach_nom
    B1_des = PO - stand * u_obj
    B2_des = PO + stand * u_obj
    J += wts.base_ref * (ca.sumsqr(P1 - B1_des) + ca.sumsqr(P2 - B2_des))
    # Heading: base 1 looks along the bar, base 2 looks back down it.  Squared
    # difference of unwrapped angles is fine here -- both series come out of the
    # NLP continuous, and the seed already puts them on the right branch.
    J += wts.base_yaw_ref * (ca.sumsqr(PSI1 - THO) + ca.sumsqr(PSI2 - (THO + np.pi)))
    # Travel, not just acceleration.
    J += wts.base_vel * (ca.sumsqr(V1) + ca.sumsqr(V2))

    opti.minimize(J)

    # ---- warm start ------------------------------------------------------
    b1_xy, b2_xy = base_guess_from_object(ref_xy, yaw_guess,
                                          geom.obj_len / 2 + reach_nom)
    b1_xy[0] = sb1[:2]; b2_xy[0] = sb2[:2]
    psi1_g = np.unwrap(yaw_guess.copy()); psi1_g[0] = sb1[2]
    psi2_g = np.unwrap(yaw_guess + np.pi); psi2_g[0] = sb2[2]
    v1_g, w1_g = _finite_diff_states(b1_xy, psi1_g, dt)
    v2_g, w2_g = _finite_diff_states(b2_xy, psi2_g, dt)

    opti.set_initial(PO, ref_xy.T)
    opti.set_initial(THO, yaw_guess.reshape(1, -1))
    opti.set_initial(P1, b1_xy.T); opti.set_initial(PSI1, psi1_g.reshape(1, -1))
    opti.set_initial(P2, b2_xy.T); opti.set_initial(PSI2, psi2_g.reshape(1, -1))
    opti.set_initial(V1, np.clip(v1_g, lim.v_min, lim.v_max).reshape(1, -1))
    opti.set_initial(V2, np.clip(v2_g, lim.v_min, lim.v_max).reshape(1, -1))
    opti.set_initial(W1, np.clip(w1_g, -lim.w_max, lim.w_max).reshape(1, -1))
    opti.set_initial(W2, np.clip(w2_g, -lim.w_max, lim.w_max).reshape(1, -1))

    # ---- solve -----------------------------------------------------------
    ipopt_opts = {
        'max_iter': opts.max_iter,
        'tol': opts.tol,
        'acceptable_tol': opts.acceptable_tol,
        'print_level': opts.print_level,
        'mu_strategy': 'adaptive',
        'hessian_approximation': opts.hessian,
    }
    ipopt_opts.update(opts.extra)
    opti.solver('ipopt', {'print_time': True}, ipopt_opts)

    success = True
    iters = -1
    try:
        sol = opti.solve()
        get = sol.value
        iters = int(sol.stats().get('iter_count', -1))
    except RuntimeError as exc:
        log(f'[nlp] IPOPT did not converge: {exc}')
        log('[nlp] returning the last iterate -- verify before trusting it')
        success = False
        get = opti.debug.value

    def col(x):
        return np.asarray(get(x), dtype=np.float64).reshape(2, -1).T

    def row(x):
        return np.asarray(get(x), dtype=np.float64).reshape(-1)

    out = {
        'success': success,
        'iters': iters,
        'N': N, 'dt': dt,
        't': np.arange(N + 1) * dt,
        'obj_xy': col(PO), 'obj_yaw': row(THO),
        'base1_xy': col(P1), 'base1_yaw': row(PSI1),
        'base2_xy': col(P2), 'base2_yaw': row(PSI2),
        'ee1_xy': col(EE1), 'ee2_xy': col(EE2),
        'v1': row(V1), 'w1': row(W1), 'v2': row(V2), 'w2': row(W2),
        'base_disk_offsets': base_off, 'base_disk_radius': base_r,
        'obj_disk_offsets': obj_off, 'obj_disk_radius': obj_r,
        'ref_xy': ref_xy,
    }
    # ---- did the solution lean on the collision slack? -------------------
    # This is the difference between "the weights are not what I wanted" and
    # "it does not fit".  A plan whose slack is zero is a genuine preference
    # trade-off and weights will move it; a plan with slack > 0 is being told
    # to go somewhere it physically cannot, and no weight fixes that.
    sc = np.abs(np.asarray(get(Sc), dtype=np.float64).reshape(-1))
    if n_sub > 0:
        sc = np.concatenate([sc, np.abs(np.asarray(get(Sm),
                                                   dtype=np.float64).reshape(-1))])
    out['collision_slack_max'] = float(sc.max()) if sc.size else 0.0
    out['collision_slack_sum'] = float(sc.sum()) if sc.size else 0.0
    out['collision_slack_n'] = int((sc > 1e-4).sum())
    # ---- how well did the object follow the seed it was given? -----------
    dev = np.linalg.norm(out['obj_xy'] - ref_xy, axis=1)
    out['ref_dev_mean'] = float(dev.mean())
    out['ref_dev_max'] = float(dev.max())
    # the EE yaw the RL policy should present the gripper at: along the bar,
    # each end pointing back towards the object centre
    out['ee1_yaw'] = out['obj_yaw']
    out['ee2_yaw'] = out['obj_yaw'] + np.pi
    return out


# ---------------------------------------------------------------- verification
def verify(sdf, res: dict, geom: Geometry, lim: 'Limits | None' = None,
           n_sub: int = 4) -> dict:
    """Independent check against the raw grid, not the b-spline the NLP saw.

    The disk cover contains the rectangle, so clear disks imply a clear
    footprint -- this is conservative, not optimistic.

    Two things beyond the knots
    ---------------------------
    ``n_sub``   also samples **between** consecutive knots.  Clearance at the
                grid points says nothing about the segment joining them, and a
                plan that steps over a wall is worse than useless -- it looks
                fine in RViz right up to the crash.

    continuity  reports the largest per-step motion against what the limits
                allow.  A non-converged solve returns its last iterate, and that
                iterate routinely teleports the base metres per step; the
                dynamics residual catches it, but the raw number is what makes
                it obvious at a glance.
    """
    report = {}
    worst = np.inf

    for tag, xy, yaw, offs, r, margin in (
            ('base1', res['base1_xy'], res['base1_yaw'],
             res['base_disk_offsets'], res['base_disk_radius'], geom.base_margin),
            ('base2', res['base2_xy'], res['base2_yaw'],
             res['base_disk_offsets'], res['base_disk_radius'], geom.base_margin),
            ('object', res['obj_xy'], res['obj_yaw'],
             res['obj_disk_offsets'], res['obj_disk_radius'], geom.obj_margin)):
        u = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
        slack = np.inf
        at = -1
        infl = max(0.0, float(getattr(geom, 'map_inflation', 0.0)))
        for off in offs:
            d = sdf.value(xy + off * u) - r + infl
            i = int(np.argmin(d))
            if d[i] < slack:
                slack, at = float(d[i]), i
        report[tag] = {'min_clearance': slack, 'required_margin': margin, 'at_step': at}
        worst = min(worst, slack)

    for tag, ee, base in (('ee1', res['ee1_xy'], res['base1_xy']),
                          ('ee2', res['ee2_xy'], res['base2_xy'])):
        d = np.linalg.norm(ee - base, axis=1)
        report[tag + '_reach'] = {'min': float(d.min()), 'max': float(d.max()),
                                  'allowed': (geom.reach_min, geom.reach_max)}

    bar = np.linalg.norm(res['ee1_xy'] - res['ee2_xy'], axis=1)
    report['bar_length'] = {'min': float(bar.min()), 'max': float(bar.max()),
                            'nominal': geom.obj_len}
    # ---- between-knot clearance ----------------------------------------
    worst_seg, seg_at, seg_tag = np.inf, -1, ''
    if n_sub > 0:
        for tag, xy, yaw, offs, r in (
                ('base1', res['base1_xy'], res['base1_yaw'],
                 res['base_disk_offsets'], res['base_disk_radius']),
                ('base2', res['base2_xy'], res['base2_yaw'],
                 res['base_disk_offsets'], res['base_disk_radius']),
                ('object', res['obj_xy'], res['obj_yaw'],
                 res['obj_disk_offsets'], res['obj_disk_radius'])):
            for i in range(1, n_sub + 1):
                f = i / (n_sub + 1.0)
                xm = (1 - f) * xy[:-1] + f * xy[1:]
                # yaw is unwrapped out of the NLP, so a plain blend is right
                ym = (1 - f) * yaw[:-1] + f * yaw[1:]
                u = np.stack([np.cos(ym), np.sin(ym)], axis=1)
                for off in offs:
                    d = sdf.value(xm + off * u) - r + infl
                    j = int(np.argmin(d))
                    if d[j] < worst_seg:
                        worst_seg, seg_at, seg_tag = float(d[j]), j, tag
    report['between_knots'] = {'min_clearance': float(worst_seg),
                               'at_step': seg_at, 'worst_of': seg_tag,
                               'samples_per_gap': n_sub}
    worst = min(worst, worst_seg)

    # ---- continuity ------------------------------------------------------
    dt_ = res['dt']
    cont = {}
    for tag, xy, yaw in (('base1', res['base1_xy'], res['base1_yaw']),
                         ('base2', res['base2_xy'], res['base2_yaw']),
                         ('object', res['obj_xy'], res['obj_yaw'])):
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dyaw = np.abs(np.diff(yaw))
        cont[tag] = {'max_step_m': float(step.max()),
                     'max_dyaw_rad': float(dyaw.max()),
                     'at_step': int(np.argmax(step))}
    if lim is not None:
        allow_base = max(abs(lim.v_max), abs(lim.v_min)) * dt_
        allow_obj = lim.obj_v_max * dt_
        cont['allowed_step_m'] = {'base': float(allow_base), 'object': float(allow_obj)}
        # 1.5x leaves room for the solver tolerance without hiding a teleport
        cont['continuous'] = bool(
            cont['base1']['max_step_m'] <= 1.5 * allow_base
            and cont['base2']['max_step_m'] <= 1.5 * allow_base
            and cont['object']['max_step_m'] <= 1.5 * allow_obj)
    else:
        cont['continuous'] = True
    report['continuity'] = cont

    report['collision_free'] = bool(worst >= 0.0)
    report['worst_clearance'] = float(worst)

    # Dynamics residual.  A non-converged solve still returns its last iterate,
    # and that iterate can be collision free while violating the unicycle
    # dynamics outright -- i.e. a base trajectory no robot can actually drive.
    # Collision clearance alone is not evidence the plan is executable.
    dt = res['dt']
    worst_dyn = 0.0
    for tag, xy, yaw, v, w in (('base1', res['base1_xy'], res['base1_yaw'],
                                res['v1'], res['w1']),
                               ('base2', res['base2_xy'], res['base2_yaw'],
                                res['v2'], res['w2'])):
        v_mid = 0.5 * (v[:-1] + v[1:])
        psi_mid = yaw[:-1] + 0.5 * dt * w[:-1]
        pred = xy[:-1] + dt * (v_mid[:, None]
                               * np.stack([np.cos(psi_mid), np.sin(psi_mid)], axis=1))
        e_pos = float(np.abs(xy[1:] - pred).max())
        e_yaw = float(np.abs(yaw[1:] - yaw[:-1] - dt * 0.5 * (w[:-1] + w[1:])).max())
        report[tag + '_dynamics'] = {'max_pos_residual': e_pos,
                                     'max_yaw_residual': e_yaw}
        worst_dyn = max(worst_dyn, e_pos, e_yaw)
    report['worst_dynamics_residual'] = worst_dyn
    # IPOPT's own constraint tolerance is 1e-4, so a converged solve lands right
    # around there; gate an order of magnitude above it.  1e-3 m per step is
    # still twenty times smaller than the collision margin.
    report['dynamically_feasible'] = bool(worst_dyn < 1e-3)
    report['executable'] = bool(report['collision_free']
                                and report['dynamically_feasible']
                                and report['continuity']['continuous'])
    return report
