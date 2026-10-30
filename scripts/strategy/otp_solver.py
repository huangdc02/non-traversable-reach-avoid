"""Proved 1D1A formulas, fixed-lift 2D1A equations and circle path geometry."""

from __future__ import annotations

from math import acos, asin, atan2, cos, pi, sin, sqrt
from typing import Any
import numpy as np

from geometry_common import lifted_distance

ROOT_TOL = 1e-8
CONSTRAINT_TOL = 1e-7
TAU = 2.0 * pi


def _point(value: Any, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite two-dimensional point")
    return point.copy()


def _validate_endpoint(x: np.ndarray, R: float, name: str) -> float:
    radius = float(np.linalg.norm(x))
    if radius < R - 1e-12 * max(1.0, R):
        raise ValueError(f"{name} lies inside the circular goal")
    return radius


def _segment_clearance(a: np.ndarray, b: np.ndarray) -> float:
    direction = b - a
    denominator = float(direction @ direction)
    if denominator == 0.0:
        return float(np.linalg.norm(a))
    fraction = float(np.clip(-(a @ direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(a + fraction * direction))


def _is_directly_visible_raw(start: np.ndarray, target: np.ndarray, R: float) -> bool:
    scale = max(1.0, R, np.linalg.norm(start), np.linalg.norm(target))
    return _segment_clearance(start, target) >= R - 1e-12 * scale


def is_directly_visible(x_start: Any, x_target: Any, R: float) -> bool:
    """Return whether the segment avoids the open circular goal."""

    start, target = _point(x_start, "x_start"), _point(x_target, "x_target")
    _validate_endpoint(start, R, "x_start")
    _validate_endpoint(target, R, "x_target")
    return _is_directly_visible_raw(start, target, R)


def central_lift(point: np.ndarray, attacker: np.ndarray) -> tuple[float, float]:
    """Lift a physical target to the closed central angular cell of A."""
    angle_a = atan2(attacker[1], attacker[0])
    angle = atan2(point[1], point[0])
    return float(np.linalg.norm(point)), angle_a + atan2(
        sin(angle - angle_a), cos(angle - angle_a)
    )


def defender_lift(
    defender: np.ndarray, attacker: np.ndarray, side: int
) -> tuple[float, float]:
    """side=-1: lower lift; side=+1: upper lift (not a travel direction).

    At exact angular alignment the coincident copy is lower and the next
    copy is upper.  This half-open convention retains the zero-winding path.
    """
    if side not in (-1, 1):
        raise ValueError("lift side must be -1 or +1")
    angle_a = atan2(attacker[1], attacker[0])
    angle_d = atan2(defender[1], defender[0])
    lower = angle_d + TAU * np.floor((angle_a - angle_d) / TAU)
    return float(np.linalg.norm(defender)), float(lower + (TAU if side == 1 else 0.0))


def lifted_q(
    target_lift, attacker, defender, R: float, alpha: float, side: int
) -> float:
    """Universal-cover q on the central cell, with a fixed defender lift."""
    radius, angle = target_lift
    source_a = (float(np.linalg.norm(attacker)), atan2(attacker[1], attacker[0]))
    return float(
        lifted_distance(source_a, radius, angle, R)
        - alpha
        * lifted_distance(defender_lift(defender, attacker, side), radius, angle, R)
    )


def _parameters(
    params: dict[str, Any], defender_speed: float | None = None
) -> tuple[float, float, float, float, float]:
    R = float(params["R"])
    v_A = float(params["v_A"])
    v_D = float(params["v_D"] if defender_speed is None else defender_speed)
    root_tol = float(params.get("root_tol", ROOT_TOL))
    constraint_tol = float(params.get("constraint_tol", CONSTRAINT_TOL))
    if (
        not np.isfinite([R, v_A, v_D, root_tol, constraint_tol]).all()
        or R <= 0.0
        or v_A <= 0.0
        or v_D <= 0.0
        or v_A > v_D * (1.0 + 1e-12)
        or root_tol <= 0.0
        or constraint_tol <= 0.0
    ):
        raise ValueError("require R>0, 0<v_A<=v_D, and positive tolerances")
    return R, v_A, v_D, root_tol, constraint_tol


def _bisect_root(function, left: float, right: float, tolerance: float) -> float:
    """Bisect one sign-changing scalar interval."""

    f_left = function(left)
    for _ in range(100):
        middle = 0.5 * (left + right)
        f_middle = function(middle)
        if abs(f_middle) <= 1e-12 or right - left <= tolerance:
            return middle
        if f_left * f_middle <= 0.0:
            right = middle
        else:
            left, f_left = middle, f_middle
    return 0.5 * (left + right)


def _analytic_direct_target(
    x_A: np.ndarray,
    x_D: np.ndarray,
    alpha: float,
    root_tol: float,
) -> np.ndarray | None:
    """Return the unconstrained analytic direct 1D1A target."""

    denominator = 1.0 - alpha * alpha
    if denominator <= 32.0 * np.finfo(float).eps:
        return None

    center = (x_A - alpha * alpha * x_D) / denominator
    center_norm = float(np.linalg.norm(center))
    if center_norm <= root_tol:
        return None
    circle_radius = alpha * float(np.linalg.norm(x_A - x_D)) / denominator
    return (center_norm - circle_radius) * center / center_norm


def _lift_is_direct(
    x_A: np.ndarray,
    x_D: np.ndarray,
    target: np.ndarray,
    side: int,
    R: float,
    tolerance: float,
) -> bool:
    """Return whether the specified defender lift reaches target directly."""

    target_lift = central_lift(target, x_A)
    source_lift = defender_lift(x_D, x_A, side)
    lifted = float(lifted_distance(source_lift, *target_lift, R))
    straight = float(np.linalg.norm(target - x_D))
    scale = max(1.0, R, lifted, straight)
    return abs(lifted - straight) <= tolerance * scale


def _equation_12_candidate(
    x_A: np.ndarray,
    x_D: np.ndarray,
    R: float,
    alpha: float,
    sigma: int,
    root_tol: float,
    constraint_tol: float,
) -> dict[str, Any] | None:
    """Find the fixed-lift 1D1A root on the signed attacker ray.

    The signed angle from AO to AP is positive for the positive lift and
    negative for the negative lift.  The circle point P is retained only as a
    diagnostic/root-search seed when it is strictly on the attacker-winning
    side; it is not a fallback candidate.
    """

    attacker_radius = float(np.linalg.norm(x_A))
    offset = asin(float(np.clip(alpha * R / attacker_radius, -1.0, 1.0)))
    psi_A = sigma * offset
    ray_angle = atan2(x_A[1], x_A[0]) + pi + psi_A
    direction = np.array([cos(ray_angle), sin(ray_angle)])

    projection = float(x_A @ direction)
    discriminant = projection * projection - (attacker_radius**2 - R**2)
    if discriminant < -constraint_tol:
        return None
    root = sqrt(max(0.0, discriminant))
    intersections = sorted(
        parameter
        for parameter in (-projection - root, -projection + root)
        if parameter >= -root_tol
    )
    if not intersections:
        return None
    goal_parameter = max(0.0, intersections[0])

    def q_on_ray(parameter: float) -> float:
        point = x_A + parameter * direction
        return lifted_q(central_lift(point, x_A), x_A, x_D, R, alpha, sigma)

    q_start, q_goal = q_on_ray(0.0), q_on_ray(goal_parameter)
    boundary = x_A + goal_parameter * direction
    boundary *= R / float(np.linalg.norm(boundary))
    if q_goal < -constraint_tol:
        return {
            "x": boundary.copy(),
            "mode": "goal_boundary",
            "pair_discard_probe": boundary.copy(),
            "pair_discard_reason": "goal_boundary_nonpositive",
        }

    if abs(q_goal) <= constraint_tol:
        parameter = goal_parameter
    elif q_start * q_goal < 0.0:
        parameter = _bisect_root(q_on_ray, 0.0, goal_parameter, root_tol)
    else:
        return {
            "x": boundary.copy(),
            "mode": "goal_boundary",
        }
    target = x_A + parameter * direction

    return {
        "x": target.copy(),
        "mode": "indirect_interception",
    }


def solve_1d1a_otp(
    x_A: Any,
    x_D: Any,
    sigma: int,
    params: dict[str, Any],
    *,
    defender_speed: float | None = None,
) -> dict[str, Any]:
    """Solve one fixed-lift 1D1A candidate at the supplied defender speed."""

    if sigma not in (-1, +1):
        raise ValueError("sigma must be +1 or -1")
    attacker, defender = _point(x_A, "x_A"), _point(x_D, "x_D")
    R, v_A, v_D, root_tol, constraint_tol = _parameters(params, defender_speed)
    _validate_endpoint(attacker, R, "x_A")
    _validate_endpoint(defender, R, "x_D")
    alpha = v_A / v_D
    direct_target = _analytic_direct_target(attacker, defender, alpha, root_tol)
    candidate = None
    pair_discard_probe = None
    pair_discard_reason = None
    if direct_target is not None:
        scale = max(1.0, R, float(np.linalg.norm(direct_target)))
        if np.linalg.norm(direct_target) < R - constraint_tol * scale:
            pair_discard_probe = direct_target.copy()
            pair_discard_reason = "direct_target_inside"
        else:
            if _lift_is_direct(
                attacker, defender, direct_target, sigma, R, constraint_tol
            ):
                candidate = {
                    "x": direct_target.copy(),
                    "mode": "direct_interception",
                }
            else:
                candidate = _equation_12_candidate(
                    attacker,
                    defender,
                    R,
                    alpha,
                    sigma,
                    root_tol,
                    constraint_tol,
                )
                if candidate is not None:
                    pair_discard_probe = candidate.get("pair_discard_probe")
                    pair_discard_reason = candidate.get("pair_discard_reason")
    boundary = (
        candidate["x"].copy()
        if candidate is not None and candidate["mode"] == "goal_boundary"
        else None
    )
    q = (
        lifted_q(
            central_lift(candidate["x"], attacker),
            attacker,
            defender,
            R,
            alpha,
            sigma,
        )
        if candidate is not None
        else float("nan")
    )
    valid = (
        candidate is not None
        and candidate["mode"] != "goal_boundary"
        and abs(q) <= constraint_tol
    )
    if not valid:
        return {
            "success": False,
            "otp": None,
            "value": float("nan"),
            "sigma": sigma,
            "q": q,
            "boundary": boundary,
            "mode": None,
            "pair_discard_probe": pair_discard_probe,
            "pair_discard_reason": pair_discard_reason,
        }

    target = candidate["x"]
    return {
        "success": True,
        "otp": target.copy(),
        "sigma": sigma,
        "value": float(np.linalg.norm(target) - R),
        "q": q,
        "boundary": None,
        "mode": candidate["mode"],
    }


def heading_to_point(
    position: Any, target: Any, tolerance: float = CONSTRAINT_TOL
) -> float:
    """Return the straight-line heading to a target, or zero at the target."""

    displacement = _point(target, "target") - _point(position, "position")
    if np.linalg.norm(displacement) <= tolerance:
        return 0.0
    return float(atan2(displacement[1], displacement[0]))


def _dsp_geometry(
    position: Any, target: Any, R: float, sigma: int, lift_angles=None
) -> dict[str, Any]:
    defender, target = _point(position, "x_D"), _point(target, "x_f")
    if sigma not in (-1, +1):
        raise ValueError("sigma must be +1 or -1")
    direct_length = float(np.linalg.norm(target - defender))
    direct = is_directly_visible(defender, target, R)
    if lift_angles is not None:
        source_angle, target_angle = lift_angles
        delta_lift = target_angle - source_angle
        beta = acos(float(np.clip(R / np.linalg.norm(defender), -1.0, 1.0))) + acos(
            float(np.clip(R / np.linalg.norm(target), -1.0, 1.0))
        )
        direct = abs(delta_lift) <= beta + 1e-12
        sigma = 1 if delta_lift >= 0 else -1
    if direct:
        return {"direct": True, "length": direct_length, "target": target}

    tau = 2.0 * pi
    r_D, r_f = float(np.linalg.norm(defender)), float(np.linalg.norm(target))
    theta_D, theta_f = atan2(defender[1], defender[0]), atan2(target[1], target[0])
    gamma_D = acos(float(np.clip(R / r_D, -1.0, 1.0)))
    gamma_f = acos(float(np.clip(R / r_f, -1.0, 1.0)))
    delta = (theta_f - theta_D) % tau
    if lift_angles is not None:
        theta_D, theta_f = lift_angles
        delta = abs(theta_f - theta_D) if sigma == 1 else tau - abs(theta_f - theta_D)
    if sigma == +1:
        start_angle, end_angle = theta_D + gamma_D, theta_f - gamma_f
        arc_angle = delta - gamma_D - gamma_f
    else:
        start_angle, end_angle = theta_D - gamma_D, theta_f + gamma_f
        arc_angle = tau - delta - gamma_D - gamma_f
    arc_angle = max(0.0, arc_angle)
    start_tangent = R * np.array([cos(start_angle), sin(start_angle)])
    end_tangent = R * np.array([cos(end_angle), sin(end_angle)])
    return {
        "direct": False,
        "target": target,
        "start_angle": start_angle,
        "start_tangent": start_tangent,
        "end_tangent": end_tangent,
        "lengths": (
            sqrt(max(0.0, r_D * r_D - R * R)),
            R * arc_angle,
            sqrt(max(0.0, r_f * r_f - R * R)),
        ),
    }


def dsp_heading(
    position: Any,
    target: Any,
    R: float,
    sigma: int,
    tolerance: float = CONSTRAINT_TOL,
    *,
    lift_angles=None,
) -> float:
    """Return the first tangent/arc/straight heading of a directional DSP."""

    defender, target = _point(position, "x_D"), _point(target, "x_f")
    if np.linalg.norm(target - defender) <= tolerance and (
        lift_angles is None or abs(lift_angles[1] - lift_angles[0]) <= tolerance
    ):
        return 0.0
    geometry = _dsp_geometry(defender, target, R, sigma, lift_angles)
    if lift_angles is not None:
        sigma = 1 if lift_angles[1] >= lift_angles[0] else -1
    if geometry["direct"]:
        return heading_to_point(defender, target, tolerance)
    first, arc, last = geometry["lengths"]
    if first > tolerance:
        return heading_to_point(defender, geometry["start_tangent"], tolerance)
    if arc <= tolerance and last > tolerance:
        return heading_to_point(geometry["end_tangent"], target, tolerance)
    tangent_direction = sigma * np.array(
        [-sin(geometry["start_angle"]), cos(geometry["start_angle"])]
    )
    return float(atan2(tangent_direction[1], tangent_direction[0]))


def advance_dsp(
    position: Any,
    target: Any,
    R: float,
    sigma: int,
    distance: float,
    *,
    lift_angles=None,
) -> np.ndarray:
    """Advance exactly along the current straight/tangent/arc DSP pieces."""

    defender = _point(position, "x_D")
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be finite and nonnegative")
    geometry = _dsp_geometry(defender, target, R, sigma, lift_angles)
    target = geometry["target"]
    if lift_angles is not None:
        sigma = 1 if lift_angles[1] >= lift_angles[0] else -1
    if geometry["direct"]:
        if geometry["length"] <= distance or geometry["length"] == 0.0:
            return target.copy()
        return defender + distance * (target - defender) / geometry["length"]

    first, arc, last = geometry["lengths"]
    if distance < first:
        return defender + (distance / first) * (geometry["start_tangent"] - defender)
    distance -= first
    if distance < arc:
        angle = geometry["start_angle"] + sigma * distance / R
        return R * np.array([cos(angle), sin(angle)])
    distance -= arc
    if distance < last:
        return geometry["end_tangent"] + (distance / last) * (
            target - geometry["end_tangent"]
        )
    return target.copy()


def _validate_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    missing = {"A", "D1", "D2"} - set(state)
    if missing:
        raise KeyError(f"state is missing {sorted(missing)}")
    output = {}
    for player in ("A", "D1", "D2"):
        point = np.asarray(state[player], dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"state[{player!r}] must be a finite 2-vector")
        output[player] = point.copy()
    return output


def _parameters_2d1a(
    params: dict[str, Any]
) -> tuple[float, float, tuple[float, float], float, float, int]:
    R = float(params["R"])
    v_A = float(params["v_A"])
    v_D1 = float(params["v_D1"])
    v_D2 = float(params["v_D2"])
    root_tol = float(params.get("root_tol", 1e-8))
    constraint_tol = float(params.get("constraint_tol", CONSTRAINT_TOL))
    newton_iterations = int(params.get("cooperative_newton_iterations", 50))
    if (
        not np.isfinite([R, v_A, v_D1, v_D2, root_tol, constraint_tol]).all()
        or R <= 0.0
        or v_A <= 0.0
        or min(v_D1, v_D2) <= 0.0
        or v_A > min(v_D1, v_D2) * (1.0 + 1e-12)
        or root_tol <= 0.0
        or constraint_tol <= 0.0
        or newton_iterations < 1
    ):
        raise ValueError("invalid 2D1A parameters")
    return (
        R,
        v_A,
        (v_D1, v_D2),
        root_tol,
        constraint_tol,
        newton_iterations,
    )


def solve_2d1a_otp(
    X: dict[str, Any],
    lift_pair: tuple[int, int],
    params: dict[str, Any],
    initial_guess: Any = None,
) -> dict[str, Any]:
    """Find q1=q2=0 intersections for a lower/upper defender-lift pair.

    A small polar mesh brackets simultaneous zero contours; damped Newton
    refines them with exact lifted-distance derivatives.  Additional local
    residual minima and individual OTPs seed tangent intersections.
    """
    pair = tuple(lift_pair)
    if pair not in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        raise ValueError("each lift-pair entry must be +1 or -1")
    state = _validate_state(X)
    R, v_A, defender_speeds, root_tol, tol, iterations = _parameters_2d1a(params)
    for name, point in state.items():
        _validate_endpoint(point, R, name)
    alphas = np.asarray([v_A / speed for speed in defender_speeds])
    a = (float(np.linalg.norm(state["A"])), atan2(state["A"][1], state["A"][0]))
    defenders = [
        defender_lift(state[d], state["A"], pair[i]) for i, d in enumerate(("D1", "D2"))
    ]
    lo, hi = np.array([R, a[1] - pi]), np.array([max(R, a[0]), a[1] + pi])

    geometry = [
        (r, theta, acos(min(1.0, R / r)), sqrt(max(0.0, r * r - R * R)))
        for r, theta in [a] + defenders
    ]

    def evaluate(point, gradients=False):
        r, theta = point
        beta = acos(min(1.0, R / r))
        tangent = sqrt(max(0.0, r * r - R * R))
        distances, derivatives = [], []
        for r0, theta0, beta0, tangent0 in geometry:
            delta = theta - theta0
            if abs(delta) <= beta + beta0:
                length = sqrt(
                    max(0.0, (r - r0) ** 2 + 4 * r * r0 * sin(delta / 2) ** 2)
                )
                distances.append(length)
                if gradients:
                    derivatives.append(
                        ((r - r0 * cos(delta)) / length, r * r0 * sin(delta) / length)
                        if length > 1e-14
                        else (0.0, 0.0)
                    )
            else:
                distances.append(tangent0 + tangent + R * (abs(delta) - beta0 - beta))
                if gradients:
                    derivatives.append((tangent / r, R if delta > 0 else -R))
        if gradients:
            g = np.asarray(derivatives)
            return g[0] - alphas[:, None] * g[1:]
        return np.array(
            [
                distances[0] - alphas[index] * distance
                for index, distance in enumerate(distances[1:])
            ]
        )

    nr = max(3, int(params.get("cooperative_radial_samples", 13)))
    nt = max(9, int(params.get("cooperative_angular_samples", 49)))
    radii = np.linspace(lo[0], hi[0], nr)
    angles = np.linspace(lo[1], hi[1], nt)
    rr, tt = np.meshgrid(radii, angles, indexing="ij")
    da = lifted_distance(a, rr, tt, R)
    fields = np.array(
        [
            da - alphas[index] * lifted_distance(defender, rr, tt, R)
            for index, defender in enumerate(defenders)
        ]
    )
    corners = np.stack(
        (fields[:, :-1, :-1], fields[:, 1:, :-1], fields[:, :-1, 1:], fields[:, 1:, 1:])
    )
    cells = np.all((corners.min(axis=0) <= tol) & (corners.max(axis=0) >= -tol), axis=0)
    seeds = [
        np.array([(radii[i] + radii[i + 1]) / 2, (angles[j] + angles[j + 1]) / 2])
        for i, j in np.argwhere(cells)
    ]
    residual = np.max(np.abs(fields), axis=0)
    # Seed all sampled local minima, including the physical and cell boundaries.
    padded = np.pad(residual, 1, constant_values=np.inf)
    minima = np.ones_like(residual, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                minima &= residual <= padded[1 + di : 1 + di + nr, 1 + dj : 1 + dj + nt]
    seeds.extend(np.array([rr[i, j], tt[i, j]]) for i, j in np.argwhere(minima))
    for defender_index, d in enumerate(("D1", "D2")):
        for direction in (-1, 1):
            result = solve_1d1a_otp(
                state["A"],
                state[d],
                direction,
                params,
                defender_speed=defender_speeds[defender_index],
            )
            point = result["otp"] if result["success"] else result["boundary"]
            if point is not None:
                seeds.append(np.array(central_lift(point, state["A"])))
    if initial_guess is not None:
        seeds.insert(
            0,
            np.array(central_lift(_point(initial_guess, "initial_guess"), state["A"])),
        )

    roots = []
    for seed in seeds:
        point = np.clip(seed, lo, hi)
        for _ in range(iterations):
            f = evaluate(point)
            error = np.linalg.norm(f, ord=np.inf)
            if error <= tol:
                break
            jac = evaluate(point, gradients=True)
            try:
                step = np.linalg.solve(jac, -f)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jac, -f, rcond=None)[0]
            if not np.isfinite(step).all():
                break
            accepted = False
            for power in range(16):
                trial = np.clip(point + step * 2.0 ** (-power), lo, hi)
                if np.linalg.norm(evaluate(trial), ord=np.inf) < error:
                    point, accepted = trial, True
                    break
            if not accepted:
                break
        q = evaluate(point)
        if np.linalg.norm(q, ord=np.inf) > tol:
            continue
        if any(
            np.linalg.norm(point - rec["otp_lift"]) <= 10 * root_tol for rec in roots
        ):
            continue
        r, theta = point
        roots.append(
            {
                "otp": r * np.array([cos(theta), sin(theta)]),
                "otp_lift": point.copy(),
                "value": float(r - R),
                "q_values": tuple(map(float, q)),
            }
        )
    if not roots:
        return {
            "success": False,
            "otp": None,
            "value": float("nan"),
            "lift_pair": pair,
            "roots": (),
        }
    best = min(roots, key=lambda item: item["value"])
    return dict(best, success=True, lift_pair=pair, roots=tuple(roots))
