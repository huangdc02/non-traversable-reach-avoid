"""Analytic radial barrier classification and interactive attacker projection."""

from __future__ import annotations

from math import acos, atan2, cos, pi, sin, sqrt
from typing import Any
import numpy as np

from geometry_common import (
    MODES,
    boundary_derivative,
    compute_d_max,
    geometry,
    lifted_distance,
    solve_p,
)

TAU = 2.0 * pi
_BISECTION_STEPS = 80


def _point(value: Any, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite two-dimensional point")
    return point.copy()


def _validate_inputs(
    D1: Any,
    D2: Any,
    v_D1: float,
    v_D2: float,
    A: Any,
    v_A: float,
    R: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float]:
    defender1 = _point(D1, "D1")
    defender2 = _point(D2, "D2")
    attacker = _point(A, "A")
    speed1, speed2 = float(v_D1), float(v_D2)
    attacker_speed, radius, tolerance = float(v_A), float(R), float(tolerance)
    scalars = np.array([speed1, speed2, attacker_speed, radius, tolerance])
    if not np.all(np.isfinite(scalars)):
        raise ValueError("speeds, R, and tolerance must be finite")
    if min(speed1, speed2, attacker_speed, radius, tolerance) <= 0.0:
        raise ValueError("speeds, R, and tolerance must be positive")
    if attacker_speed > min(speed1, speed2) * (1.0 + tolerance):
        raise ValueError("the classifier requires v_A <= min(v_D1, v_D2)")
    for name, point in (("D1", defender1), ("D2", defender2)):
        if np.linalg.norm(point) < radius - tolerance:
            raise ValueError(f"{name} lies inside the target disk")
    return (
        defender1,
        defender2,
        attacker,
        speed1,
        speed2,
        attacker_speed,
        radius,
        tolerance,
    )


def _canonical_barrier_point(
    defender_radius: float,
    radius: float,
    alpha: float,
    target_angle: float,
) -> np.ndarray:
    """Evaluate the canonical 1D1A envelope at one generating angle."""

    tangent_length = sqrt(max(0.0, defender_radius**2 - radius**2))
    tangent_angle = acos(float(np.clip(radius / defender_radius, -1.0, 1.0)))
    source = (defender_radius, 0.5 * pi)
    distance = float(lifted_distance(source, radius, target_angle, radius))
    derivative = boundary_derivative(
        (defender_radius, 0.5 * pi, tangent_angle, tangent_length),
        target_angle,
        -1,
        radius,
    )

    radial = np.array([cos(target_angle), sin(target_angle)])
    tangent = np.array([-sin(target_angle), cos(target_angle)])
    eta = float(np.clip(alpha * derivative / radius, -1.0, 1.0))
    outward = sqrt(max(0.0, 1.0 - eta * eta)) * radial - eta * tangent
    return radius * radial + alpha * distance * outward


def _individual_barrier_radius(
    defender: np.ndarray,
    direction: np.ndarray,
    radius: float,
    alpha: float,
) -> float:
    """Return the single-defender barrier radius on one target-centred ray."""

    defender_radius = float(np.linalg.norm(defender))
    defender_angle = atan2(defender[1], defender[0])
    query_angle = (atan2(direction[1], direction[0]) + 0.5 * pi - defender_angle) % TAU
    if not 0.5 * pi <= query_angle <= 1.5 * pi:
        query_angle = (pi - query_angle) % TAU
    query_angle = float(np.clip(query_angle, 0.5 * pi, 1.5 * pi))

    tangent_angle = acos(float(np.clip(radius / defender_radius, -1.0, 1.0)))
    tangent_length = sqrt(max(0.0, defender_radius**2 - radius**2))
    start = 0.5 * pi if alpha < 1.0 else 0.5 * pi + tangent_angle
    maximum_distance = compute_d_max(defender, alpha, radius)
    end = 0.5 * pi + tangent_angle + (maximum_distance - tangent_length) / radius

    def polar_angle(generating_angle: float) -> float:
        point = _canonical_barrier_point(
            defender_radius, radius, alpha, generating_angle
        )
        angle = atan2(point[1], point[0])
        return angle if angle >= 0.5 * pi else angle + TAU

    lo, hi = start, end
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (lo + hi)
        if polar_angle(middle) < query_angle:
            lo = middle
        else:
            hi = middle
    point = _canonical_barrier_point(defender_radius, radius, alpha, 0.5 * (lo + hi))
    return float(np.linalg.norm(point))


def _tangency_point(
    defender: np.ndarray,
    target: np.ndarray,
    sigma: int,
    radius: float,
    alpha: float,
    path_distance: float,
) -> np.ndarray:
    """Return the analytic p-circle/1D1A-barrier tangency."""

    target_angle = atan2(target[1], target[0]) % TAU
    defender_geometry = geometry(defender, radius)
    tangent = np.array([-sin(target_angle), cos(target_angle)])
    derivative = boundary_derivative(defender_geometry, target_angle, sigma, radius)
    tangential = float(np.clip(-alpha * derivative / radius, -1.0, 1.0))
    radial = sqrt(max(0.0, 1.0 - tangential * tangential))
    outward = radial * target / radius + tangential * tangent
    return target + alpha * float(path_distance) * outward


def _on_ccw_arc(angle: float, start: float, end: float, tolerance: float) -> bool:
    return (angle - start) % TAU <= (end - start) % TAU + tolerance


def _cooperative_arc_radius(
    direction: np.ndarray,
    target: np.ndarray,
    circle_radius: float,
    tangent1: np.ndarray,
    tangent2: np.ndarray,
    target_radius: float,
    tolerance: float,
) -> float:
    """Return the ray intersection with the outward p-circle arc, if any."""

    scale = max(1.0, target_radius, circle_radius)
    if np.linalg.norm(tangent1 - tangent2) <= tolerance * scale:
        return float("inf")

    start = atan2((tangent1 - target)[1], (tangent1 - target)[0]) % TAU
    end = atan2((tangent2 - target)[1], (tangent2 - target)[0]) % TAU
    # The active cooperative boundary is the exterior short arc between the
    # two analytic tangencies. The complementary long arc passes through the
    # covered side of the candidate circle.
    active_ccw = (end - start) % TAU <= pi

    projection = float(target @ direction)
    discriminant = projection**2 + circle_radius**2 - target_radius**2
    if discriminant < -tolerance * scale:
        return float("inf")
    root = sqrt(max(0.0, discriminant))
    radii = []
    for ray_radius in (projection - root, projection + root):
        if ray_radius < target_radius - tolerance * scale:
            continue
        point = ray_radius * direction
        relative_angle = atan2((point - target)[1], (point - target)[0]) % TAU
        on_arc = (
            _on_ccw_arc(relative_angle, start, end, tolerance)
            if active_ccw
            else _on_ccw_arc(relative_angle, end, start, tolerance)
        )
        if on_arc:
            radii.append(max(target_radius, float(ray_radius)))
    return min(radii, default=float("inf"))


def _barrier_radius(
    D1: np.ndarray,
    D2: np.ndarray,
    v_D1: float,
    v_D2: float,
    A: np.ndarray,
    v_A: float,
    R: float,
    tolerance: float,
) -> tuple[float, str, dict[str, Any]]:
    attacker_radius = float(np.linalg.norm(A))
    direction = A / attacker_radius
    alpha1, alpha2 = v_A / v_D1, v_A / v_D2

    candidates: list[tuple[float, str, dict[str, Any]]] = [
        (
            _individual_barrier_radius(D1, direction, R, alpha1),
            "D1",
            {"type": "individual", "defender": "D1"},
        ),
        (
            _individual_barrier_radius(D2, direction, R, alpha2),
            "D2",
            {"type": "individual", "defender": "D2"},
        ),
    ]

    solution = solve_p(D1, D2, radius=R, alpha=alpha1, beta=v_D2 / v_D1)
    for item in solution["candidates"]:
        if not item["distance_feasible"]:
            continue
        target = np.asarray(item["p"], dtype=float)
        sigma = tuple(int(value) for value in MODES[item["mode_index"]])
        tangent1 = _tangency_point(D1, target, sigma[0], R, alpha1, float(item["d1"]))
        tangent2 = _tangency_point(D2, target, sigma[1], R, alpha2, float(item["d2"]))
        cooperative_radius = alpha1 * float(item["arrival_time"])
        ray_radius = _cooperative_arc_radius(
            direction,
            target,
            cooperative_radius,
            tangent1,
            tangent2,
            R,
            tolerance,
        )
        if np.isfinite(ray_radius):
            candidates.append(
                (
                    ray_radius,
                    f"cooperative_{sigma[0]:+d}_{sigma[1]:+d}",
                    {
                        "type": "cooperative",
                        "sigma": sigma,
                        "target": target.copy(),
                        "circle_radius": cooperative_radius,
                    },
                )
            )

    barrier_radius, source, details = min(candidates, key=lambda item: item[0])
    return float(barrier_radius), source, details


def classify_game_outcome(
    D1: Any,
    D2: Any,
    v_D1: float,
    v_D2: float,
    A: Any,
    v_A: float,
    R: float,
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Classify a circular 2D1A state under equilibrium play.

    Region is inside, outside, or barrier. Inside is the attacker-winning side;
    outside is the defender-winning side. The signed radial margin is
    barrier_radius minus norm(A), hence positive inside.

    Work is bounded by two scalar 1D1A evaluations and the four fixed lift
    modes in circle_obs.solve_p. It is independent of barrier resolution.
    """

    (
        defender1,
        defender2,
        attacker,
        speed1,
        speed2,
        attacker_speed,
        radius,
        tolerance,
    ) = _validate_inputs(D1, D2, v_D1, v_D2, A, v_A, R, tolerance)
    attacker_radius = float(np.linalg.norm(attacker))
    if attacker_radius <= radius + tolerance:
        return {
            "region": "inside",
            "winner": "attacker",
            "on_barrier": False,
            "signed_margin": float(radius - attacker_radius),
            "attacker_radius": attacker_radius,
            "barrier_radius": radius,
            "active_source": "goal",
            "active_candidate": {"type": "goal"},
        }

    barrier_radius, source, details = _barrier_radius(
        defender1,
        defender2,
        speed1,
        speed2,
        attacker,
        attacker_speed,
        radius,
        tolerance,
    )
    signed_margin = barrier_radius - attacker_radius
    scale = max(1.0, radius, barrier_radius, attacker_radius)
    on_barrier = abs(signed_margin) <= tolerance * scale
    if on_barrier:
        region, winner = "barrier", "simultaneous"
    elif signed_margin > 0.0:
        region, winner = "inside", "attacker"
    else:
        region, winner = "outside", "defenders"
    return {
        "region": region,
        "winner": winner,
        "on_barrier": on_barrier,
        "signed_margin": float(signed_margin),
        "attacker_radius": attacker_radius,
        "barrier_radius": barrier_radius,
        "active_source": source,
        "active_candidate": details,
    }


def classify_state(
    state: dict[str, Any],
    params: dict[str, Any],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Dictionary wrapper compatible with the strategy_2D1A state format."""

    return classify_game_outcome(
        state["D1"],
        state["D2"],
        params["v_D1"],
        params["v_D2"],
        state["A"],
        params["v_A"],
        params["R"],
        tolerance=tolerance,
    )


def project_attacker_outside_barrier(
    D1: Any,
    D2: Any,
    v_D1: float,
    v_D2: float,
    A: Any,
    v_A: float,
    R: float,
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Project an interactive attacker radially onto the admissible side.

    The defender-winning side is exterior to the closest radial barrier.
    Therefore an attacker requested inside that barrier is moved outward to
    the barrier on the same target-centred ray.  The origin uses +x as its
    deterministic ray because it has no polar direction.
    """

    attacker = _point(A, "A")
    attacker_radius = float(np.linalg.norm(attacker))
    direction = (
        attacker / attacker_radius
        if attacker_radius > tolerance
        else np.array([1.0, 0.0])
    )
    probe_radius = max(2.0 * float(R), float(R) + 1.0)
    (
        defender1,
        defender2,
        _,
        speed1,
        speed2,
        attacker_speed,
        radius,
        tolerance,
    ) = _validate_inputs(
        D1,
        D2,
        v_D1,
        v_D2,
        direction * probe_radius,
        v_A,
        R,
        tolerance,
    )
    barrier_radius, source, details = _barrier_radius(
        defender1,
        defender2,
        speed1,
        speed2,
        direction * probe_radius,
        attacker_speed,
        radius,
        tolerance,
    )
    scale = max(1.0, radius, attacker_radius, barrier_radius)
    constrained = attacker_radius < barrier_radius - tolerance * scale
    projected_radius = barrier_radius if constrained else attacker_radius
    projected = projected_radius * direction
    return {
        "point": projected,
        "constrained": bool(constrained),
        "requested_radius": attacker_radius,
        "barrier_radius": barrier_radius,
        "active_source": source,
        "active_candidate": details,
    }
