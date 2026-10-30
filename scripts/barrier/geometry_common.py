"""Fixed-lift circle distances and simultaneous-arrival boundary roots.

Boundary sigma labels defender lifts (-1 lower, +1 upper), not travel direction.
"""

from math import acos, atan2, cos, pi, sin, sqrt
import numpy as np

TAU = 2.0 * pi
MODES = ((-1, +1), (+1, -1), (-1, -1), (+1, +1))


def lifted_distance(source, radius, angle, R):
    r0, theta0 = source
    r0 = np.asarray(r0, dtype=float)
    r = np.maximum(np.asarray(radius, dtype=float), R)
    delta = np.asarray(angle, dtype=float) - theta0
    beta = np.arccos(np.clip(R / r0, -1.0, 1.0)) + np.arccos(np.clip(R / r, -1.0, 1.0))
    straight = np.sqrt(
        np.maximum(0.0, (r - r0) ** 2 + 2 * r * r0 * (1 - np.cos(delta)))
    )
    around = (
        np.sqrt(np.maximum(0.0, r0 * r0 - R * R))
        + np.sqrt(np.maximum(0.0, r * r - R * R))
        + R * np.maximum(0.0, np.abs(delta) - beta)
    )
    return np.where(np.abs(delta) <= beta, straight, around)


def boundary_delta(g, phi, sigma):
    """Return the HSP angular difference for a paper defender-lift index.

    ``sigma=-1`` selects the adjacent lower defender lift and ``sigma=+1``
    selects the adjacent upper lift.  The signs therefore label lifts as in
    (9)--(11), rather than clockwise/counterclockwise travel directions.
    """
    if sigma not in (-1, 1):
        raise ValueError("sigma must be +1 or -1")
    lower_delta = (np.asarray(phi, dtype=float) - g[1]) % (2 * np.pi)
    return lower_delta if sigma == -1 else lower_delta - 2 * np.pi


def boundary_distance(g, phi, sigma, R):
    delta = boundary_delta(g, phi, sigma)
    value = lifted_distance((g[0], 0.0), R, delta, R)
    return float(value) if value.ndim == 0 else value


def boundary_derivative(g, phi, sigma, R):
    delta = boundary_delta(g, phi, sigma)
    distance = boundary_distance(g, phi, sigma, R)
    direct = np.abs(delta) <= g[2] + 1e-12
    derivative = np.divide(
        g[0] * R * np.sin(delta),
        distance,
        out=np.zeros_like(delta),
        where=np.asarray(distance) > 1e-14,
    )
    value = np.where(direct, derivative, -sigma * R)
    return float(value) if value.ndim == 0 else value


def geometry(x, radius=1.0):
    """Return (radial distance, angle, tangent angle, tangent length)."""
    x = np.asarray(x, dtype=float)
    r = float(np.linalg.norm(x))
    if r < radius:
        raise ValueError("A defender is inside the target disk.")
    return r, atan2(x[1], x[0]) % TAU, acos(radius / r), sqrt(r * r - radius * radius)


def dsp(g, phi, sigma, radius=1.0):
    """HSP distance to the lower/upper boundary lift in the universal cover."""
    return boundary_distance(g, phi, sigma, radius)


def compute_d_max(d1, alpha=2.0 / 3.0, radius=1.0):
    """Solve paper equation (29) at the antipodal attacker position."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1.")
    r = float(np.linalg.norm(d1)) / radius
    tangent_angle = acos(1.0 / r)

    def beta(L):
        return pi - tangent_angle - acos(alpha) + acos(alpha / L)

    def f(L):
        distance = float(lifted_distance((r, 0.0), 1.0, tangent_angle + beta(L), 1.0))
        return (
            sqrt(L * L - alpha * alpha) - sqrt(1.0 - alpha * alpha) - alpha * distance
        )

    lo, hi = 1.0, 2.0
    while f(hi) <= 0.0:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return float(
        lifted_distance(
            (r * radius, 0.0), radius, tangent_angle + beta(0.5 * (lo + hi)), radius
        )
    )


def _bisect(f, a, b):
    fa = f(a)
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < 1e-12:
            return m
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def _direct_distance(g, phi, radius):
    """Cover distance on the local, visible lift (used only on direct intervals)."""
    delta = (phi - g[1] + pi) % TAU - pi
    return float(lifted_distance((g[0], 0.0), radius, delta, radius))


def _indirect_distance(g, phi, sigma, radius):
    return boundary_distance(g, phi, sigma, radius)


def _is_direct(g, phi, sigma=None):
    """Check visibility of a specified lift, or the nearest copy if unspecified."""
    delta = (
        (phi - g[1] + pi) % TAU - pi if sigma is None else boundary_delta(g, phi, sigma)
    )
    return abs(delta) <= g[2] + 1e-12


def _visibility_cuts(g1, g2):
    """Visibility-transition angles for both defenders on [0, 2*pi]."""
    raw_cuts = []
    for _, theta, beta, _ in (g1, g2):
        # theta is the seam between the adjacent lower and upper lifts.
        raw_cuts += [(theta - beta) % TAU, theta % TAU, (theta + beta) % TAU]
    cuts = [0.0]
    for value in sorted(raw_cuts):
        if value <= 1e-13 or TAU - value <= 1e-13:
            continue
        if value - cuts[-1] > 1e-13:
            cuts.append(float(value))
    cuts.append(TAU)
    return cuts


def _direct_direct_roots(g1, g2, radius, speed_ratio=1.0):
    """Closed-form target intersections of the weighted Apollonius locus."""
    r1, theta1, _, _ = g1
    r2, theta2, _, _ = g2
    defender1 = r1 * np.array([cos(theta1), sin(theta1)])
    defender2 = r2 * np.array([cos(theta2), sin(theta2)])
    weight = float(speed_ratio) ** 2
    difference = weight * defender1 - defender2
    norm = float(np.linalg.norm(difference))
    rhs = (weight * (radius * radius + r1 * r1) - (radius * radius + r2 * r2)) / (
        2.0 * radius
    )

    if norm <= 1e-14:
        # A zero rhs is a continuum handled interval-by-interval by
        # mode_roots(); it must not be replaced by a minimum-d sample.
        return []

    ratio = rhs / norm
    if ratio < -1.0 - 1e-12 or ratio > 1.0 + 1e-12:
        return []
    offset = acos(float(np.clip(ratio, -1.0, 1.0)))
    base = atan2(difference[1], difference[0])
    return [(base - offset) % TAU, (base + offset) % TAU]


def _direct_derivative(g, phi, radius):
    """Derivative of straight-line boundary distance with respect to angle."""

    r, theta, _, _ = g
    length = _direct_distance(g, phi, radius)
    if length <= 1e-14:
        return 0.0
    return r * radius * sin(phi - theta) / length


def mode_roots(
    g1,
    g2,
    mode,
    radius=1.0,
    samples=6000,
    beta=1.0,
    return_intervals=False,
):
    """Find equal-arrival-time roots on the complete target boundary.

    Visibility cuts make both distance branches fixed on each open interval.
    Direct/direct and indirect/indirect roots are then available in closed
    form.  A weighted mixed branch can have one stationary point and hence
    two roots; its derivative root is inserted before monotone
    bisection.  ``samples`` remains in the signature for API compatibility
    but dense angular sampling is no longer used.
    """
    del samples
    beta = float(beta)
    if not beta > 0.0:
        raise ValueError("beta must be positive.")
    f = lambda phi: (
        dsp(g1, phi, mode[0], radius) - dsp(g2, phi, mode[1], radius) / beta
    )
    cuts = _visibility_cuts(g1, g2)

    roots = []
    root_intervals = []

    # Evaluate cuts using the same specified lifts as the open intervals.
    for x in cuts[:-1]:
        if abs(f(x)) < 1e-10:
            roots.append(x)

    direct_roots = _direct_direct_roots(g1, g2, radius, beta)
    for left, right in zip(cuts[:-1], cuts[1:]):
        width = right - left
        if width <= 1e-13:
            continue

        midpoint = 0.5 * (left + right)
        direct1, direct2 = _is_direct(g1, midpoint, mode[0]), _is_direct(
            g2, midpoint, mode[1]
        )

        if direct1 and direct2:
            # Detect an identically-zero direct/direct equation without a
            # beta-specific code path.  A nonconstant sinusoid cannot vanish
            # at these three distinct probes.
            probe = min(0.25 * width, 2e-10)
            if (
                abs(f(left + probe)) <= 1e-10
                and abs(f(midpoint)) <= 1e-10
                and abs(f(right - probe)) <= 1e-10
            ):
                root_intervals.append((left, right))
                continue
            for root in direct_roots:
                if left + 1e-11 < root < right - 1e-11:
                    roots.append(root)
            continue

        def branch_f(phi):
            value1 = (
                _direct_distance(g1, phi, radius)
                if direct1
                else _indirect_distance(g1, phi, mode[0], radius)
            )
            value2 = (
                _direct_distance(g2, phi, radius)
                if direct2
                else _indirect_distance(g2, phi, mode[1], radius)
            )
            return value1 - value2 / beta

        # Avoid the lift's visibility tolerance when probing open intervals.
        # Cut points have already been evaluated with the same cover metric.
        edge = min(0.25 * width, 2e-12)
        a, b = left + edge, right - edge
        if a >= b:
            continue

        if not direct1 and not direct2:
            slope = radius * (-mode[0] + mode[1] / beta)
            if abs(slope) <= 1e-13:
                # A zero constant is a continuum, not an isolated root.
                if abs(branch_f(midpoint)) <= 1e-10:
                    root_intervals.append((left, right))
                continue
            root = midpoint - branch_f(midpoint) / slope
            if a <= root <= b:
                roots.append(root)
            continue

        # A mixed branch can turn once when its direct defender is the faster
        # one in the current normalization.  Locate that derivative root
        # explicitly; this covers both beta <= 1 and internally relabelled
        # beta > 1 configurations without dense angular sampling.
        critical = []
        if direct1 and not direct2:
            derivative = lambda phi: (
                _direct_derivative(g1, phi, radius) + mode[1] * radius / beta
            )
        else:
            derivative = lambda phi: (
                -mode[0] * radius - _direct_derivative(g2, phi, radius) / beta
            )
        da, db = derivative(a), derivative(b)
        if da * db < 0.0:
            critical.append(_bisect(derivative, a, b))

        partitions = [a, *critical, b]
        for sub_left, sub_right in zip(partitions[:-1], partitions[1:]):
            fleft, fright = branch_f(sub_left), branch_f(sub_right)
            if abs(fleft) <= 1e-11:
                roots.append(sub_left)
            if fleft * fright < 0.0:
                roots.append(_bisect(branch_f, sub_left, sub_right))
        if abs(branch_f(b)) <= 1e-11:
            roots.append(b)

    unique = []
    # Verify roots with the cover metric and merge angular duplicates.
    valid = [x % TAU for x in roots if abs(f(x % TAU)) <= 1e-8]
    for x in sorted(valid):
        if not unique or abs((x - unique[-1] + pi) % TAU - pi) > 1e-7:
            unique.append(x)
    if len(unique) > 1 and abs((unique[0] - unique[-1] + pi) % TAU - pi) <= 1e-7:
        unique.pop()
    if return_intervals:
        return unique, root_intervals
    return unique


def solve_p(
    d1,
    d2,
    radius=1.0,
    alpha=2.0 / 3.0,
    d1_max=None,
    d2_max=None,
    beta=1.0,
):
    """Return four-mode simultaneous-arrival roots that satisfy both d_max limits.

    Speeds are normalized as ``v_D1=1``, ``v_D2=beta``, ``v_A=alpha``.
    Thus a root satisfies ``distance_D1 = distance_D2 / beta``.  Candidate
    ``d1`` and ``d2`` are physical path lengths; ``arrival_time`` and the
    backward-compatible ``d`` field are the common normalized time.  The
    latter keeps the attacker-circle radius equal to ``alpha * d``.

    The distance test is defender-specific: a candidate survives only when
    ``d1 <= d1_max`` and ``d2 <= d2_max``.  By default the maxima use attacker
    ratios ``alpha`` for D1 and ``alpha/beta`` for D2.  No minimum-distance
    selection is imposed within a defender-lift mode.
    """
    d1, d2 = np.asarray(d1, dtype=float), np.asarray(d2, dtype=float)
    beta = float(beta)
    if not beta > 0.0:
        raise ValueError("beta must be positive.")
    if not 0.0 < alpha <= min(1.0, beta) + 1e-14:
        raise ValueError("The attacker ratio must satisfy 0 < alpha <= min(1, beta).")
    g1, g2 = geometry(d1, radius), geometry(d2, radius)
    d1_max = compute_d_max(d1, alpha, radius) if d1_max is None else float(d1_max)
    d2_max = (
        compute_d_max(d2, alpha / beta, radius) if d2_max is None else float(d2_max)
    )
    if not d1_max > 0.0 or not d2_max > 0.0:
        raise ValueError("d1_max and d2_max must be positive.")

    candidates = []
    root_intervals = []
    for mode_index, mode in enumerate(MODES):
        mode_candidates = []
        roots, intervals = mode_roots(
            g1,
            g2,
            mode,
            radius,
            beta=beta,
            return_intervals=True,
        )
        generating_mode = (
            "+" if mode[0] > 0 else "-",
            "+" if mode[1] > 0 else "-",
        )
        root_intervals.extend(
            {
                "mode_index": mode_index,
                "mode": generating_mode,
                "phi_start": float(left),
                "phi_end": float(right),
            }
            for left, right in intervals
        )
        for phi in roots:
            path1 = float(dsp(g1, phi, mode[0], radius))
            path2 = float(dsp(g2, phi, mode[1], radius))
            time1 = path1
            time2 = path2 / beta
            arrival_time = 0.5 * (time1 + time2)
            direct = (_is_direct(g1, phi, mode[0]), _is_direct(g2, phi, mode[1]))
            distance_feasible = bool(
                path1 <= d1_max + 1e-10 and path2 <= d2_max + 1e-10
            )
            item = {
                "mode_index": mode_index,
                "mode": generating_mode,
                "effective_mode": tuple(
                    "d" if direct[index] else generating_mode[index]
                    for index in range(2)
                ),
                "direct": direct,
                "phi": phi,
                "p": radius * np.array([cos(phi), sin(phi)]),
                "d": arrival_time,
                "d1": path1,
                "d2": path2,
                "time1": time1,
                "time2": time2,
                "arrival_time": arrival_time,
                "arrival_residual": abs(time1 - time2),
                "distance_feasible": distance_feasible,
            }
            mode_candidates.append(item)
        candidates.extend(mode_candidates)

    # Merge identical distance-feasible points while retaining their mode labels.
    points = []
    for item in [x for x in candidates if x["distance_feasible"]]:
        point = next(
            (q for q in points if abs((q["phi"] - item["phi"] + pi) % TAU - pi) < 1e-7),
            None,
        )
        if point is None:
            point = {"phi": item["phi"], "p": item["p"], "solutions": []}
            points.append(point)
        point["solutions"].append(item)

    by_mode = {index: [] for index in range(len(MODES))}
    for item in candidates:
        if item["distance_feasible"]:
            by_mode[item["mode_index"]].append(item)

    status = "feasible_p_points" if points else "no_feasible_p"

    points.sort(key=lambda q: min(x["mode_index"] for x in q["solutions"]))
    return {
        "status": status,
        "d1_max": d1_max,
        "d2_max": d2_max,
        "beta": beta,
        "speeds": {"D1": 1.0, "D2": beta, "A": float(alpha)},
        "points": points,
        "candidates": candidates,
        "by_mode": by_mode,
        "root_intervals": root_intervals,
        "modes": MODES,
    }
