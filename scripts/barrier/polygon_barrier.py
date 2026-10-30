"""Fixed-lift convex-polygon barriers and cooperative boundary envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, pi
import time
from typing import Any, Sequence
import numpy as np

ArrayLike2 = Sequence[float] | np.ndarray
TAU = 2.0 * pi
ALPHA = 2.0 / 3.0
BETA = 1.0
LIFT_PAIRS = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))
P_ARC_COLORS = ("#2a9d68", "#805ad5", "#a05a2c", "#d65c8a", "#7b8e23", "#28a5a5")


@dataclass(frozen=True)
class BarrierResult:
    """Numerical barrier and the active target/path mode at every point."""

    points: np.ndarray
    meeting_point: np.ndarray
    active_target_points: np.ndarray
    active_modes: np.ndarray
    target_vertices: np.ndarray
    defender: np.ndarray
    alpha: float
    vertex_distances: np.ndarray
    visible_edges: np.ndarray


@dataclass(frozen=True)
class _DirectionalBranch:
    """One recursively generated fixed-lift barrier chain."""

    points: np.ndarray
    segment_modes: np.ndarray
    segment_targets: np.ndarray
    boundary_parameters: np.ndarray


@dataclass(frozen=True)
class _BoundaryHSPPiece:
    """One analytic piece of a fixed-lift distance on a polygon edge."""

    edge_index: int
    start_s: float
    end_s: float
    progress: float
    distance_at_start: float
    direct: bool
    traversal_sign: int


@dataclass(frozen=True)
class BoundaryHSPMetric:
    """HSP distances from one defender to both adjacent boundary lifts.

    Lift index ``-1`` is the lower lift and advances along the CCW-oriented
    polygon boundary. Lift index ``+1`` is the upper lift and advances CW.
    """

    defender: np.ndarray
    polygon: np.ndarray
    tangents: np.ndarray
    pieces: dict[int, tuple[_BoundaryHSPPiece, ...]]
    seam_edge: int
    seam_s: float
    tolerance: float

    def evaluate(
        self,
        edge_index: int,
        local_s: np.ndarray | float,
        sigma: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return distance and d(distance)/d(local_s)."""

        if sigma not in (-1, +1):
            raise ValueError("sigma must be -1 (lower) or +1 (upper).")
        values = np.asarray(local_s, dtype=float)
        flat = values.reshape(-1)
        distances = np.full(len(flat), np.inf)
        derivatives = np.zeros(len(flat))
        progress = np.full(len(flat), np.inf)
        tangent = self.tangents[edge_index]
        targets = self.polygon[edge_index] + flat[:, None] * tangent

        for piece in self.pieces[sigma]:
            if piece.edge_index != edge_index:
                continue
            lo, hi = sorted((piece.start_s, piece.end_s))
            mask = (flat >= lo - self.tolerance) & (flat <= hi + self.tolerance)
            if not np.any(mask):
                continue
            masked_indices = np.flatnonzero(mask)
            travel = piece.traversal_sign * (flat[masked_indices] - piece.start_s)
            travel = np.maximum(0.0, travel)
            candidate_progress = piece.progress + travel
            at_upper_seam = (
                sigma == +1
                and edge_index == self.seam_edge
                and np.abs(flat[masked_indices] - self.seam_s) <= self.tolerance
            )
            replace = np.where(
                at_upper_seam,
                candidate_progress > progress[masked_indices],
                candidate_progress < progress[masked_indices],
            )
            replace |= at_upper_seam & ~np.isfinite(progress[masked_indices])
            indices = masked_indices[replace]
            if not len(indices):
                continue
            selected_travel = piece.traversal_sign * (flat[indices] - piece.start_s)
            if piece.direct:
                delta = targets[indices] - self.defender
                candidate_distance = np.linalg.norm(delta, axis=1)
                candidate_derivative = np.divide(
                    delta @ tangent,
                    candidate_distance,
                    out=np.zeros_like(candidate_distance),
                    where=candidate_distance > self.tolerance,
                )
            else:
                candidate_distance = piece.distance_at_start + selected_travel
                candidate_derivative = np.full(len(indices), piece.traversal_sign)
            distances[indices] = candidate_distance
            derivatives[indices] = candidate_derivative
            progress[indices] = piece.progress + selected_travel

        if np.any(~np.isfinite(distances)):
            raise RuntimeError("Boundary point is not covered by the HSP lift.")
        shape = values.shape
        return distances.reshape(shape), derivatives.reshape(shape)


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
    )


def _prepare_polygon(vertices: Sequence[ArrayLike2], tol: float) -> np.ndarray:
    polygon = np.asarray(vertices, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("vertices must have shape (n, 2).")
    if len(polygon) > 1 and np.linalg.norm(polygon[0] - polygon[-1]) <= tol:
        polygon = polygon[:-1]
    if len(polygon) < 3:
        raise ValueError("A polygon needs at least three distinct vertices.")

    # Remove consecutive duplicates and collinear middle vertices.
    changed = True
    while changed and len(polygon) >= 3:
        changed = False
        keep = []
        n = len(polygon)
        for i in range(n):
            prev, cur, nxt = polygon[i - 1], polygon[i], polygon[(i + 1) % n]
            if np.linalg.norm(cur - prev) <= tol:
                changed = True
                continue
            if (
                abs(_cross2(cur - prev, nxt - cur))
                <= tol * (np.linalg.norm(cur - prev) + np.linalg.norm(nxt - cur) + 1.0)
                and np.dot(cur - prev, nxt - cur) >= 0.0
            ):
                changed = True
                continue
            keep.append(cur)
        polygon = np.asarray(keep, dtype=float)

    if len(polygon) < 3 or abs(_signed_area(polygon)) <= tol:
        raise ValueError("The polygon is degenerate.")
    if _signed_area(polygon) < 0.0:
        polygon = polygon[::-1].copy()

    edges = np.roll(polygon, -1, axis=0) - polygon
    turns = _cross2(edges, np.roll(edges, -1, axis=0))
    if np.any(turns < -tol):
        raise ValueError("vertices must describe a convex polygon in boundary order.")
    return polygon


def _polygon_geometry(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    tangents = edges / lengths[:, None]
    # A CCW polygon has its exterior on the right side of every directed edge.
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    return lengths, tangents, normals


def _strictly_inside_convex(point: np.ndarray, polygon: np.ndarray, tol: float) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    return bool(np.all(_cross2(edges, point - polygon) > tol))


def _vertex_geodesic_distances(
    defender: np.ndarray,
    polygon: np.ndarray,
    lengths: np.ndarray,
    normals: np.ndarray,
    visible_edges: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Exact D-to-vertex distances for one convex polygon."""

    n = len(polygon)
    vertex_visible = visible_edges | np.roll(visible_edges, 1)
    distances = np.full(n, np.inf)
    distances[vertex_visible] = np.linalg.norm(
        polygon[vertex_visible] - defender, axis=1
    )

    # Multi-source Dijkstra on the polygon boundary cycle.
    used = np.zeros(n, dtype=bool)
    for _ in range(n):
        candidates = np.where(used, np.inf, distances)
        i = int(np.argmin(candidates))
        if not np.isfinite(candidates[i]):
            break
        used[i] = True
        j_next = (i + 1) % n
        j_prev = (i - 1) % n
        distances[j_next] = min(distances[j_next], distances[i] + lengths[i])
        distances[j_prev] = min(distances[j_prev], distances[i] + lengths[j_prev])

    if np.any(~np.isfinite(distances)):
        raise RuntimeError("Could not construct defender geodesic distances.")

    # Directly visible vertices must retain their Euclidean distance.
    euclidean = np.linalg.norm(polygon - defender, axis=1)
    distances[vertex_visible] = np.minimum(
        distances[vertex_visible], euclidean[vertex_visible]
    )
    return distances


def _closest_boundary_start(
    defender: np.ndarray,
    polygon: np.ndarray,
    lengths: np.ndarray,
    tangents: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, int, int | None]:
    """Closest target-boundary point and its edge/arclength representation."""

    best_distance = np.inf
    best_edge = 0
    best_point = polygon[0].copy()
    for edge_index, (start, length, tangent) in enumerate(
        zip(polygon, lengths, tangents)
    ):
        s = float(np.clip(np.dot(defender - start, tangent), 0.0, length))
        point = start + s * tangent
        distance = float(np.linalg.norm(defender - point))
        if distance < best_distance:
            best_distance = distance
            best_edge = edge_index
            best_point = point

    scale = max(float(np.max(lengths)), 1.0)
    vertex_distances = np.linalg.norm(polygon - best_point, axis=1)
    nearest_vertex = int(np.argmin(vertex_distances))
    if vertex_distances[nearest_vertex] <= 32.0 * tol * scale:
        return polygon[nearest_vertex].copy(), best_edge, nearest_vertex
    return best_point, best_edge, None


def _directional_edge_pieces(
    polygon: np.ndarray,
    start_point: np.ndarray,
    start_edge: int,
    start_vertex: int | None,
    direction_sign: int,
    tol: float,
    traversal_count: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Split one complete directed boundary traversal into oriented edge pieces."""

    n = len(polygon)
    pieces: list[tuple[int, np.ndarray, np.ndarray]] = []

    def append_piece(edge: int, first: np.ndarray, second: np.ndarray) -> None:
        if np.linalg.norm(second - first) > tol:
            pieces.append((edge, first.copy(), second.copy()))

    if start_vertex is not None:
        for offset in range(n):
            if direction_sign > 0:
                edge = (start_vertex + offset) % n
                append_piece(edge, polygon[edge], polygon[(edge + 1) % n])
            else:
                edge = (start_vertex - 1 - offset) % n
                append_piece(edge, polygon[(edge + 1) % n], polygon[edge])
    else:
        if direction_sign > 0:
            append_piece(start_edge, start_point, polygon[(start_edge + 1) % n])
            for offset in range(1, n):
                edge = (start_edge + offset) % n
                append_piece(edge, polygon[edge], polygon[(edge + 1) % n])
            append_piece(start_edge, polygon[start_edge], start_point)
        else:
            append_piece(start_edge, start_point, polygon[start_edge])
            for offset in range(1, n):
                edge = (start_edge - offset) % n
                append_piece(edge, polygon[(edge + 1) % n], polygon[edge])
            append_piece(start_edge, polygon[(start_edge + 1) % n], start_point)

    one_traversal = pieces
    return [piece for _ in range(traversal_count) for piece in one_traversal]


def boundary_hsp_metric(
    defender: np.ndarray,
    polygon: np.ndarray,
    lengths: np.ndarray,
    tangents: np.ndarray,
    visible_edges: np.ndarray,
    tol: float,
) -> BoundaryHSPMetric:
    """Construct the two adjacent-lift HSP metrics on a convex polygon.

    The cover seam is the boundary point nearest the defender.  The lower
    lift uses a nonnegative CCW traversal from that seam; the upper lift uses
    a nonpositive CW traversal.  Direct visibility is retained only before
    the corresponding lifted path reaches its first silhouette vertex.
    """

    start_point, start_edge, start_vertex = _closest_boundary_start(
        defender, polygon, lengths, tangents, tol
    )
    if start_vertex is None:
        seam_edge = start_edge
        seam_s = float(np.dot(start_point - polygon[start_edge], tangents[start_edge]))
    else:
        # Boundary vertices use their outgoing CCW edge as the canonical
        # half-open representation, matching s in [0, perimeter).
        seam_edge = start_vertex
        seam_s = 0.0
    by_lift: dict[int, tuple[_BoundaryHSPPiece, ...]] = {}
    for sigma in (-1, +1):
        traversal_sign = -sigma
        traversed = _directional_edge_pieces(
            polygon,
            start_point,
            start_edge,
            start_vertex,
            traversal_sign,
            tol,
        )
        current_distance = float(np.linalg.norm(defender - start_point))
        progress = 0.0
        direct = True
        records: list[_BoundaryHSPPiece] = []
        for edge_index, first, second in traversed:
            if direct and not visible_edges[edge_index]:
                direct = False
            start_s = float(np.dot(first - polygon[edge_index], tangents[edge_index]))
            end_s = float(np.dot(second - polygon[edge_index], tangents[edge_index]))
            records.append(
                _BoundaryHSPPiece(
                    edge_index=edge_index,
                    start_s=start_s,
                    end_s=end_s,
                    progress=progress,
                    distance_at_start=current_distance,
                    direct=direct,
                    traversal_sign=traversal_sign,
                )
            )
            piece_length = float(np.linalg.norm(second - first))
            progress += piece_length
            current_distance = (
                float(np.linalg.norm(defender - second))
                if direct
                else current_distance + piece_length
            )
        by_lift[sigma] = tuple(records)
    return BoundaryHSPMetric(
        defender=np.asarray(defender, dtype=float).copy(),
        polygon=np.asarray(polygon, dtype=float).copy(),
        tangents=np.asarray(tangents, dtype=float).copy(),
        pieces=by_lift,
        seam_edge=seam_edge,
        seam_s=seam_s,
        tolerance=64.0 * tol * max(1.0, float(np.max(lengths))),
    )


def _sample_oriented_circle_arc(
    center: np.ndarray,
    radius: float,
    first_direction: np.ndarray,
    second_direction: np.ndarray,
    direction_sign: int,
    angular_resolution: int,
    tol: float,
) -> np.ndarray:
    """Sample the direction-consistent circle arc between two unit radii."""

    first_direction = first_direction / max(float(np.linalg.norm(first_direction)), tol)
    second_direction = second_direction / max(
        float(np.linalg.norm(second_direction)), tol
    )
    first_angle = float(np.arctan2(first_direction[1], first_direction[0]))
    second_angle = float(np.arctan2(second_direction[1], second_direction[0]))
    if direction_sign > 0:
        sweep = (second_angle - first_angle) % (2.0 * np.pi)
    else:
        sweep = -((first_angle - second_angle) % (2.0 * np.pi))
    if abs(abs(sweep) - 2.0 * np.pi) <= 128.0 * tol:
        sweep = 0.0

    interval_count = max(
        1,
        int(np.ceil(abs(sweep) * angular_resolution / (2.0 * np.pi))),
    )
    angles = np.linspace(first_angle, first_angle + sweep, interval_count + 1)
    points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    points[0] = center + radius * first_direction
    points[-1] = center + radius * second_direction
    return points


def _build_directional_branch(
    defender: np.ndarray,
    polygon: np.ndarray,
    alpha: float,
    lengths: np.ndarray,
    normals: np.ndarray,
    visible_edges: np.ndarray,
    start_point: np.ndarray,
    start_edge: int,
    start_vertex: int | None,
    direction_sign: int,
    samples_per_edge: int,
    angular_resolution: int,
    tol: float,
    traversal_count: int = 1,
) -> _DirectionalBranch:
    """Propagate one fixed-lift barrier chain around the polygon."""

    pieces = _directional_edge_pieces(
        polygon,
        start_point,
        start_edge,
        start_vertex,
        direction_sign,
        tol,
        traversal_count,
    )
    if not pieces:
        raise RuntimeError("Could not construct a fixed-lift target traversal.")

    points: list[np.ndarray] = []
    modes: list[str] = []
    targets: list[np.ndarray] = []
    parameters: list[float] = []
    boundary_parameter = 0.0
    current_distance = float(np.linalg.norm(defender - start_point))
    direct = True
    previous_radius_direction: np.ndarray | None = None
    previous_edge: int | None = None

    for edge_index, first_target, second_target in pieces:
        piece_vector = second_target - first_target
        piece_length = float(np.linalg.norm(piece_vector))
        direction = piece_vector / piece_length
        outward_normal = normals[edge_index]
        if direct and not visible_edges[edge_index]:
            direct = False

        if direct:
            interval_count = max(
                1,
                int(np.ceil(samples_per_edge * piece_length / lengths[edge_index])),
            )
        else:
            # The indirect envelope on one polygon edge is exactly a line.
            interval_count = 1
        local_s = np.linspace(0.0, piece_length, interval_count + 1)
        target_points = first_target + local_s[:, None] * direction

        if direct:
            target_delta = target_points - defender
            distances = np.linalg.norm(target_delta, axis=1)
            derivatives = np.divide(
                target_delta @ direction,
                distances,
                out=np.zeros_like(distances),
                where=distances > tol,
            )
            mode = "direct"
        else:
            distances = current_distance + local_s
            derivatives = np.ones_like(local_s)
            mode = "indirect_ccw" if direction_sign > 0 else "indirect_cw"

        # Envelope condition for ||x-p(s)|| = alpha*d(s): if u is the
        # target-to-barrier unit radius, then u dot p'(s) = -alpha*d'(s).
        # On an indirect edge d'=1, so the resulting envelope is an exact
        # straight line making asin(alpha) with that target edge.
        tangential_component = -alpha * derivatives
        normal_component = np.sqrt(np.maximum(0.0, 1.0 - tangential_component**2))
        radius_directions = (
            tangential_component[:, None] * direction
            + normal_component[:, None] * outward_normal
        )
        envelope_points = target_points + alpha * distances[:, None] * radius_directions

        if previous_radius_direction is not None and edge_index != previous_edge:
            arc = _sample_oriented_circle_arc(
                first_target,
                alpha * current_distance,
                previous_radius_direction,
                radius_directions[0],
                direction_sign,
                angular_resolution,
                tol,
            )
            arc[-1] = envelope_points[0]
            for arc_point in arc[1:]:
                points.append(arc_point)
                modes.append("vertex_arc")
                targets.append(first_target.copy())
                parameters.append(boundary_parameter)
        elif previous_radius_direction is None:
            points.append(envelope_points[0])
            parameters.append(boundary_parameter)

        for index in range(1, len(envelope_points)):
            points.append(envelope_points[index])
            modes.append(mode)
            targets.append(0.5 * (target_points[index - 1] + target_points[index]))
            parameters.append(boundary_parameter + local_s[index])

        boundary_parameter += piece_length
        current_distance = float(distances[-1])
        previous_radius_direction = radius_directions[-1]
        previous_edge = edge_index

    point_array = np.asarray(points, dtype=float)
    mode_array = np.asarray(modes, dtype=object)
    target_array = np.asarray(targets, dtype=float)
    parameter_array = np.asarray(parameters, dtype=float)
    if len(point_array) != len(mode_array) + 1:
        raise RuntimeError("Directional barrier metadata became inconsistent.")
    return _DirectionalBranch(
        points=point_array,
        segment_modes=mode_array,
        segment_targets=target_array,
        boundary_parameters=parameter_array,
    )


def _first_branch_intersection(
    first: _DirectionalBranch,
    second: _DirectionalBranch,
    scale: float,
    tol: float,
) -> tuple[int, float, int, float, np.ndarray]:
    """Find the first nontrivial meeting of two advancing lift chains."""

    first_starts = first.points[:-1]
    first_vectors = first.points[1:] - first.points[:-1]
    second_starts = second.points[:-1]
    second_vectors = second.points[1:] - second.points[:-1]
    perimeter = max(
        float(first.boundary_parameters[-1]),
        float(second.boundary_parameters[-1]),
        tol,
    )
    parameter_tol = 64.0 * tol * perimeter
    best: tuple[float, int, float, int, float, np.ndarray] | None = None

    for chunk_start in range(0, len(first_vectors), 64):
        chunk_stop = min(chunk_start + 64, len(first_vectors))
        p = first_starts[chunk_start:chunk_stop, None, :]
        r = first_vectors[chunk_start:chunk_stop, None, :]
        q = second_starts[None, :, :]
        s = second_vectors[None, :, :]
        difference = q - p
        denominator = _cross2(r, s)
        denominator_tol = tol * (
            np.linalg.norm(r, axis=2) * np.linalg.norm(s, axis=2) + scale
        )
        nonparallel = np.abs(denominator) > denominator_tol
        first_fraction = np.full(denominator.shape, np.inf)
        second_fraction = np.full(denominator.shape, np.inf)
        np.divide(
            _cross2(difference, s),
            denominator,
            out=first_fraction,
            where=nonparallel,
        )
        np.divide(
            _cross2(difference, r),
            denominator,
            out=second_fraction,
            where=nonparallel,
        )
        valid = (
            nonparallel
            & (first_fraction >= -tol)
            & (first_fraction <= 1.0 + tol)
            & (second_fraction >= -tol)
            & (second_fraction <= 1.0 + tol)
        )
        if not np.any(valid):
            continue

        first_indices = np.arange(chunk_start, chunk_stop)[:, None]
        second_indices = np.arange(len(second_vectors))[None, :]
        safe_first_fraction = np.where(np.isfinite(first_fraction), first_fraction, 0.0)
        safe_second_fraction = np.where(
            np.isfinite(second_fraction), second_fraction, 0.0
        )
        first_parameter = first.boundary_parameters[
            first_indices
        ] + safe_first_fraction * (
            first.boundary_parameters[first_indices + 1]
            - first.boundary_parameters[first_indices]
        )
        second_parameter = second.boundary_parameters[
            second_indices
        ] + safe_second_fraction * (
            second.boundary_parameters[second_indices + 1]
            - second.boundary_parameters[second_indices]
        )
        valid &= (first_parameter > parameter_tol) & (second_parameter > parameter_tol)
        if np.linalg.norm(first.points[0] - second.points[0]) <= 64.0 * tol * scale:
            candidate_points = p + safe_first_fraction[:, :, None] * r
            valid &= (
                np.linalg.norm(candidate_points - first.points[0], axis=2)
                > 64.0 * tol * scale
            )
        if not np.any(valid):
            continue

        progress = np.maximum(first_parameter, second_parameter) / perimeter
        progress += 1.0e-6 * (first_parameter + second_parameter) / perimeter
        progress[~valid] = np.inf
        flat_index = int(np.argmin(progress))
        local_first, local_second = np.unravel_index(flat_index, progress.shape)
        score = float(progress[local_first, local_second])
        if best is None or score < best[0]:
            i = chunk_start + local_first
            j = local_second
            first_t = float(np.clip(first_fraction[local_first, j], 0.0, 1.0))
            second_t = float(np.clip(second_fraction[local_first, j], 0.0, 1.0))
            first_point = first.points[i] + first_t * first_vectors[i]
            second_point = second.points[j] + second_t * second_vectors[j]
            best = (
                score,
                i,
                first_t,
                j,
                second_t,
                0.5 * (first_point + second_point),
            )

    if best is None:
        raise RuntimeError(
            "The lower- and upper-lift barrier chains did not intersect; "
            "increase grid_size."
        )
    _, first_index, first_t, second_index, second_t, point = best
    return first_index, first_t, second_index, second_t, point


def _truncate_branch(
    branch: _DirectionalBranch,
    segment_index: int,
    fraction: float,
    endpoint: np.ndarray,
    tol: float,
) -> _DirectionalBranch:
    """Keep one branch from its start through a specified segment point."""

    if fraction <= tol:
        points = branch.points[: segment_index + 1].copy()
        if len(points) == 0:
            points = endpoint[None, :].copy()
        else:
            points[-1] = endpoint
        modes = branch.segment_modes[:segment_index].copy()
        targets = branch.segment_targets[:segment_index].copy()
        parameters = branch.boundary_parameters[: segment_index + 1].copy()
    else:
        points = np.vstack((branch.points[: segment_index + 1], endpoint))
        modes = branch.segment_modes[: segment_index + 1].copy()
        targets = branch.segment_targets[: segment_index + 1].copy()
        interpolated_parameter = branch.boundary_parameters[
            segment_index
        ] + fraction * (
            branch.boundary_parameters[segment_index + 1]
            - branch.boundary_parameters[segment_index]
        )
        parameters = np.r_[
            branch.boundary_parameters[: segment_index + 1],
            interpolated_parameter,
        ]
    return _DirectionalBranch(points, modes, targets, parameters)


def _assemble_recursive_barrier(
    ccw: _DirectionalBranch,
    cw: _DirectionalBranch,
    intersection: tuple[int, float, int, float, np.ndarray],
    start_point: np.ndarray,
    angular_resolution: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join truncated lower/upper chains and their optional front vertex arc."""

    ccw_index, ccw_fraction, cw_index, cw_fraction, meeting_point = intersection
    ccw_part = _truncate_branch(ccw, ccw_index, ccw_fraction, meeting_point, tol)
    cw_part = _truncate_branch(cw, cw_index, cw_fraction, meeting_point, tol)

    points = np.vstack((ccw_part.points, cw_part.points[-2::-1]))
    modes = np.concatenate((ccw_part.segment_modes, cw_part.segment_modes[::-1]))
    targets = np.vstack((ccw_part.segment_targets, cw_part.segment_targets[::-1]))

    scale = max(float(np.max(np.linalg.norm(points - start_point, axis=1))), 1.0)
    if np.linalg.norm(points[-1] - points[0]) <= 64.0 * tol * scale:
        points[-1] = points[0]
    else:
        radius = float(np.linalg.norm(points[0] - start_point))
        if radius <= tol:
            points = np.vstack((points, points[0]))
            modes = np.r_[modes, "vertex_arc"]
            targets = np.vstack((targets, start_point))
        else:
            first_direction = (points[-1] - start_point) / radius
            second_direction = (points[0] - start_point) / radius
            front_arc = _sample_oriented_circle_arc(
                start_point,
                radius,
                first_direction,
                second_direction,
                +1,
                angular_resolution,
                tol,
            )
            front_arc[0] = points[-1]
            front_arc[-1] = points[0]
            points = np.vstack((points, front_arc[1:]))
            modes = np.concatenate(
                (modes, np.full(len(front_arc) - 1, "vertex_arc", dtype=object))
            )
            targets = np.vstack(
                (
                    targets,
                    np.repeat(start_point[None, :], len(front_arc) - 1, axis=0),
                )
            )

    if len(points) != len(modes) + 1:
        raise RuntimeError("Closed recursive barrier metadata became inconsistent.")
    if _signed_area(points[:-1]) < 0.0:
        points = points[::-1].copy()
        modes = modes[::-1].copy()
        targets = targets[::-1].copy()
    active_modes = np.r_[modes, modes[:1]]
    active_targets = np.vstack((targets, targets[:1]))
    return points, active_modes, active_targets


def compute_barrier(
    defender: ArrayLike2,
    vertices: Sequence[ArrayLike2],
    alpha: float,
    *,
    samples_per_edge: int = 180,
    grid_size: int = 620,
    padding_fraction: float = 0.06,
    tol: float = 1.0e-10,
) -> BarrierResult:
    """Compute the outer 1D1A barrier for an arbitrary convex polygon.

    Parameters
    ----------
    defender:
        Defender position, outside the target polygon.
    vertices:
        Convex polygon vertices in clockwise or counter-clockwise boundary
        order.  A repeated last vertex is allowed.
    alpha:
        Speed ratio v_A / v_D, with 0 < alpha <= 1.
    samples_per_edge:
        Samples used for the continuous family of boundary target points.
    grid_size:
        Angular resolution used for polygon-vertex connecting arcs.  The
        implementation uses at least ``max(720, 4*grid_size)`` samples per
        complete circle.
    padding_fraction:
        Retained for call compatibility with the former Cartesian-grid solver.
        It has no geometric effect in the recursive construction.

    Returns
    -------
    BarrierResult
        Closed CCW barrier polyline and active-mode diagnostics.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1.")
    if samples_per_edge < 20 or grid_size < 150:
        raise ValueError("Use samples_per_edge >= 20 and grid_size >= 150.")
    if padding_fraction < 0.0:
        raise ValueError("padding_fraction must be nonnegative.")

    polygon = _prepare_polygon(vertices, tol)
    d = np.asarray(defender, dtype=float)
    if d.shape != (2,):
        raise ValueError("defender must be a length-2 vector.")
    if _strictly_inside_convex(d, polygon, tol):
        raise ValueError("The defender must not lie inside the target polygon.")

    lengths, tangents, normals = _polygon_geometry(polygon)
    visible_edges = np.einsum("ij,ij->i", d - polygon, normals) >= -tol
    if not np.any(visible_edges):
        raise ValueError("The defender must be outside the convex polygon.")
    vertex_dist = _vertex_geodesic_distances(
        d, polygon, lengths, normals, visible_edges, tol
    )
    start_point, start_edge, start_vertex = _closest_boundary_start(
        d, polygon, lengths, tangents, tol
    )
    angular_resolution = max(720, 4 * int(grid_size))
    scale = max(
        float(np.max(np.linalg.norm(polygon - np.mean(polygon, axis=0), axis=1))),
        float(np.linalg.norm(d - start_point)),
        1.0,
    )
    intersection = None
    for traversal_count in (1, 2):
        lower_branch = _build_directional_branch(
            d,
            polygon,
            alpha,
            lengths,
            normals,
            visible_edges,
            start_point,
            start_edge,
            start_vertex,
            +1,
            samples_per_edge,
            angular_resolution,
            tol,
            traversal_count,
        )
        upper_branch = _build_directional_branch(
            d,
            polygon,
            alpha,
            lengths,
            normals,
            visible_edges,
            start_point,
            start_edge,
            start_vertex,
            -1,
            samples_per_edge,
            angular_resolution,
            tol,
            traversal_count,
        )
        try:
            intersection = _first_branch_intersection(
                lower_branch, upper_branch, scale, tol
            )
            break
        except RuntimeError:
            if traversal_count == 2:
                raise
    if intersection is None:
        raise RuntimeError("Could not close the recursive barrier.")
    points, active_modes, active_target_points = _assemble_recursive_barrier(
        lower_branch,
        upper_branch,
        intersection,
        start_point,
        angular_resolution,
        tol,
    )
    return BarrierResult(
        points=points,
        meeting_point=np.asarray(intersection[4], dtype=float).copy(),
        active_target_points=active_target_points,
        active_modes=active_modes,
        target_vertices=polygon,
        defender=d,
        alpha=float(alpha),
        vertex_distances=vertex_dist,
        visible_edges=visible_edges,
    )


@dataclass(frozen=True)
class _ArrivalMode:
    """One defender HSP lift on one target edge."""

    defender_index: int
    lift: int


def _defender_geometry(
    defender: np.ndarray,
    polygon: np.ndarray,
    lengths: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    tol: float,
) -> dict[str, Any]:
    visible_edges = np.einsum("ij,ij->i", defender - polygon, normals) >= -tol
    return {
        "visible_edges": visible_edges,
        "hsp_metric": boundary_hsp_metric(
            defender,
            polygon,
            lengths,
            tangents,
            visible_edges,
            tol,
        ),
    }


def _mode_radius(
    mode: _ArrivalMode,
    local_s: np.ndarray | float,
    edge_index: int,
    defender_alphas: tuple[float, float],
    defender_data: list[dict[str, Any]],
) -> np.ndarray:
    values = np.asarray(local_s, dtype=float)
    alpha = defender_alphas[mode.defender_index]
    metric: BoundaryHSPMetric = defender_data[mode.defender_index]["hsp_metric"]
    distance, _ = metric.evaluate(edge_index, values, mode.lift)
    return alpha * np.asarray(distance, dtype=float)


def _mode_radius_derivative(
    mode: _ArrivalMode,
    local_s: np.ndarray,
    edge_index: int,
    defender_alphas: tuple[float, float],
    defender_data: list[dict[str, Any]],
) -> np.ndarray:
    values = np.asarray(local_s, dtype=float)
    alpha = defender_alphas[mode.defender_index]
    metric: BoundaryHSPMetric = defender_data[mode.defender_index]["hsp_metric"]
    _, derivative = metric.evaluate(edge_index, values, mode.lift)
    return alpha * derivative


def _sample_ccw_arc(
    center: np.ndarray,
    radius: float,
    first_direction: np.ndarray,
    second_direction: np.ndarray,
    angular_resolution: int,
    tol: float,
) -> tuple[np.ndarray, float]:
    first_direction = first_direction / max(float(np.linalg.norm(first_direction)), tol)
    second_direction = second_direction / max(
        float(np.linalg.norm(second_direction)), tol
    )
    first_angle = atan2(first_direction[1], first_direction[0])
    second_angle = atan2(second_direction[1], second_direction[0])
    sweep = (second_angle - first_angle) % TAU
    if abs(sweep - TAU) <= 128.0 * tol or sweep <= 128.0 * tol:
        sweep = 0.0
    interval_count = max(1, int(ceil(sweep * angular_resolution / TAU)))
    angles = np.linspace(first_angle, first_angle + sweep, interval_count + 1)
    points = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    points[0] = center + radius * first_direction
    points[-1] = center + radius * second_direction
    return points, sweep


def _lift_mode_for_defender(
    defender_index: int,
    lift: int,
    edge_index: int,
    local_s: float,
    defender_alphas: tuple[float, float],
    defender_data: list[dict[str, Any]],
) -> tuple[float, _ArrivalMode]:
    mode = _ArrivalMode(defender_index, lift)
    radius = float(
        _mode_radius(
            mode,
            local_s,
            edge_index,
            defender_alphas,
            defender_data,
        )
    )
    return radius, mode


def _defender_radius_difference(
    edge_index: int,
    local_s: float,
    defender_alphas: tuple[float, float],
    defender_data: list[dict[str, Any]],
    lift_pair: tuple[int, int],
) -> float:
    first, _ = _lift_mode_for_defender(
        0,
        lift_pair[0],
        edge_index,
        local_s,
        defender_alphas,
        defender_data,
    )
    second, _ = _lift_mode_for_defender(
        1,
        lift_pair[1],
        edge_index,
        local_s,
        defender_alphas,
        defender_data,
    )
    return first - second


def _bisect_defender_switch(
    edge_index: int,
    lo: float,
    hi: float,
    defender_alphas: tuple[float, float],
    defender_data: list[dict[str, Any]],
    lift_pair: tuple[int, int],
) -> float:
    def value(local_s: float) -> float:
        return _defender_radius_difference(
            edge_index,
            local_s,
            defender_alphas,
            defender_data,
            lift_pair,
        )

    flo = value(lo)
    fhi = value(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    for _ in range(60):
        middle = 0.5 * (lo + hi)
        fmid = value(middle)
        if abs(fmid) <= 2e-13 * max(1.0, abs(flo), abs(fhi)):
            return middle
        if flo * fmid <= 0.0:
            hi, fhi = middle, fmid
        else:
            lo, flo = middle, fmid
    return 0.5 * (lo + hi)


def _mode_radius_direction(
    mode: _ArrivalMode,
    edge_index: int,
    local_s: float,
    defender_alphas: tuple[float, float],
    defender_data: list[dict[str, Any]],
    tangents: np.ndarray,
    normals: np.ndarray,
    tol: float,
) -> np.ndarray:
    derivative = float(
        _mode_radius_derivative(
            mode,
            np.array([local_s]),
            edge_index,
            defender_alphas,
            defender_data,
        )[0]
    )
    tangential = -float(np.clip(derivative, -1.0, 1.0))
    normal = float(np.sqrt(max(0.0, 1.0 - tangential * tangential)))
    direction = tangential * tangents[edge_index] + normal * normals[edge_index]
    return direction / max(float(np.linalg.norm(direction)), tol)


def _find_p_items(
    polygon: np.ndarray,
    defenders: np.ndarray,
    defender_alphas: tuple[float, float],
    defender_data: list[dict[str, Any]],
    lengths: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    samples_per_edge: int,
    angular_resolution: int,
    tol: float,
) -> list[dict[str, Any]]:
    """Find fixed-lift equal-arrival points for all four HSP pairs."""

    roots: list[tuple[tuple[int, int], int, float, np.ndarray]] = []
    value_scale = max(
        1.0,
        float(
            np.max(np.linalg.norm(defenders[:, None, :] - polygon[None, :, :], axis=2))
        ),
    )
    value_tolerance = 64.0 * tol * value_scale
    metrics = [data["hsp_metric"] for data in defender_data]

    def append_root(
        lift_pair: tuple[int, int], edge_index: int, local_s: float
    ) -> None:
        length = float(lengths[edge_index])
        if local_s >= length - 128.0 * tol * max(1.0, length):
            edge_index = (edge_index + 1) % len(polygon)
            local_s = 0.0
        point = polygon[edge_index] + local_s * tangents[edge_index]
        roots.append((lift_pair, edge_index, local_s, point))

    for lift_pair in LIFT_PAIRS:
        for edge_index, length_value in enumerate(lengths):
            length = float(length_value)
            cuts = [0.0, length]
            for metric in metrics:
                if metric.seam_edge == edge_index:
                    cuts.append(float(np.clip(metric.seam_s, 0.0, length)))
            cuts = sorted(set(cuts))

            def difference(local_s: float) -> float:
                return _defender_radius_difference(
                    edge_index,
                    local_s,
                    defender_alphas,
                    defender_data,
                    lift_pair,
                )

            for cut in cuts:
                if cut >= length - 128.0 * tol * max(1.0, length):
                    continue
                if abs(difference(cut)) <= value_tolerance:
                    append_root(lift_pair, edge_index, cut)

            for interval_start, interval_end in zip(cuts[:-1], cuts[1:]):
                width = interval_end - interval_start
                if width <= 128.0 * tol * max(1.0, length):
                    continue
                edge_offset = min(
                    0.25 * width,
                    256.0 * tol * max(1.0, length),
                )
                lo, hi = interval_start + edge_offset, interval_end - edge_offset
                probe_count = max(
                    12,
                    int(np.ceil(2 * samples_per_edge * width / max(length, tol))),
                )
                probes = np.linspace(lo, hi, probe_count + 1)
                values = np.array([difference(float(value)) for value in probes])
                if float(np.max(np.abs(values))) <= value_tolerance:
                    continue
                for index in range(probe_count):
                    left, right = float(values[index]), float(values[index + 1])
                    if abs(left) <= value_tolerance:
                        root = float(probes[index])
                    elif left * right < 0.0:
                        root = _bisect_defender_switch(
                            edge_index,
                            float(probes[index]),
                            float(probes[index + 1]),
                            defender_alphas,
                            defender_data,
                            lift_pair,
                        )
                    else:
                        continue
                    append_root(lift_pair, edge_index, root)

    position_scale = max(1.0, float(np.max(lengths)))
    unique_roots: list[tuple[tuple[int, int], int, float, np.ndarray]] = []
    for root in roots:
        if not any(
            root[0] == existing[0]
            and np.linalg.norm(root[3] - existing[3]) <= 128.0 * tol * position_scale
            for existing in unique_roots
        ):
            unique_roots.append(root)

    p_items: list[dict[str, Any]] = []
    for lift_pair, edge_index, local_s, point in unique_roots:
        length = float(lengths[edge_index])
        endpoint_tolerance = 128.0 * tol * max(1.0, length)
        if local_s <= endpoint_tolerance:
            before_edge = (edge_index - 1) % len(polygon)
            before_s = float(lengths[before_edge])
            after_edge = edge_index
            after_s = 0.0
        elif local_s >= length - endpoint_tolerance:
            before_edge = edge_index
            before_s = length
            after_edge = (edge_index + 1) % len(polygon)
            after_s = 0.0
        else:
            before_edge = after_edge = edge_index
            before_s = after_s = local_s

        probe_step = 1e-6 * max(1.0, float(np.max(lengths)))
        before_probe = max(0.0, before_s - probe_step)
        after_probe = min(float(lengths[after_edge]), after_s + probe_step)
        before_value = _defender_radius_difference(
            before_edge,
            before_probe,
            defender_alphas,
            defender_data,
            lift_pair,
        )
        after_value = _defender_radius_difference(
            after_edge,
            after_probe,
            defender_alphas,
            defender_data,
            lift_pair,
        )
        before_defender = 0 if before_value < 0.0 else 1
        after_defender = 0 if after_value < 0.0 else 1
        if before_defender == after_defender:
            continue

        before_radius, before_mode = _lift_mode_for_defender(
            before_defender,
            lift_pair[before_defender],
            before_edge,
            before_s,
            defender_alphas,
            defender_data,
        )
        after_radius, after_mode = _lift_mode_for_defender(
            after_defender,
            lift_pair[after_defender],
            after_edge,
            after_s,
            defender_alphas,
            defender_data,
        )
        radius = 0.5 * (before_radius + after_radius)
        first_direction = _mode_radius_direction(
            before_mode,
            before_edge,
            before_s,
            defender_alphas,
            defender_data,
            tangents,
            normals,
            tol,
        )
        second_direction = _mode_radius_direction(
            after_mode,
            after_edge,
            after_s,
            defender_alphas,
            defender_data,
            tangents,
            normals,
            tol,
        )
        arc, sweep = _sample_ccw_arc(
            point,
            radius,
            first_direction,
            second_direction,
            angular_resolution,
            tol,
        )
        if sweep <= 128.0 * tol:
            continue
        label = f"p{len(p_items) + 1}"
        before_source = f"D{before_defender + 1}"
        after_source = f"D{after_defender + 1}"
        p_items.append(
            {
                "label": label,
                "point": point.copy(),
                "radius": radius,
                "arc_points": arc,
                "sources": (before_source, after_source),
                "tangent_points": {
                    before_source: arc[0].copy(),
                    after_source: arc[-1].copy(),
                },
                "color": P_ARC_COLORS[len(p_items) % len(P_ARC_COLORS)],
                "edge_index": edge_index,
                "local_s": local_s,
                "lift_pair": lift_pair,
            }
        )
    return p_items


def _flatten_source_segments(
    pieces: list[dict[str, Any]],
    source_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    source_lookup = {name: index for index, name in enumerate(source_names)}
    for piece in pieces:
        points = np.asarray(piece["points"], dtype=float)
        if len(points) < 2:
            continue
        delta = np.diff(points, axis=0)
        nonzero = np.linalg.norm(delta, axis=1) > 1e-14
        starts.append(points[:-1][nonzero])
        vectors.append(delta[nonzero])
        sources.append(
            np.full(
                int(np.count_nonzero(nonzero)),
                source_lookup[piece["source"]],
                dtype=int,
            )
        )
    if not starts:
        raise RuntimeError("The cooperative candidate envelope has no segments.")
    return np.vstack(starts), np.vstack(vectors), np.concatenate(sources)


def _ray_segment_radii(
    starts: np.ndarray,
    vectors: np.ndarray,
    reference: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    relative_starts = starts - reference
    denominator = vectors[:, 0] * direction[1] - vectors[:, 1] * direction[0]
    numerator = -(
        relative_starts[:, 0] * direction[1] - relative_starts[:, 1] * direction[0]
    )
    scale = max(1.0, float(np.max(np.linalg.norm(relative_starts, axis=1))))
    usable = np.abs(denominator) > 2e-14 * scale
    fraction = np.zeros_like(denominator)
    fraction[usable] = numerator[usable] / denominator[usable]
    hits = relative_starts + fraction[:, None] * vectors
    radius = hits @ direction
    valid = usable & (fraction >= -1e-10) & (fraction <= 1.0 + 1e-10)
    return np.where(valid, radius, np.inf)


def _source_transition_point(
    source_segments: list[tuple[np.ndarray, np.ndarray]],
    source1: int,
    source2: int,
    angle1: float,
    angle2: float,
    reference: np.ndarray,
    target_segments: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    def radii(angle: float) -> tuple[float, float]:
        direction = np.array([np.cos(angle), np.sin(angle)])
        target_hits = _ray_segment_radii(*target_segments, reference, direction)
        target_radius = float(np.min(target_hits[target_hits >= -1e-10]))

        answers = []
        for source in (source1, source2):
            hits = _ray_segment_radii(*source_segments[source], reference, direction)
            hits = hits[hits >= target_radius - 2e-9 * max(1.0, target_radius)]
            answers.append(float(np.min(hits)) if len(hits) else float("inf"))
        return answers[0], answers[1]

    samples = []
    for angle in np.linspace(angle1, angle2, 17):
        first, second = radii(float(angle))
        if np.isfinite(first) and np.isfinite(second):
            samples.append((float(angle), first - second, first, second))
    if not samples:
        angle = 0.5 * (angle1 + angle2)
        direction = np.array([np.cos(angle), np.sin(angle)])
        return reference + direction

    bracket = None
    for left, right in zip(samples[:-1], samples[1:]):
        if left[1] == 0.0 or left[1] * right[1] <= 0.0:
            bracket = (left[0], right[0])
            break
    if bracket is None:
        angle, _, first, second = min(samples, key=lambda item: abs(item[1]))
    else:
        lo, hi = bracket
        flo = radii(lo)[0] - radii(lo)[1]
        for _ in range(55):
            middle = 0.5 * (lo + hi)
            first, second = radii(middle)
            value = first - second
            if not np.isfinite(value):
                break
            if abs(value) <= 2e-12 * max(1.0, first, second):
                lo = hi = middle
                break
            if flo * value <= 0.0:
                hi = middle
            else:
                lo, flo = middle, value
        angle = 0.5 * (lo + hi)
        first, second = radii(angle)
    radius = 0.5 * (first + second)
    return reference + radius * np.array([np.cos(angle), np.sin(angle)])


def _closest_radial_envelope(
    polygon: np.ndarray,
    candidate_pieces: list[dict[str, Any]],
    source_names: list[str],
    resolution: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep the closest target-exterior candidate on every interior ray."""

    reference = np.mean(polygon, axis=0)
    starts, vectors, segment_sources = _flatten_source_segments(
        candidate_pieces, source_names
    )
    target_starts = polygon
    target_vectors = np.roll(polygon, -1, axis=0) - polygon
    target_segments = (target_starts, target_vectors)
    source_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for source_index in range(len(source_names)):
        mask = segment_sources == source_index
        source_segments.append((starts[mask], vectors[mask]))

    count = max(720, 4 * int(resolution))
    angles = np.linspace(0.0, TAU, count, endpoint=False)
    envelope = np.empty((count, 2), dtype=float)
    active_sources = np.full(count, -1, dtype=int)
    for index, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        target_hits = _ray_segment_radii(
            target_starts, target_vectors, reference, direction
        )
        positive_target_hits = target_hits[target_hits >= -1e-10]
        if not len(positive_target_hits):
            raise RuntimeError("A reference ray did not leave the convex target.")
        target_radius = float(np.min(positive_target_hits))
        hits = _ray_segment_radii(starts, vectors, reference, direction)
        valid = hits >= target_radius - 2e-9 * max(1.0, target_radius)
        if not np.any(valid):
            raise RuntimeError("The cooperative candidate envelope has an angular gap.")
        segment_index = int(np.argmin(np.where(valid, hits, np.inf)))
        envelope[index] = reference + hits[segment_index] * direction
        active_sources[index] = segment_sources[segment_index]

    angle_step = TAU / count
    change_edges = [
        index
        for index in range(count)
        if active_sources[index] != active_sources[(index + 1) % count]
    ]
    if not change_edges:
        closed = np.vstack((envelope, envelope[0]))
        return closed, [
            {
                "kind": "visible_run",
                "source": source_names[int(active_sources[0])],
                "points": closed,
            }
        ]

    transitions: dict[int, np.ndarray] = {}
    for edge_index in change_edges:
        first_source = int(active_sources[edge_index])
        second_source = int(active_sources[(edge_index + 1) % count])
        transitions[edge_index] = _source_transition_point(
            source_segments,
            first_source,
            second_source,
            edge_index * angle_step,
            (edge_index + 1) * angle_step,
            reference,
            target_segments,
        )

    runs: list[dict[str, Any]] = []
    for position, previous_change in enumerate(change_edges):
        next_change = change_edges[(position + 1) % len(change_edges)]
        start_index = (previous_change + 1) % count
        indices = [start_index]
        while indices[-1] != next_change:
            indices.append((indices[-1] + 1) % count)
        source_index = int(active_sources[start_index])
        points = np.vstack(
            (
                transitions[previous_change],
                envelope[np.asarray(indices, dtype=int)],
                transitions[next_change],
            )
        )
        runs.append(
            {
                "kind": "visible_run",
                "source": source_names[source_index],
                "points": points,
            }
        )
    joined = [runs[0]["points"]]
    joined.extend(run["points"][1:] for run in runs[1:])
    closed = np.vstack(joined)
    closed[-1] = closed[0]
    return closed, runs


def compute_geometry_scene(
    D1: Any,
    D2: Any,
    vertices: Any,
    alpha: float = ALPHA,
    beta: float = BETA,
    samples_per_edge: int = 140,
    grid_size: int = 400,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Compute two individual barriers and the cooperative polygon barrier."""

    start_time = time.perf_counter()
    defenders = np.asarray([D1, D2], dtype=float)
    if defenders.shape != (2, 2) or not np.all(np.isfinite(defenders)):
        raise ValueError("D1 and D2 must be finite two-dimensional points.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha=v_A/v_D2 must satisfy 0 < alpha < 1.")
    if not beta > 0.0 or alpha * beta > 1.0:
        raise ValueError("require beta=v_D2/v_D1>0 and v_A/v_D1=alpha*beta<=1")
    if samples_per_edge < 20 or grid_size < 150:
        raise ValueError("Use samples_per_edge >= 20 and grid_size >= 150.")

    polygon = _prepare_polygon(vertices, tol)
    if any(_strictly_inside_convex(defender, polygon, tol) for defender in defenders):
        raise ValueError("Defenders must not lie inside the target polygon.")
    lengths, tangents, normals = _polygon_geometry(polygon)
    defender_alphas = (float(alpha * beta), float(alpha))
    defender_data = [
        _defender_geometry(defender, polygon, lengths, tangents, normals, tol)
        for defender in defenders
    ]

    angular_resolution = max(720, 4 * int(grid_size))
    individual_barriers = {
        "D1": compute_barrier(
            defenders[0],
            polygon,
            defender_alphas[0],
            samples_per_edge=samples_per_edge,
            grid_size=grid_size,
            tol=tol,
        ),
        "D2": compute_barrier(
            defenders[1],
            polygon,
            defender_alphas[1],
            samples_per_edge=samples_per_edge,
            grid_size=grid_size,
            tol=tol,
        ),
    }
    p_items = _find_p_items(
        polygon,
        defenders,
        defender_alphas,
        defender_data,
        lengths,
        tangents,
        normals,
        samples_per_edge,
        angular_resolution,
        tol,
    )
    candidate_pieces: list[dict[str, Any]] = [
        {
            "kind": "individual_barrier",
            "source": source,
            "points": result.points,
        }
        for source, result in individual_barriers.items()
    ]
    candidate_pieces.extend(
        {
            "kind": "p_arc",
            "source": item["label"],
            "points": item["arc_points"],
        }
        for item in p_items
    )
    source_names = ["D1", "D2", *(item["label"] for item in p_items)]
    closed_barrier, barrier_pieces = _closest_radial_envelope(
        polygon,
        candidate_pieces,
        source_names,
        grid_size,
    )
    visible_sources = {piece["source"] for piece in barrier_pieces}
    for item in p_items:
        item["on_envelope"] = item["label"] in visible_sources

    finite = all(
        np.all(np.isfinite(points))
        for points in [
            closed_barrier,
            *(piece["points"] for piece in barrier_pieces),
            *(result.points for result in individual_barriers.values()),
        ]
    )
    closure_error = float(np.linalg.norm(closed_barrier[-1] - closed_barrier[0]))
    return {
        "polygon": polygon,
        "defenders": defenders.copy(),
        "defender_alphas": {"D1": defender_alphas[0], "D2": defender_alphas[1]},
        "individual_barriers": individual_barriers,
        "closed_barrier": closed_barrier,
        "candidate_pieces": candidate_pieces,
        "barrier_pieces": barrier_pieces,
        "p_items": p_items,
        "lift_pairs": LIFT_PAIRS,
        "visible_edges": {
            "D1": defender_data[0]["visible_edges"].copy(),
            "D2": defender_data[1]["visible_edges"].copy(),
        },
        "finite": finite,
        "closure_error": closure_error,
        "alpha": float(alpha),
        "beta": float(beta),
        "samples_per_edge": int(samples_per_edge),
        "grid_size": int(grid_size),
        "computation_time": time.perf_counter() - start_time,
    }
