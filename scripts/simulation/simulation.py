"""Stateful browser simulation for the current 2D1A equilibrium strategy."""

from typing import Any

import numpy as np

from barrier_classifier import project_attacker_outside_barrier
from circle_barrier import compute_geometry_scene as circle_scene
from equilibrium_strategy import solve_equilibrium
from otp_solver import advance_dsp


PLAYERS = ("A", "D1", "D2")


def _entry_fraction(start: np.ndarray, end: np.ndarray, radius: float) -> float | None:
    delta = end - start
    a = float(delta @ delta)
    if a == 0.0:
        return None
    b = 2.0 * float(start @ delta)
    c = float(start @ start) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    roots = [
        root
        for root in (
            (-b - np.sqrt(discriminant)) / (2.0 * a),
            (-b + np.sqrt(discriminant)) / (2.0 * a),
        )
        if -1e-12 <= root <= 1.0 + 1e-12
    ]
    return float(np.clip(min(roots), 0.0, 1.0)) if roots else None


def _capture_fraction(
    attacker_start: np.ndarray,
    attacker_end: np.ndarray,
    defender_start: np.ndarray,
    defender_end: np.ndarray,
    tolerance: float,
) -> float | None:
    relative = attacker_start - defender_start
    change = (attacker_end - attacker_start) - (defender_end - defender_start)
    c = float(relative @ relative) - tolerance * tolerance
    if c <= 0.0:
        return 0.0
    a = float(change @ change)
    if a == 0.0:
        return None
    b = 2.0 * float(relative @ change)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    roots = [
        root
        for root in (
            (-b - np.sqrt(discriminant)) / (2.0 * a),
            (-b + np.sqrt(discriminant)) / (2.0 * a),
        )
        if -1e-12 <= root <= 1.0 + 1e-12
    ]
    return float(np.clip(min(roots), 0.0, 1.0)) if roots else None


def _advance_manual_defender(
    start: np.ndarray,
    heading: float,
    distance: float,
    radius: float,
    fallback_direction: int,
) -> np.ndarray:
    """Advance a manual defender, redirecting blocked motion to a tangent."""
    direction = np.array([np.cos(heading), np.sin(heading)])
    endpoint = start + distance * direction
    contact = _entry_fraction(start, endpoint, radius)
    if contact is None:
        return endpoint

    contact_point = start + contact * (endpoint - start)
    if float(contact_point @ direction) >= -1e-12:
        return endpoint

    remaining_distance = distance * (1.0 - contact)
    radial = contact_point / radius
    ccw_tangent = np.array([-radial[1], radial[0]])
    tangent_component = float(direction @ ccw_tangent)
    if abs(tangent_component) <= 1e-12:
        tangent_sign = 1 if fallback_direction >= 0 else -1
    else:
        tangent_sign = 1 if tangent_component > 0.0 else -1
    return contact_point + tangent_sign * remaining_distance * ccw_tangent


class EquilibriumSimulation:
    """One equilibrium solve per step, with optional live barrier updates."""

    def __init__(self, config: dict[str, Any]):
        defender_speed = float(config.get("v_D", 0.3))
        self.params = {
            "R": float(config.get("R", 1.0)),
            "v_A": float(config.get("v_A", 0.2)),
            "v_D1": float(config.get("v_D1", defender_speed)),
            "v_D2": float(config.get("v_D2", defender_speed)),
            "constraint_tol": float(config.get("constraint_tol", 1e-7)),
        }
        self.t_max = float(config.get("t_max", 15.0))
        self.goal_tolerance = float(config.get("goal_tolerance", 1e-3))
        self.capture_tolerance = float(config.get("capture_tolerance", 1e-2))
        self.barrier_resolution = max(160, int(config.get("barrier_resolution", 180)))
        values = [
            *self.params.values(),
            self.t_max,
            self.goal_tolerance,
            self.capture_tolerance,
        ]
        if not np.all(np.isfinite(values)) or min(values) <= 0.0:
            raise ValueError("radius, speeds, time, and tolerances must be positive")
        if self.params["v_A"] > min(self.params["v_D1"], self.params["v_D2"]) * (
            1.0 + 1e-12
        ):
            raise ValueError("the equilibrium solver requires v_A <= min(v_D1, v_D2)")

        defenders = {}
        for player in ("D1", "D2"):
            point = np.asarray(config[player], dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ValueError(f"{player} must be a finite 2-vector")
            if np.any(np.abs(point) > 10.0 + 1e-12):
                raise ValueError(f"{player} coordinates must be in [-10, 10]")
            if np.linalg.norm(point) < self.params["R"] - 1e-12:
                raise ValueError(f"{player} starts inside the circular goal")
            defenders[player] = point.copy()
        requested_attacker = np.asarray(config["A"], dtype=float)
        if requested_attacker.shape != (2,) or not np.all(
            np.isfinite(requested_attacker)
        ):
            raise ValueError("A must be a finite 2-vector")
        if np.any(np.abs(requested_attacker) > 10.0 + 1e-12):
            raise ValueError("A coordinates must be in [-10, 10]")
        constraint = project_attacker_outside_barrier(
            defenders["D1"],
            defenders["D2"],
            self.params["v_D1"],
            self.params["v_D2"],
            requested_attacker,
            self.params["v_A"],
            self.params["R"],
        )
        constrained_attacker = np.asarray(constraint["point"], dtype=float)
        if np.any(np.abs(constrained_attacker) > 10.0 + 1e-12):
            raise ValueError(
                "no attacker position satisfies both the D1/D2 barrier and "
                "the [-10, 10] coordinate limits on this ray"
            )
        self.state = {"A": constrained_attacker, **defenders}
        self.attacker_constrained = bool(constraint["constrained"])
        self.constraint_source = str(constraint["active_source"])
        self.constraint_barrier_radius = float(constraint["barrier_radius"])

        self.time = 0.0
        self.terminated = False
        self.result = None
        self.otp = None
        self.otp_lift = None
        self.sigma = None
        self.lift_pair = None
        self.path_lifts = None
        self.stationary_mask = (False, False)
        self.value = None
        self.mode = None
        self.selected_branch = None
        self.active_defenders: tuple[str, ...] = ()
        self.headings = {player: None for player in PLAYERS}
        self.manual_player = None
        self.manual_heading = None
        self.manual_speed = 0.0
        self.trajectory = {player: [self.state[player].copy()] for player in PLAYERS}
        self.times = [0.0]
        self.values: list[float] = []
        self.value_times: list[float] = []
        supplied_barrier = config.get("_initial_barrier")
        self.barrier = (
            self._current_barrier()
            if supplied_barrier is None
            else np.asarray(supplied_barrier, dtype=float).copy()
        )
        if self.barrier.ndim != 2 or self.barrier.shape[1] != 2:
            raise ValueError("_initial_barrier must have shape (n, 2)")
        self.barrier_reused = bool(config.get("_barrier_reused", False))
        self.barrier_revision = 0
        self.barrier_is_live = False
        self.barrier_defenders = {
            player: self.state[player].copy() for player in ("D1", "D2")
        }

    def _current_barrier(self) -> np.ndarray:
        radius = self.params["R"]
        defenders = [self.state["D1"].copy(), self.state["D2"].copy()]
        for defender in defenders:
            norm = float(np.linalg.norm(defender))
            if norm <= radius:
                defender *= (radius + 1e-8) / max(norm, 1e-14)
        scene = circle_scene(
            defenders[0],
            defenders[1],
            radius,
            self.params["v_A"] / self.params["v_D2"],
            self.barrier_resolution,
            self.params["v_D2"] / self.params["v_D1"],
        )
        return np.asarray(scene["closed_barrier"], dtype=float)

    def _finish(self, result: str) -> None:
        self.terminated = True
        self.result = result

    def step(
        self,
        dt: float,
        update_barrier: bool = False,
        manual_player: str | None = None,
        manual_heading: float | None = None,
        manual_speed: float = 1.0,
    ) -> dict[str, Any]:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive")
        if manual_player is not None and manual_player not in PLAYERS:
            raise ValueError("manual_player must be A, D1, D2, or None")
        if manual_heading is not None:
            manual_heading = float(manual_heading)
            if not np.isfinite(manual_heading):
                raise ValueError("manual_heading must be finite or None")
            if manual_player is None:
                raise ValueError("manual_heading requires manual_player")
        manual_speed = float(manual_speed)
        if not np.isfinite(manual_speed) or not 0.0 <= manual_speed <= 1.0:
            raise ValueError("manual_speed must be between 0 and 1")
        self.manual_player = manual_player
        self.manual_heading = manual_heading
        self.manual_speed = manual_speed if manual_heading is not None else 0.0
        if self.terminated:
            return self.snapshot()
        for defender in ("D1", "D2"):
            if (
                np.linalg.norm(self.state["A"] - self.state[defender])
                <= self.capture_tolerance
            ):
                self._finish(f"Captured by {defender}")
                return self.snapshot()
        if np.linalg.norm(self.state["A"]) <= self.params["R"] + self.goal_tolerance:
            self._finish("Attacker reaches target")
            return self.snapshot()
        if self.time >= self.t_max - 1e-12:
            self._finish("Time limit reached")
            return self.snapshot()

        solution = solve_equilibrium(self.state, self.params)
        if not solution["success"]:
            self._finish("Solver failure")
            return self.snapshot()
        self.otp = np.asarray(solution["otp"], dtype=float).copy()
        self.otp_lift = np.asarray(solution["otp_lift"], dtype=float).copy()
        self.sigma = tuple(int(value) for value in solution["sigma"])
        self.lift_pair = tuple(int(value) for value in solution["lift_pair"])
        self.path_lifts = {
            player: tuple(float(value) for value in solution["path_lifts"][player])
            for player in PLAYERS
        }
        self.stationary_mask = tuple(
            bool(value) for value in solution["stationary_mask"]
        )
        self.value = float(solution["value"])
        self.selected_branch = str(solution["branch_id"])
        selected = solution["selected_candidate"]
        self.active_defenders = tuple(str(value) for value in selected["active"])
        self.mode = (
            f"{self.active_defenders[0]} only"
            if len(self.active_defenders) == 1
            else "cooperative"
        )
        self.headings = {
            "A": float(solution["phi_A"]),
            "D1": float(solution["phi_D1"]),
            "D2": float(solution["phi_D2"]),
        }
        if manual_player is not None:
            self.headings[manual_player] = manual_heading
        self.value_times.append(float(self.time))
        self.values.append(self.value)

        step_time = min(dt, self.t_max - self.time)
        attack_distance = float(np.linalg.norm(self.otp - self.state["A"]))
        if manual_player != "A" and attack_distance > 1e-14:
            step_time = min(step_time, attack_distance / self.params["v_A"])
        starts = {player: self.state[player].copy() for player in PLAYERS}

        def advance(player: str, elapsed: float) -> np.ndarray:
            if player == manual_player:
                if manual_heading is None:
                    return starts[player].copy()
                speed = (
                    self.params["v_A" if player == "A" else f"v_{player}"]
                    * self.manual_speed
                )
                distance = speed * elapsed
                if player == "A":
                    return starts[player] + distance * np.array(
                        [np.cos(manual_heading), np.sin(manual_heading)]
                    )
                index = 0 if player == "D1" else 1
                return _advance_manual_defender(
                    starts[player],
                    manual_heading,
                    distance,
                    self.params["R"],
                    self.sigma[index],
                )
            if player == "A":
                direction = 1
                speed = self.params["v_A"]
            else:
                index = 0 if player == "D1" else 1
                if self.stationary_mask[index]:
                    return starts[player].copy()
                direction = self.sigma[index]
                speed = self.params[f"v_{player}"]
            return advance_dsp(
                starts[player],
                self.otp,
                self.params["R"],
                direction,
                speed * elapsed,
                lift_angles=self.path_lifts[player],
            )

        ends = {player: advance(player, step_time) for player in PLAYERS}
        events = []
        for index, defender in enumerate(("D1", "D2"), 1):
            fraction = _capture_fraction(
                starts["A"],
                ends["A"],
                starts[defender],
                ends[defender],
                self.capture_tolerance,
            )
            if fraction is not None:
                events.append((fraction, 0, f"Captured by D{index}"))
        fraction = _entry_fraction(starts["A"], ends["A"], self.params["R"])
        if fraction is not None:
            events.append((fraction, 1, "Attacker reaches target"))
        event_fraction, _, event = min(events, default=(1.0, 3, None))
        actual_time = step_time * event_fraction
        self.state = {player: advance(player, actual_time) for player in PLAYERS}
        self.time += actual_time
        self.times.append(float(self.time))
        for player in PLAYERS:
            self.trajectory[player].append(self.state[player].copy())
        if update_barrier:
            self.barrier = self._current_barrier()
            self.barrier_reused = False
            self.barrier_revision += 1
            self.barrier_is_live = True
            self.barrier_defenders = {
                player: self.state[player].copy() for player in ("D1", "D2")
            }
        else:
            self.barrier_is_live = False
        if event is not None:
            self._finish(event)
        elif self.time >= self.t_max - 1e-12:
            self._finish("Time limit reached")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "time": float(self.time),
            **{player: self.state[player].tolist() for player in PLAYERS},
            "otp": None if self.otp is None else self.otp.tolist(),
            "otp_lift": None if self.otp_lift is None else self.otp_lift.tolist(),
            "sigma": None if self.sigma is None else list(self.sigma),
            "lift_pair": None if self.lift_pair is None else list(self.lift_pair),
            "path_lifts": (
                None
                if self.path_lifts is None
                else {player: list(path) for player, path in self.path_lifts.items()}
            ),
            "stationary_mask": list(self.stationary_mask),
            "V": self.value,
            "mode": self.mode,
            "selected_branch": self.selected_branch,
            "active_defenders": list(self.active_defenders),
            "headings": dict(self.headings),
            "manual_player": self.manual_player,
            "manual_heading": self.manual_heading,
            "manual_speed": self.manual_speed,
            "terminated": bool(self.terminated),
            "result": self.result,
            "attacker_constrained": self.attacker_constrained,
            "constraint_source": self.constraint_source,
            "constraint_barrier_radius": self.constraint_barrier_radius,
            "barrier": self.barrier.tolist(),
            "barrier_reused": self.barrier_reused,
            "barrier_revision": self.barrier_revision,
            "barrier_is_live": self.barrier_is_live,
            "barrier_defenders": {
                player: point.tolist()
                for player, point in self.barrier_defenders.items()
            },
            "trajectory": {
                player: [point.tolist() for point in self.trajectory[player]]
                for player in PLAYERS
            },
            "times": list(self.times),
            "value_times": list(self.value_times),
            "value_history": list(self.values),
        }
