"""Small JSON-in/JSON-out interface used directly by app.js."""

import json
from typing import Any

import numpy as np

from barrier_classifier import project_attacker_outside_barrier
from circle_barrier import compute_geometry_scene as circle_scene
from polygon_barrier import compute_geometry_scene as polygon_scene
from simulation import EquilibriumSimulation


_simulation: EquilibriumSimulation | None = None
_simulation_barrier_key: tuple[Any, ...] | None = None
_simulation_barrier: list[list[float]] | None = None
COORDINATE_LIMIT = 10.0


def _config(value: str | dict[str, Any]) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value


def _json(value: Any) -> str:
    def default(item: Any):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, tuple):
            return list(item)
        raise TypeError(type(item).__name__)

    return json.dumps(value, default=default, allow_nan=False)


def _player_point(value: Any, name: str) -> list[float]:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite 2-vector")
    if np.any(np.abs(point) > COORDINATE_LIMIT + 1e-12):
        raise ValueError(f"{name} coordinates must be in [-10, 10]")
    return point.tolist()


def compute_circle_barrier(config: str | dict[str, Any]) -> str:
    c = dict(_config(config))
    c["D1"] = _player_point(c["D1"], "D1")
    c["D2"] = _player_point(c["D2"], "D2")
    scene = circle_scene(
        c["D1"],
        c["D2"],
        float(c["R"]),
        float(c["alpha"]),
        int(c.get("resolution", 220)),
        float(c.get("beta", 1.0)),
    )
    return _json(
        {
            "type": "circle",
            "segments": [
                {
                    "kind": "envelope",
                    "source": segment["source"],
                    "points": segment["points"],
                }
                for segment in scene["envelope_segments"]
            ],
            "barrier": scene["closed_barrier"],
            "individual": scene["barriers"],
            "p_items": [
                {
                    "label": item["label"],
                    "point": item["point"],
                    "mode": item["mode"],
                    "radius": item["circle_radius"],
                    "arc": item["arc_points"],
                    "tangencies": item["tangent_points"],
                }
                for item in scene["p_items"]
            ],
            "residual": scene["residual"],
            "computation_time": scene["computation_time"],
        }
    )


def compute_polygon_barrier(config: str | dict[str, Any]) -> str:
    c = dict(_config(config))
    c["D1"] = _player_point(c["D1"], "D1")
    c["D2"] = _player_point(c["D2"], "D2")
    scene = polygon_scene(
        c["D1"],
        c["D2"],
        c["vertices"],
        float(c["alpha"]),
        float(c.get("beta", 1.0)),
        int(c.get("samples_per_edge", 45)),
        int(c.get("grid_size", 150)),
    )
    return _json(
        {
            "type": "polygon",
            "polygon": scene["polygon"],
            "segments": [
                {
                    "kind": "envelope",
                    "source": piece["source"],
                    "points": piece["points"],
                }
                for piece in scene["barrier_pieces"]
            ],
            "barrier": scene["closed_barrier"],
            "individual": {
                name: result.points
                for name, result in scene["individual_barriers"].items()
            },
            "p_items": [
                {
                    "label": item["label"],
                    "point": item["point"],
                    "radius": item["radius"],
                    "arc": item["arc_points"],
                    "sources": item["sources"],
                    "lift_pair": item["lift_pair"],
                }
                for item in scene["p_items"]
            ],
            "visible_edges": scene["visible_edges"],
            "closure_error": scene["closure_error"],
            "computation_time": scene["computation_time"],
        }
    )


def constrain_attacker_position(config: str | dict[str, Any]) -> str:
    c = dict(_config(config))
    for player in ("A", "D1", "D2"):
        c[player] = _player_point(c[player], player)
    speed = float(c.get("v_D", 0.3))
    result = project_attacker_outside_barrier(
        c["D1"],
        c["D2"],
        float(c.get("v_D1", speed)),
        float(c.get("v_D2", speed)),
        c["A"],
        float(c["v_A"]),
        float(c["R"]),
    )
    return _json(result)


def init_simulation(config: str | dict[str, Any]) -> str:
    global _simulation, _simulation_barrier_key, _simulation_barrier
    c = dict(_config(config))
    for player in ("A", "D1", "D2"):
        c[player] = _player_point(c[player], player)
    defender_speed = float(c.get("v_D", 0.3))
    key = (
        *(float(value) for value in c["D1"]),
        *(float(value) for value in c["D2"]),
        float(c["R"]),
        float(c["v_A"]),
        float(c.get("v_D1", defender_speed)),
        float(c.get("v_D2", defender_speed)),
        int(c.get("barrier_resolution", 180)),
    )
    reused = key == _simulation_barrier_key and _simulation_barrier is not None
    if reused:
        c["_initial_barrier"] = _simulation_barrier
        c["_barrier_reused"] = True
    _simulation = EquilibriumSimulation(c)
    if not reused:
        _simulation_barrier_key = key
        _simulation_barrier = _simulation.barrier.tolist()
    return _json(_simulation.snapshot())


def simulation_step(
    dt: float,
    update_barrier: bool = False,
    manual_player: str | None = None,
    manual_heading: float | None = None,
    manual_speed: float = 1.0,
) -> str:
    if _simulation is None:
        raise RuntimeError("initialize the simulation first")
    return _json(
        _simulation.step(
            float(dt),
            bool(update_barrier),
            manual_player,
            manual_heading,
            float(manual_speed),
        )
    )


def get_simulation_state() -> str:
    if _simulation is None:
        raise RuntimeError("initialize the simulation first")
    return _json(_simulation.snapshot())
