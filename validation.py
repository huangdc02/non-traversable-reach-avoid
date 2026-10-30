"""Compact numerical regression checks for the independently ported solver."""

import json
import sys
from math import pi, sqrt
from pathlib import Path

import numpy as np

scripts = Path(__file__).with_name("scripts")
for directory in ("barrier", "strategy", "simulation", "api"):
    sys.path.insert(0, str(scripts / directory))
import web_api  # noqa: E402
from equilibrium_strategy import solve_equilibrium  # noqa: E402
from geometry_common import boundary_derivative, dsp, geometry, mode_roots  # noqa: E402
from otp_solver import (  # noqa: E402
    advance_dsp,
    lifted_distance,
    lifted_q,
    is_directly_visible,
    solve_1d1a_otp,
    solve_2d1a_otp,
)
from simulation import _advance_manual_defender  # noqa: E402


def close(actual, expected, tolerance):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


blocked_ccw = _advance_manual_defender(
    np.array([1.02, 0.0]), pi, 0.05, 1.0, +1
)
blocked_cw = _advance_manual_defender(
    np.array([1.02, 0.0]), pi, 0.05, 1.0, -1
)
close(blocked_ccw, [1.0, 0.03], 2e-12)
close(blocked_cw, [1.0, -0.03], 2e-12)
close(
    _advance_manual_defender(np.array([1.0, 0.0]), 3 * pi / 4, 0.05, 1.0, -1),
    [1.0, 0.05],
    2e-12,
)


circle = json.loads(
    web_api.compute_circle_barrier(
        {
            "D1": [0, 1.5],
            "D2": [-2, 0],
            "R": 1,
            "alpha": 2 / 3,
            "beta": 1,
            "resolution": 360,
        }
    )
)
assert len(circle["p_items"]) == 2
close(circle["p_items"][0]["point"], [-0.8418690678, 0.5396818254], 2e-8)
close(circle["p_items"][1]["point"], [0.5492397361, -0.8356648325], 2e-8)
close(np.min(circle["barrier"], axis=0), [-1.69442822, -2.72695675], 2e-7)
close(np.max(circle["barrier"], axis=0), [2.46367354, 1.41666641], 2e-7)

polygon = json.loads(
    web_api.compute_polygon_barrier(
        {
            "D1": [-2.35, 2.05],
            "D2": [2.45, 1.95],
            "alpha": 2 / 3,
            "beta": 1,
            "vertices": [
                [-1.15, -0.65],
                [1.15, -0.65],
                [1.25, 0.55],
                [0.25, 1.15],
                [-1.05, 0.75],
            ],
            "samples_per_edge": 60,
            "grid_size": 180,
        }
    )
)
assert len(polygon["segments"]) == 4 and len(polygon["p_items"]) == 4
close(polygon["p_items"][0]["point"], [-0.0238844849, -0.65], 2e-8)
close(polygon["p_items"][1]["point"], [0.0848437151, 1.0991826816], 2e-8)
close(polygon["p_items"][2]["point"], [0.0090731253, -0.65], 2e-8)
close(polygon["p_items"][3]["point"], [0.0308870968, 1.0825806452], 2e-8)
close(np.min(polygon["barrier"], axis=0), [-3.11974459, -3.37046378], 2e-7)
close(np.max(polygon["barrier"], axis=0), [3.08789377, 2.79584286], 2e-7)

initial = {
    "A": [-0.406422989, -3.339212412],
    "D1": [0, 1.5],
    "D2": [-2.3, -3.8],
    "R": 1,
    "v_A": 0.2,
    "v_D": 0.3,
    "barrier_resolution": 160,
}
initial_state = json.loads(web_api.init_simulation(initial))
assert not initial_state["barrier_reused"]
initial_simulation_barrier = np.asarray(initial_state["barrier"])
first = json.loads(web_api.simulation_step(0.05))
assert first["barrier_revision"] == 0 and not first["barrier_is_live"]
close(first["otp"], [0.3977883201, -0.9174772217], 2e-8)
close(first["otp_lift"], [1.0, -1.1616913542], 2e-8)
assert first["sigma"] == [1, 1]
assert first["lift_pair"] == [-1, -1]
assert first["stationary_mask"] == [False, True]
assert first["selected_branch"] == "D1-" and first["mode"] == "D1 only"
close(first["V"], 0.0, 2e-12)
close(
    [first["headings"][p] for p in ("A", "D1", "D2")],
    [1.2501736431, -2.3005239830, 0.0],
    2e-8,
)
assert len(first["barrier"]) > 300
close(first["barrier"], initial_simulation_barrier, 0.0)
live_barrier_state = json.loads(web_api.simulation_step(0.05, True))
assert not live_barrier_state["barrier_reused"]
assert live_barrier_state["barrier_revision"] == 1
assert live_barrier_state["barrier_is_live"]
close(live_barrier_state["barrier_defenders"]["D1"], live_barrier_state["D1"], 0.0)
close(live_barrier_state["barrier_defenders"]["D2"], live_barrier_state["D2"], 0.0)
assert not np.array_equal(live_barrier_state["barrier"], initial_simulation_barrier)
second_live_barrier_state = json.loads(web_api.simulation_step(0.05, True))
assert second_live_barrier_state["barrier_revision"] == 2
assert not np.array_equal(
    second_live_barrier_state["barrier"], live_barrier_state["barrier"]
)
state = second_live_barrier_state
while not state["terminated"]:
    state = json.loads(web_api.simulation_step(0.05))
assert state["result"] == "Captured by D1"
close(state["time"], 12.7144878817, 2e-6)
close(state["A"], [0.3949904395, -0.9259025271], 2e-7)
close(state["D1"], [0.3855357529, -0.9226928976], 2e-7)
close(state["D2"], [-2.3, -3.8], 0.0)
close(state["barrier"], second_live_barrier_state["barrier"], 0.0)

json.loads(web_api.init_simulation(initial))
manual = json.loads(web_api.simulation_step(0.05, False, "D2", 0.0))
assert manual["manual_player"] == "D2"
close(manual["manual_heading"], 0.0, 0.0)
close(manual["D2"], [-2.285, -3.8], 2e-12)
close(np.linalg.norm(np.asarray(manual["D2"]) - initial["D2"]), 0.015, 2e-12)
close(manual["headings"]["D2"], 0.0, 0.0)
close(manual["manual_speed"], 1.0, 0.0)

json.loads(web_api.init_simulation(initial))
partial = json.loads(web_api.simulation_step(0.05, False, "D2", 0.0, 0.4))
close(partial["D2"], [-2.294, -3.8], 2e-12)
close(np.linalg.norm(np.asarray(partial["D2"]) - initial["D2"]), 0.006, 2e-12)
close(partial["manual_speed"], 0.4, 0.0)

json.loads(web_api.init_simulation(initial))
centered = json.loads(web_api.simulation_step(0.05, False, "D2", None))
close(centered["D2"], initial["D2"], 0.0)
close(centered["manual_speed"], 0.0, 0.0)

json.loads(web_api.init_simulation(initial))
tangent_slide = json.loads(
    web_api.simulation_step(2.0, False, "D1", -pi / 2, 1.0)
)
close(tangent_slide["D1"], [-0.1, 1.0], 2e-12)
close(np.linalg.norm(tangent_slide["D1"]), sqrt(1.01), 2e-12)

constrained = json.loads(web_api.init_simulation({**initial, "A": [0, 0]}))
assert constrained["attacker_constrained"]
assert constrained["barrier_reused"]
close(np.linalg.norm(constrained["A"]), constrained["constraint_barrier_radius"], 2e-10)
changed_defender = json.loads(
    web_api.init_simulation(
        {
            **initial,
            "D1": [0.05, 1.5],
        }
    )
)
assert not changed_defender["barrier_reused"]
try:
    web_api.init_simulation({**initial, "A": [10.01, 0]})
except ValueError as error:
    assert "[-10, 10]" in str(error)
else:
    raise AssertionError("out-of-range player coordinate was accepted")

long_lift_length = 2 * sqrt(3) + 2 * pi - 2 * pi / 3
close(lifted_distance((2.0, 0.0), 2.0, 2 * pi, 1.0), long_lift_length, 1e-12)
moved = advance_dsp(
    [2.0, 0.0],
    [2.0, 0.0],
    1.0,
    1,
    0.1,
    lift_angles=(0.0, 2 * pi),
)
assert np.linalg.norm(moved - [2.0, 0.0]) > 0.09

params = {"R": 1.0, "v_A": 0.2, "v_D": 0.3, "v_D1": 0.3, "v_D2": 0.3}
distance = sqrt(3) + pi / 6
goal_root_state = {
    "D1": np.array([-2.0, 0.0]),
    "D2": np.array([2.0, 0.0]),
    "A": np.array([0.0, 1.0 + (2 / 3) * distance]),
}
goal_root = solve_2d1a_otp(goal_root_state, (1, -1), params)
assert goal_root["success"]
close(goal_root["otp"], [0.0, 1.0], 2e-5)

case_99 = {
    "D1": np.array([0.103378375, -1.486749558]),
    "D2": np.array([-1.795688118, -2.678076627]),
    "A": np.array([0.143905352, 4.569056667]),
}
lifted_solution = solve_equilibrium(case_99, params)
assert lifted_solution["success"]
minima = []
assert set(lifted_solution["lift_groups"]) == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
for pair, group in lifted_solution["lift_groups"].items():
    for candidate in group:
        q = [
            lifted_q(
                candidate["otp_lift"],
                case_99["A"],
                case_99[defender],
                1.0,
                2 / 3,
                pair[index],
            )
            for index, defender in enumerate(("D1", "D2"))
        ]
        assert max(q) <= 1e-7
    if group:
        minima.append(min(candidate["value"] for candidate in group))
close(lifted_solution["value"], max(minima), 1e-10)

# A physically visible endpoint must not shorten the other, long lift.
g = geometry([2.0, 0.0])
assert dsp(g, 0.1, -1) < 1.1 and dsp(g, 0.1, +1) > 6.0
for side in (-1, 1):
    for angle in (0.1, 2.0, 4.0, 6.0):
        finite_difference = (
            dsp(g, angle + 1e-6, side) - dsp(g, angle - 1e-6, side)
        ) / 2e-6
        close(boundary_derivative(g, angle, side, 1.0), finite_difference, 1e-7)
for beta in (0.8, 1.0, 1.25):
    g1, g2 = geometry([0.0, 1.5]), geometry([-2.3, -3.8])
    for pair in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        for angle in mode_roots(g1, g2, pair, beta=beta):
            close(dsp(g1, angle, pair[0]), dsp(g2, angle, pair[1]) / beta, 1e-8)

unequal_params = {"R": 1.0, "v_A": 0.2, "v_D1": 0.36, "v_D2": 0.3}
unequal_solution = solve_equilibrium(case_99, unequal_params)
assert unequal_solution["success"]
for candidate in unequal_solution["candidates"]:
    assert is_directly_visible(case_99["A"], candidate["otp"], 1.0)
    for i, defender in enumerate(("D1", "D2")):
        q = lifted_q(
            candidate["otp_lift"],
            case_99["A"],
            case_99[defender],
            1.0,
            0.2 / unequal_params[f"v_{defender}"],
            candidate["lift_pair"][i],
        )
        assert q <= 1e-7
        if defender in candidate["active"]:
            assert abs(q) <= 1e-7
unequal_initial = {**initial, **unequal_params}
del unequal_initial["v_D"]
unequal_state = json.loads(web_api.init_simulation(unequal_initial))
assert not unequal_state["barrier_reused"]
assert json.loads(web_api.init_simulation(unequal_initial))["barrier_reused"]
unequal_step = json.loads(web_api.simulation_step(0.05, True))
assert unequal_step["time"] > 0 and unequal_step["barrier_is_live"]
for player in ("A", "D1", "D2"):
    assert (
        np.linalg.norm(np.array(unequal_step[player]) - unequal_state[player])
        <= unequal_params[f"v_{player}"] * 0.05 + 1e-12
    )
try:
    web_api.init_simulation({**unequal_initial, "v_D1": 0.1})
except ValueError as error:
    assert "v_A <= min(v_D1, v_D2)" in str(error)
else:
    raise AssertionError("defender slower than attacker was accepted")

boundary_case = {
    "D1": np.array([0.0, 1.5]),
    "D2": np.array([-2.3, -3.8]),
    "A": np.array([-0.406422989, -3.339212412]),
}
boundary_solution = solve_equilibrium(boundary_case, params)
assert not boundary_solution["lift_groups"][(1, -1)]
assert len(boundary_solution["lift_groups"][(-1, 1)]) == 1
assert boundary_solution["branch_id"] == "D1-"
for side in (-1, 1):
    failed = solve_1d1a_otp(
        boundary_case["A"],
        boundary_case["D2"],
        side,
        {"R": 1.0, "v_A": 0.2, "v_D": 0.3},
    )
    assert not failed["success"] and failed["otp"] is None
    assert failed["boundary"] is None

switch_state = {
    "D1": np.array([-0.9973919445, -0.0721755436]),
    "D2": np.array([-1.7011333057, -1.9757854615]),
    "A": np.array([-0.0030212956, -2.1244420330]),
}
switch_solution = solve_equilibrium(switch_state, params)
reason = switch_solution["discarded_pairs"][(1, -1)]
assert reason["trigger_defender"] == "D2"
assert reason["reason"] == "direct_target_inside"
assert not switch_solution["lift_groups"][(1, -1)]

stationary = solve_equilibrium(boundary_case, params)
assert stationary["stationary_mask"] == (False, True)
close(stationary["phi_D2"], 0.0, 0.0)

print("barriers, fixed-lift equilibrium, simulation, and A constraint passed")
