"""Compare four fixed-lift groups and derive the three equilibrium controls."""

from typing import Any
import numpy as np

from otp_solver import (
    CONSTRAINT_TOL,
    central_lift,
    defender_lift,
    dsp_heading,
    is_directly_visible,
    lifted_q,
    solve_1d1a_otp,
    solve_2d1a_otp,
)

LIFT_PAIRS = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))


def solve_equilibrium(
    X: dict[str, Any], params: dict, branch_id: str | None = None
) -> dict:
    """Minimize candidates within each fixed-lift pair, then maximize pairs.

    Returned sigma contains actual travel directions; lift_pair and path_lifts
    retain the distinct homotopy information needed for execution.
    """
    state = {player: np.asarray(X[player], dtype=float) for player in ("A", "D1", "D2")}
    radius = float(params["R"])
    attacker_speed = float(params["v_A"])
    defender_names = ("D1", "D2")
    defender_speeds = tuple(
        float(params[f"v_{defender}"]) for defender in defender_names
    )
    if (
        not np.isfinite((attacker_speed, *defender_speeds)).all()
        or attacker_speed <= 0.0
        or min(defender_speeds) <= 0.0
        or attacker_speed > min(defender_speeds) * (1.0 + 1e-12)
    ):
        raise ValueError("require 0 < v_A <= min(v_D1, v_D2)")
    speed_ratios = tuple(attacker_speed / speed for speed in defender_speeds)
    constraint_tolerance = float(params.get("constraint_tol", CONSTRAINT_TOL))
    tie_tolerance = float(params.get("equilibrium_tie_tol", CONSTRAINT_TOL))
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("equilibrium_tie_tol must be finite and nonnegative")
    candidates: list[dict[str, Any]] = []
    discarded_pairs: dict[tuple[int, int], dict[str, Any]] = {}

    def add_candidate(
        point: np.ndarray,
        target_lift: tuple[float, float] | np.ndarray,
        lift_pair: tuple[int, int],
        active_defenders: tuple[str, ...],
    ) -> None:
        if not is_directly_visible(state["A"], point, radius):
            return
        q_values = tuple(
            lifted_q(
                target_lift,
                state["A"],
                state[defender],
                radius,
                speed_ratios[index],
                lift_pair[index],
            )
            for index, defender in enumerate(defender_names)
        )
        if max(q_values) > constraint_tolerance:
            return
        if any(
            abs(q_values[index]) > constraint_tolerance
            for index, defender in enumerate(defender_names)
            if defender in active_defenders
        ):
            return
        if any(
            candidate["lift_pair"] == lift_pair
            and candidate["active"] == active_defenders
            and np.linalg.norm(candidate["otp_lift"] - target_lift)
            <= constraint_tolerance
            for candidate in candidates
        ):
            return
        path_lifts = {}
        travel_directions = []
        for index, defender in enumerate(defender_names):
            source_lift = defender_lift(state[defender], state["A"], lift_pair[index])
            path_lifts[defender] = (source_lift[1], float(target_lift[1]))
            lift_delta = target_lift[1] - source_lift[1]
            travel_directions.append(1 if lift_delta >= 0 else -1)
        path_lifts["A"] = (
            float(np.arctan2(state["A"][1], state["A"][0])),
            float(target_lift[1]),
        )
        branch_id = "_".join(
            f'{defender}{"+" if lift_pair[index] > 0 else "-"}'
            for index, defender in enumerate(defender_names)
            if defender in active_defenders
        )
        candidates.append(
            {
                "branch_id": branch_id,
                "active": active_defenders,
                "otp": np.asarray(point).copy(),
                "otp_lift": np.asarray(target_lift).copy(),
                "value": float(target_lift[0] - radius),
                "sigma": tuple(travel_directions),
                "lift_pair": lift_pair,
                "path_lifts": path_lifts,
                "q_values": q_values,
            }
        )

    # Each fixed-lift pair contains only its matching individual branches.
    # All four adjacent-lift combinations in the paper's set Sigma are
    # enumerated. Failed 1D1A branches add no fallback.
    for lift_pair in LIFT_PAIRS:
        pair_start = len(candidates)
        discard_reason = None
        for index, defender in enumerate(defender_names):
            lift_side = lift_pair[index]
            result = solve_1d1a_otp(
                state["A"],
                state[defender],
                lift_side,
                params,
                defender_speed=defender_speeds[index],
            )
            probe = result.get("pair_discard_probe")
            if probe is not None:
                other_index = 1 - index
                other_defender = defender_names[other_index]
                q_other = lifted_q(
                    central_lift(probe, state["A"]),
                    state["A"],
                    state[other_defender],
                    radius,
                    speed_ratios[other_index],
                    lift_pair[other_index],
                )
                if q_other <= -constraint_tolerance:
                    discard_reason = {
                        "trigger_defender": defender,
                        "other_defender": other_defender,
                        "reason": result["pair_discard_reason"],
                        "point": np.asarray(probe, dtype=float).copy(),
                        "q_other": float(q_other),
                    }
                    break
            if result["success"]:
                add_candidate(
                    result["otp"],
                    central_lift(result["otp"], state["A"]),
                    lift_pair,
                    (defender,),
                )
        if discard_reason is not None:
            del candidates[pair_start:]
            discarded_pairs[lift_pair] = discard_reason
            continue
        result = solve_2d1a_otp(state, lift_pair, params)
        for root in result["roots"]:
            add_candidate(root["otp"], root["otp_lift"], lift_pair, ("D1", "D2"))

    lift_groups = {
        lift_pair: tuple(
            candidate for candidate in candidates if candidate["lift_pair"] == lift_pair
        )
        for lift_pair in LIFT_PAIRS
    }
    lift_optima = {
        lift_pair: min(group, key=lambda candidate: candidate["value"])
        for lift_pair, group in lift_groups.items()
        if group
    }
    if not lift_optima:
        return {
            "success": False,
            "otp": None,
            "value": float("nan"),
            "sigma": None,
            "selected_candidate": None,
            "candidates": tuple(candidates),
            "lift_groups": lift_groups,
            "lift_optima": lift_optima,
            "discarded_pairs": discarded_pairs,
            "stationary_mask": (False, False),
            "phi_A": float("nan"),
            "phi_D1": float("nan"),
            "phi_D2": float("nan"),
        }
    best_lift_optimum = max(
        lift_optima.values(), key=lambda candidate: candidate["value"]
    )
    optimal_candidates = tuple(
        candidate
        for lift_pair, group in lift_groups.items()
        if lift_pair in lift_optima
        and best_lift_optimum["value"] - lift_optima[lift_pair]["value"]
        <= tie_tolerance
        for candidate in group
        if candidate["value"] - lift_optima[lift_pair]["value"] <= tie_tolerance
    )
    selected = (
        best_lift_optimum
        if branch_id is None
        else next(
            (
                candidate
                for candidate in optimal_candidates
                if candidate["branch_id"] == branch_id
            ),
            None,
        )
    )
    if selected is None:
        raise ValueError(f"branch_id {branch_id!r} is not an optimal candidate")
    otp = selected["otp"]
    travel_directions = selected["sigma"]
    path_lifts = selected["path_lifts"]
    stationary_mask = tuple(
        defender not in selected["active"] for defender in defender_names
    )
    headings = {}
    for index, player in enumerate(("D1", "D2", "A")):
        if index < 2 and stationary_mask[index]:
            headings[f"phi_{player}"] = 0.0
            continue
        direction = (
            travel_directions[index]
            if index < 2
            else (1 if path_lifts[player][1] >= path_lifts[player][0] else -1)
        )
        headings[f"phi_{player}"] = dsp_heading(
            state[player], otp, radius, direction, lift_angles=path_lifts[player]
        )
    return dict(
        success=True,
        otp=otp.copy(),
        otp_lift=selected["otp_lift"].copy(),
        value=selected["value"],
        sigma=travel_directions,
        lift_pair=selected["lift_pair"],
        path_lifts=path_lifts,
        branch_id=selected["branch_id"],
        selected_candidate=selected,
        candidates=tuple(candidates),
        lift_groups=lift_groups,
        lift_optima=lift_optima,
        discarded_pairs=discarded_pairs,
        stationary_mask=stationary_mask,
        **headings,
    )
