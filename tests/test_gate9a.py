from respiratory_sound.gate9a import (
    resolve_gate_status,
    single_seed_decision,
    three_seed_decision,
)


def _summary(
    seed: int,
    icbhi: tuple[float, float, float],
    sprsound: tuple[float, float, float],
) -> dict:
    return {
        "candidate": "panns_cnn6",
        "seed": seed,
        "best_validation_metrics": {
            "icbhi2017": {
                "average_score": icbhi[0],
                "sensitivity": icbhi[1],
                "specificity": icbhi[2],
            },
            "sprsound2022": {
                "average_score": sprsound[0],
                "sensitivity": sprsound[1],
                "specificity": sprsound[2],
            },
        },
    }


def test_gate9a_single_seed_requires_all_frozen_thresholds() -> None:
    passing = _summary(20260729, (0.71, 0.70, 0.72), (0.89, 0.86, 0.92))
    unsafe = _summary(20260729, (0.71, 0.49, 0.93), (0.89, 0.86, 0.92))

    assert single_seed_decision(passing)["passed"]
    assert not single_seed_decision(unsafe)["passed"]


def test_gate9a_three_seed_requires_two_individual_passes_and_safety() -> None:
    summaries = [
        _summary(20260729, (0.71, 0.70, 0.72), (0.89, 0.86, 0.92)),
        _summary(20260730, (0.72, 0.71, 0.73), (0.90, 0.87, 0.93)),
        _summary(20260731, (0.69, 0.68, 0.70), (0.87, 0.84, 0.90)),
    ]

    decision = three_seed_decision(summaries)

    assert decision["passed"]
    assert sum(item["passed"] for item in decision["single_seed_decisions"]) == 2


def test_gate9a_overall_status_reports_completed_confirmation() -> None:
    candidate_results = {
        "panns_cnn6": {
            "single_seed_screen_passed": False,
            "three_seed_decision": None,
        },
        "beats": {
            "single_seed_screen_passed": True,
            "three_seed_decision": {"passed": True},
        },
    }

    assert (
        resolve_gate_status(
            candidate_results,
            both_single_seed_screens_complete=True,
        )
        == "gate9a_passed_three_seed_confirmation"
    )
