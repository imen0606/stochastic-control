"""Tests for the Goldilocks validation suite."""
from __future__ import annotations

import pytest

from financial_gym.validation.goldilocks import GoldilocksValidator, GoldilocksReport

# Use production-quality grid with enough instances for statistical stability.
# grid_size=200 ensures solver accuracy; n_instances=200 smooths variance.
VALIDATOR = GoldilocksValidator(n_instances=200, grid_size=200, n_quad_nodes=20)


@pytest.fixture(scope="module")
def report() -> GoldilocksReport:
    return VALIDATOR.run()


def test_report_structure(report: GoldilocksReport) -> None:
    assert len(report.difficulty_levels) == 3
    assert set(report.difficulty_levels) == {"easy", "medium", "hard"}
    assert len(report.agents) == 3
    assert set(report.agents) == {"random", "greedy", "optimal"}


def test_optimal_scores_exactly_one(report: GoldilocksReport) -> None:
    for level in report.difficulty_levels:
        score = report.mean_scores["optimal"][level]
        assert abs(score - 1.0) < 1e-10, (
            f"Optimal mean score at {level} is {score}, expected 1.0"
        )


def test_greedy_below_optimal(report: GoldilocksReport) -> None:
    for level in report.difficulty_levels:
        greedy = report.mean_scores["greedy"][level]
        optimal = report.mean_scores["optimal"][level]
        assert greedy < optimal, (
            f"Greedy ({greedy}) should be < Optimal ({optimal}) at {level}"
        )


def test_random_near_zero(report: GoldilocksReport) -> None:
    for level in report.difficulty_levels:
        score = report.mean_scores["random"][level]
        assert abs(score) < 0.2, (
            f"Random mean score at {level} is {score}, expected abs < 0.2"
        )


def test_gap_positive_at_all_levels(report: GoldilocksReport) -> None:
    """Planning always helps: greedy-optimal gap is positive at every level."""
    gaps = report.greedy_optimal_gaps()
    for level, gap in gaps.items():
        assert gap > 0.0, (
            f"Expected positive gap at {level}, got {gap}"
        )


def test_pass_criteria(report: GoldilocksReport) -> None:
    assert report.all_pass(), "Expected all_pass() to return True"


def test_report_string(report: GoldilocksReport) -> None:
    s = str(report)
    assert "Goldilocks" in s
    assert "PASS" in s or "FAIL" in s
