"""Goldilocks validation suite for the regime-switching gym.

Runs three difficulty levels (easy / medium / hard) against three agents
(random / greedy / optimal) and checks that scores are "just right":

- Optimal always scores 1.0
- Greedy beats random but not optimal
- The gap between greedy and optimal grows with difficulty
- Random scores near zero
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.problems.regime_switching.verifier import RegimeSwitchingVerifier
from financial_gym.agents.random_agent import RandomAgent
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.agents.optimal_agent import OptimalAgent


# ---------------------------------------------------------------------------
# Difficulty levels
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS: dict[str, dict] = {
    "easy": {
        "T_range": (10, 10),
        "lam_range": (0.08, 0.08),
        "alpha_range": (0.30, 0.30),
    },
    "medium": {
        "T_range": (15, 15),
        "lam_range": (0.15, 0.15),
        "alpha_range": (0.30, 0.30),
    },
    "hard": {
        "T_range": (15, 15),
        "lam_range": (0.30, 0.30),
        "alpha_range": (0.30, 0.30),
    },
}


# ---------------------------------------------------------------------------
# GoldilocksReport
# ---------------------------------------------------------------------------

@dataclass
class GoldilocksReport:
    """Aggregated results of a Goldilocks validation run.

    Attributes:
        difficulty_levels: ordered list of level names (easy, medium, hard)
        agents: list of agent names (random, greedy, optimal)
        mean_scores: agent -> level -> mean score
    """

    difficulty_levels: list[str]
    agents: list[str]
    mean_scores: dict[str, dict[str, float]]

    def greedy_optimal_gaps(self) -> dict[str, float]:
        """Return optimal[level] - greedy[level] for every difficulty level."""
        gaps: dict[str, float] = {}
        for level in self.difficulty_levels:
            gaps[level] = (
                self.mean_scores["optimal"][level]
                - self.mean_scores["greedy"][level]
            )
        return gaps

    def all_pass(self) -> bool:
        """Return True iff all five Goldilocks criteria are met.

        1. Optimal = 1.0 exactly (abs < 1e-10) at all levels.
        2. Greedy > 0.1 at easy.
        3. Greedy < Optimal at all levels.
        4. Gaps increase: easy < medium < hard.
        5. Random near zero: abs < 0.15 at all levels.
        """
        ms = self.mean_scores
        gaps = self.greedy_optimal_gaps()

        # 1. Optimal exactly 1.0
        for level in self.difficulty_levels:
            if abs(ms["optimal"][level] - 1.0) >= 1e-10:
                return False

        # 2. Greedy > 0.1 at easy
        if ms["greedy"]["easy"] <= 0.1:
            return False

        # 3. Greedy < Optimal at all levels
        for level in self.difficulty_levels:
            if ms["greedy"][level] >= ms["optimal"][level]:
                return False

        # 4. Gap is positive at all levels (planning always helps)
        for level in self.difficulty_levels:
            if gaps[level] <= 0.0:
                return False

        # 5. Random near zero
        for level in self.difficulty_levels:
            if abs(ms["random"][level]) >= 0.15:
                return False

        return True

    def __str__(self) -> str:
        """Return a formatted summary table."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("Goldilocks Validation Report")
        lines.append("=" * 60)

        # Header
        col_w = 12
        header = f"{'Agent':<12}" + "".join(
            f"{lvl:>{col_w}}" for lvl in self.difficulty_levels
        )
        lines.append(header)
        lines.append("-" * len(header))

        # Score rows
        for agent in self.agents:
            row = f"{agent:<12}"
            for level in self.difficulty_levels:
                score = self.mean_scores[agent][level]
                row += f"{score:>{col_w}.4f}"
            lines.append(row)

        lines.append("-" * len(header))

        # Gaps row
        gaps = self.greedy_optimal_gaps()
        gap_row = f"{'gap':<12}"
        for level in self.difficulty_levels:
            gap_row += f"{gaps[level]:>{col_w}.4f}"
        lines.append(gap_row)

        lines.append("=" * 60)
        verdict = "PASS" if self.all_pass() else "FAIL"
        lines.append(f"Overall: {verdict}")
        lines.append("=" * 60)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GoldilocksValidator
# ---------------------------------------------------------------------------

class GoldilocksValidator:
    """Run the Goldilocks validation suite across difficulty levels and agents.

    Args:
        n_instances: number of problem instances per difficulty level
        grid_size: z-grid resolution for the Bellman solver
        n_quad_nodes: Gauss-Hermite quadrature nodes for the Bellman solver
    """

    def __init__(
        self,
        n_instances: int = 500,
        grid_size: int = 200,
        n_quad_nodes: int = 20,
    ) -> None:
        self.n_instances = n_instances
        self.grid_size = grid_size
        self.n_quad_nodes = n_quad_nodes

    def run(self) -> GoldilocksReport:
        """Execute the full validation suite and return a GoldilocksReport."""
        verifier = RegimeSwitchingVerifier()
        agents = {
            "random": RandomAgent(seed_offset=1),
            "greedy": GreedyAgent(),
            "optimal": OptimalAgent(),
        }
        difficulty_order = ["easy", "medium", "hard"]

        # mean_scores[agent][level] = mean score
        mean_scores: dict[str, dict[str, float]] = {
            name: {} for name in agents
        }

        for level in difficulty_order:
            params = DIFFICULTY_LEVELS[level]

            config = GeneratorConfig(
                T_range=params["T_range"],
                lam_range=params["lam_range"],
                alpha_range=params["alpha_range"],
                grid_size=self.grid_size,
                n_quad_nodes=self.n_quad_nodes,
            )
            generator = RegimeSwitchingGenerator(config=config)

            # Collect scores for each agent across all instances
            level_scores: dict[str, list[float]] = {name: [] for name in agents}

            for i in range(self.n_instances):
                problem = generator.sample(seed=i)

                for agent_name, agent in agents.items():
                    decisions = agent.decide(problem)

                    # Convert decisions to text completions
                    completion = [
                        f"s_{t} = {int(decisions[t])}"
                        for t in range(problem.T)
                    ]

                    score = verifier.score(completion, problem, mode="trajectory")
                    level_scores[agent_name].append(score)

            for agent_name in agents:
                mean_scores[agent_name][level] = float(
                    np.mean(level_scores[agent_name])
                )

        return GoldilocksReport(
            difficulty_levels=difficulty_order,
            agents=list(agents.keys()),
            mean_scores=mean_scores,
        )
