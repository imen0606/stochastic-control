"""Financial RLVR Gym — verifiable environments for LLM post-training."""
from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    RegimeSwitchingProblem,
)
from financial_gym.problems.regime_switching.verifier import RegimeSwitchingVerifier
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.agents.random_agent import RandomAgent
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.validation.goldilocks import GoldilocksValidator

__all__ = [
    "GeneratorConfig",
    "RegimeSwitchingGenerator",
    "RegimeSwitchingProblem",
    "RegimeSwitchingVerifier",
    "setup_prompt",
    "step_prompt",
    "RandomAgent",
    "GreedyAgent",
    "OptimalAgent",
    "GoldilocksValidator",
]
