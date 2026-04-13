"""
TRL-compatible environment wrapper for the regime switching gym.

Wraps our Bellman-verified gym as a TRL GRPOTrainer environment_factory.
The model interacts via a `decide` tool method each turn.

Usage with GRPOTrainer:
    from scripts.trl_env import RegimeSwitchingEnv, reward_func
    trainer = GRPOTrainer(
        ...,
        environment_factory=RegimeSwitchingEnv,
        reward_funcs=reward_func,
    )
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    _compute_z_grid,
)
from financial_gym.problems.regime_switching.verifier import (
    RegimeSwitchingVerifier,
    _parse_decision,
    _compute_realized_utility,
    _linear_utility,
)
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt


# Thread-safe seed counter (multiple envs may run in parallel)
_seed_lock = threading.Lock()
_seed_counter = 0


def _next_seed():
    global _seed_counter
    with _seed_lock:
        _seed_counter += 1
        return _seed_counter


def _has_plannable_hard_decision(problem, min_t=2, trend_threshold=0.6):
    """Check if an episode has at least one plannable hard decision.

    A plannable hard decision occurs at t >= min_t where:
    - Greedy and optimal disagree
    - The signal history shows a clear trend (>= trend_threshold same-sign)
    """
    z_grid = _compute_z_grid(problem.theta, problem.sigma_z, problem.kappa, 200)
    prev = problem.initial_regime

    for t in range(problem.T):
        zi = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
        oa = int(problem.optimal_policy_table[t, zi, prev])

        qoff = 0.0 - (problem.lam if 0 != prev else 0.0)
        qon = problem.alpha * problem.z_path[t] - (problem.lam if 1 != prev else 0.0)
        ga = 1 if qon > qoff else 0

        if oa != ga and t >= min_t:
            history = problem.z_path[:t]
            if oa == 1:
                trend_frac = sum(1 for z in history if z > 0) / len(history)
            else:
                trend_frac = sum(1 for z in history if z < 0) / len(history)
            if trend_frac >= trend_threshold:
                return True

        prev = oa  # track optimal's trajectory for fair assessment

    return False


class RegimeSwitchingEnv:
    """TRL-compatible environment for the regime switching gym.

    The model calls the `decide` tool at each time step with its
    reasoning and decision. The environment returns the next observation.
    After T steps, the reward is computed via the Bellman-verified
    regret-normalised score.
    """

    def __init__(self):
        self.gen = RegimeSwitchingGenerator(GeneratorConfig.planning_zone())
        self.verifier = RegimeSwitchingVerifier()
        self.reward = 0.0

        # Episode state
        self.problem = None
        self.t = 0
        self.prev_regime = 0
        self.completions = []

    def reset(self, **kwargs) -> str:
        """Sample a new episode with at least one plannable hard decision.

        Returns the system prompt + first observation as the initial context.
        """
        # Sample until we get an episode with a plannable hard decision
        max_attempts = 50
        for _ in range(max_attempts):
            seed = _next_seed()
            problem = self.gen.sample(seed=seed)
            if _has_plannable_hard_decision(problem):
                break

        self.problem = problem
        self.t = 0
        self.prev_regime = problem.initial_regime
        self.completions = []
        self.reward = 0.0

        # Return system prompt + first step as initial observation
        system = setup_prompt(problem)
        first_step = step_prompt(0, problem.z_path[0], self.prev_regime)
        return f"{system}\n\n{first_step}"

    def decide(self, reasoning_and_decision: str) -> str:
        """Submit your reasoning and decision for this time step.

        Analyse the current signal, consider the trading strategy's
        parameters, and decide whether to be ON (s_t = 1) or OFF (s_t = 0).
        Include your reasoning followed by your decision as: s_t = 0 or s_t = 1

        Args:
            reasoning_and_decision: Your analysis and decision. Must end with s_t = 0 or s_t = 1

        Returns:
            The next time step's observation, or signals episode completion.
        """
        # Record the completion
        self.completions.append(reasoning_and_decision)

        # Parse the decision
        parsed = _parse_decision(reasoning_and_decision)
        self.prev_regime = parsed
        self.t += 1

        # Check if episode is done
        if self.t >= self.problem.T:
            # Compute reward
            self.reward = self._compute_reward()
            raise ValueError("Episode complete. All time steps decided.")

        # Return next observation
        return step_prompt(self.t, self.problem.z_path[self.t], self.prev_regime)

    def _compute_reward(self) -> float:
        """Compute regret-normalised reward from the episode's decisions."""
        # Build decision array from completions
        decisions = np.array(
            [_parse_decision(c) for c in self.completions],
            dtype=np.int8,
        )

        # Compute J values
        j_model = _compute_realized_utility(
            decisions, self.problem.x_path, self.problem.lam,
            self.problem.initial_regime, _linear_utility,
        )

        # Compute random baseline (average over multiple random seeds)
        rng = np.random.default_rng(self.problem.seed + 99999)
        j_randoms = []
        for _ in range(20):
            random_d = rng.integers(0, 2, size=self.problem.T).astype(np.int8)
            j_r = _compute_realized_utility(
                random_d, self.problem.x_path, self.problem.lam,
                self.problem.initial_regime, _linear_utility,
            )
            j_randoms.append(j_r)
        j_random = np.mean(j_randoms)

        # Compute optimal J
        from financial_gym.agents.optimal_agent import OptimalAgent
        opt_d = OptimalAgent().decide(self.problem)
        j_optimal = _compute_realized_utility(
            opt_d, self.problem.x_path, self.problem.lam,
            self.problem.initial_regime, _linear_utility,
        )

        # Regret-normalised score
        denom = j_optimal - j_random
        if abs(denom) < 0.001:
            return 1.0 if abs(j_model - j_optimal) < 0.001 else 0.0

        return float(np.clip((j_model - j_random) / denom, -2.0, 2.0))


def reward_func(environments, **kwargs) -> list[float]:
    """Reward function for TRL GRPOTrainer.

    Reads the reward computed by each environment instance
    after the episode completes.
    """
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Local testing (no GPU needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing RegimeSwitchingEnv locally...")

    env = RegimeSwitchingEnv()

    # Test 3 episodes
    for ep in range(3):
        obs = env.reset()
        print(f"\n{'='*50}")
        print(f"Episode {ep+1} (seed={_seed_counter})")
        print(f"Problem: T={env.problem.T}, kappa={env.problem.kappa:.3f}")
        print(f"Initial obs (first 200 chars): {obs[:200]}...")

        # Simulate a greedy model
        step = 0
        while True:
            # Simple greedy decision based on signal
            z = env.problem.z_path[env.t] if env.t < env.problem.T else 0
            pnl = env.problem.alpha * z
            if env.prev_regime == 0:
                # Currently OFF: switch ON if PnL > switching cost
                if pnl > env.problem.lam:
                    decision = f"Signal is {z:+.4f}, PnL={pnl:.4f} > lambda={env.problem.lam:.4f}. s_t = 1"
                else:
                    decision = f"Signal is {z:+.4f}, PnL={pnl:.4f} < lambda={env.problem.lam:.4f}. s_t = 0"
            else:
                # Currently ON: switch OFF if loss > switching cost
                if -pnl > env.problem.lam:
                    decision = f"Signal is {z:+.4f}, loss={-pnl:.4f} > lambda={env.problem.lam:.4f}. s_t = 0"
                else:
                    decision = f"Signal is {z:+.4f}, staying ON. s_t = 1"

            try:
                next_obs = env.decide(decision)
                step += 1
            except ValueError as e:
                print(f"  Episode ended after {step+1} steps: {e}")
                print(f"  Reward: {env.reward:+.3f}")
                break

    # Test reward_func
    envs = [RegimeSwitchingEnv() for _ in range(2)]
    for e in envs:
        obs = e.reset()
        for t in range(e.problem.T):
            try:
                e.decide(f"s_t = {1 if e.problem.z_path[e.t] > 0 else 0}")
            except ValueError:
                break

    rewards = reward_func(envs)
    print(f"\nBatch reward_func test: {rewards}")
    print("\nAll tests passed!")
