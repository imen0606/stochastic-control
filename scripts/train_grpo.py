"""Integration example: how to wire the Financial RLVR Gym into a GRPO training loop.

This is NOT a training script. It shows labs exactly how to use the gym's
three components (generator, prompts, verifier) in their own infrastructure.

The gym provides:
  1. Problem generation (infinite, parameterised, each with exact optimal)
  2. Multi-turn prompt construction (one signal per turn, stateful)
  3. Reward computation (score model completions against Bellman optimal)

The lab provides:
  - The model
  - The training loop (TRL GRPOTrainer, veRL, or custom)
  - Multi-turn rollout management
  - Log-probability computation and gradient updates
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from financial_gym import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    RegimeSwitchingVerifier,
    setup_prompt,
    step_prompt,
)


# ================================================================
# STEP 1: Generate a problem
# ================================================================

# Default config varies all parameters across episodes:
#   kappa  ∈ [0.1, 0.5]   (signal persistence — unknown to model)
#   sigma_z ∈ [0.1, 0.3]  (signal noise — unknown to model)
#   alpha  ∈ [0.1, 0.5]   (signal strength — given to model)
#   lambda ∈ [0.0, 0.3]   (switching cost — given to model)
#   T      ∈ [3, 20]      (horizon — given to model)

generator = RegimeSwitchingGenerator(GeneratorConfig())
problem = generator.sample(seed=42)
# problem is fully solved — includes Bellman-optimal policy table


# ================================================================
# STEP 2: Construct the multi-turn conversation
# ================================================================

# The setup prompt (system message) — given once per episode
system_message = setup_prompt(problem)
print("SYSTEM MESSAGE:")
print(system_message)
print()

# Each step is one turn. The user message depends on the model's
# previous decision (stateful — cannot be pre-generated).

prev_regime = problem.initial_regime
completions = []

for t in range(problem.T):
    # Construct user message for this step
    user_message = step_prompt(t, problem.z_path[t], prev_regime)
    print(f"USER (t={t}): {user_message}")

    # ── YOUR MODEL GENERATES HERE ──
    # response = model.generate(system=system_message, messages=conversation)
    # For this example, simulate with a simple heuristic:
    expected_pnl = problem.alpha * problem.z_path[t]
    if prev_regime == 0:
        decision = 1 if expected_pnl > problem.lam else 0
    else:
        decision = 0 if expected_pnl < -problem.lam else 1
    response = f"s_{t} = {decision}"
    # ── END MODEL GENERATION ──

    print(f"ASSISTANT: {response}")
    completions.append(response)

    # Parse the model's decision to construct the next turn
    from financial_gym.problems.regime_switching.verifier import _parse_decision
    prev_regime = _parse_decision(response)

print()


# ================================================================
# STEP 3: Score the full episode
# ================================================================

verifier = RegimeSwitchingVerifier()
score = verifier.score(completions, problem, mode="trajectory")
print(f"TRAJECTORY SCORE: {score:.4f}")
print(f"  (0.0 = random level, 1.0 = optimal)")
print()

# Diagnostic: per-step accuracy
per_step = verifier.score(completions, problem, mode="per_step")
print(f"PER-STEP ACCURACY: {per_step:.1%}")
print(f"  (fraction of decisions matching optimal given model's own trajectory)")
print()


# ================================================================
# STEP 4: Batch interface for GRPO
# ================================================================

# GRPO generates K completions per problem, scores them, and
# updates the policy toward higher-scoring completions.
#
# The reward function for a batch:

def reward_fn(completions_batch: list[list[str]],
              problems_batch: list) -> list[float]:
    """Score a batch of multi-turn completions.

    Args:
        completions_batch: List of episodes. Each episode is a list
            of T assistant response strings (one per turn).
        problems_batch: List of RegimeSwitchingProblem instances.

    Returns:
        List of float rewards, one per episode.
    """
    verifier = RegimeSwitchingVerifier()
    return [
        verifier.score(completions, problem, mode="trajectory")
        for completions, problem in zip(completions_batch, problems_batch)
    ]


# Example: score 3 episodes
problems = [generator.sample(seed=i) for i in range(3)]
# (In practice, completions come from model generation)
fake_completions = [
    [f"s_{t} = 0" for t in range(p.T)]  # always-OFF baseline
    for p in problems
]
scores = reward_fn(fake_completions, problems)
print(f"BATCH SCORES (always-OFF baseline): {scores}")
print()


# ================================================================
# SUMMARY: What the lab wires into their GRPO loop
# ================================================================

print("INTEGRATION SUMMARY")
print("=" * 50)
print("""
1. PROBLEM GENERATION:
   generator = RegimeSwitchingGenerator(GeneratorConfig())
   problem = generator.sample(seed=i)

2. MULTI-TURN ROLLOUT (per step):
   system_msg = setup_prompt(problem)
   user_msg = step_prompt(t, problem.z_path[t], prev_regime)
   response = YOUR_MODEL.generate(...)
   prev_regime = _parse_decision(response)

3. REWARD:
   score = verifier.score(completions, problem)

That's it. The gym handles generation, solving, and scoring.
The lab handles the model, the rollout, and the training.
""")
