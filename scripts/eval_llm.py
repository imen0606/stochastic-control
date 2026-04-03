"""Evaluate frontier LLMs on the Financial RLVR Gym.

Runs each model through multi-turn regime switching problems,
scores against Bellman-optimal, and compares to greedy/random baselines.

Usage:
    .venv/bin/python scripts/eval_llm.py
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import anthropic
from dotenv import load_dotenv

# Load API key from .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "financial_gym", ".env"))

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.problems.regime_switching.verifier import (
    RegimeSwitchingVerifier,
    _parse_decision,
    _compute_realized_utility,
    _linear_utility,
)
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.agents.random_agent import RandomAgent


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """Runs a Claude model through the multi-turn gym conversation."""

    def __init__(self, model: str, client: anthropic.Anthropic) -> None:
        self.model = model
        self.client = client

    def decide(self, problem) -> tuple[np.ndarray, list[str]]:
        """Run multi-turn conversation and return decisions + raw completions."""
        T = problem.T
        decisions = np.zeros(T, dtype=np.int8)
        completions = []

        # Build conversation
        system_msg = setup_prompt(problem)
        messages = []

        prev_regime = problem.initial_regime
        for t in range(T):
            # User turn
            user_msg = step_prompt(t, problem.z_path[t], prev_regime)
            messages.append({"role": "user", "content": user_msg})

            # Model response
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    system=system_msg,
                    messages=messages,
                )
                assistant_text = response.content[0].text
            except Exception as e:
                assistant_text = f"Error: {e}"

            completions.append(assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})

            # Parse decision
            decisions[t] = _parse_decision(assistant_text)
            prev_regime = int(decisions[t])

        return decisions, completions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    client: anthropic.Anthropic,
    problems: list,
    verifier: RegimeSwitchingVerifier,
) -> dict:
    """Evaluate a single model on a list of problems."""
    agent = LLMAgent(model_name, client)
    scores = []
    j_values = []
    decisions_log = []

    for i, problem in enumerate(problems):
        print(f"    Instance {i+1}/{len(problems)} (T={problem.T})...", end=" ", flush=True)
        start = time.time()

        decisions, completions = agent.decide(problem)
        score = verifier.score(completions, problem, mode="trajectory")
        j_val = _compute_realized_utility(
            decisions, problem.x_path, problem.lam,
            problem.initial_regime, _linear_utility,
        )

        elapsed = time.time() - start
        scores.append(score)
        j_values.append(j_val)
        decisions_log.append({
            "seed": problem.seed,
            "decisions": decisions.tolist(),
            "score": score,
            "j_value": j_val,
            "time_s": elapsed,
        })
        print(f"score={score:.3f}, J={j_val:.4f} ({elapsed:.1f}s)")

    return {
        "model": model_name,
        "mean_score": float(np.mean(scores)),
        "mean_j": float(np.mean(j_values)),
        "std_j": float(np.std(j_values)),
        "instances": decisions_log,
    }


def main():
    from dotenv import dotenv_values
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "financial_gym", ".env")
    vals = dotenv_values(env_path)
    client = anthropic.Anthropic(api_key=vals["ANTHROPIC_API_KEY"])

    # Models to test (verified working)
    models = [
        "claude-3-haiku-20240307",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ]

    # Test on medium difficulty (where planning matters but not extreme)
    config = GeneratorConfig(
        T_range=(15, 15),
        lam_range=(0.10, 0.10),
        alpha_range=(0.30, 0.30),
        kappa_range=(0.5, 0.5),
        grid_size=200,
        n_quad_nodes=20,
    )
    gen = RegimeSwitchingGenerator(config)
    verifier = RegimeSwitchingVerifier()

    # Generate 10 problems (same for all models)
    n_instances = 10
    problems = [gen.sample(seed=i) for i in range(n_instances)]

    # Compute baselines
    print("=" * 65)
    print("Financial RLVR Gym — LLM Evaluation (Approach 3)")
    print("=" * 65)
    print(f"Difficulty: Medium (T=15, λ=0.10, κ=0.5, α=0.30)")
    print(f"Instances: {n_instances}")
    print()

    # Baseline scores
    baseline_agents = {
        "random": RandomAgent(seed_offset=1),
        "greedy": GreedyAgent(),
        "optimal": OptimalAgent(),
    }
    baseline_j = {}
    for name, agent in baseline_agents.items():
        j_vals = []
        for p in problems:
            d = agent.decide(p)
            j = _compute_realized_utility(d, p.x_path, p.lam, p.initial_regime, _linear_utility)
            j_vals.append(j)
        baseline_j[name] = float(np.mean(j_vals))
        print(f"  Baseline {name:>8s}: mean J = {baseline_j[name]:.4f}")

    print()

    # Evaluate each model
    all_results = {}
    for model in models:
        short_name = model.split("-")[1]  # haiku, sonnet, opus
        print(f"  Testing {short_name} ({model})...")
        try:
            result = evaluate_model(model, client, problems, verifier)
            all_results[short_name] = result
            print(f"    → mean J = {result['mean_j']:.4f}, mean score = {result['mean_score']:.3f}")
        except Exception as e:
            print(f"    → ERROR: {e}")
            all_results[short_name] = {"model": model, "error": str(e)}
        print()

    # Summary table
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Agent':<12} {'Mean J':>10} {'% of Optimal':>14}")
    print("-" * 40)
    for name in ["random", "greedy"]:
        pct = baseline_j[name] / baseline_j["optimal"] * 100 if baseline_j["optimal"] != 0 else 0
        print(f"{name:<12} {baseline_j[name]:>10.4f} {pct:>13.1f}%")

    for short_name, result in all_results.items():
        if "error" in result:
            print(f"{short_name:<12} {'ERROR':>10}")
        else:
            pct = result["mean_j"] / baseline_j["optimal"] * 100 if baseline_j["optimal"] != 0 else 0
            print(f"{short_name:<12} {result['mean_j']:>10.4f} {pct:>13.1f}%")

    print(f"{'optimal':<12} {baseline_j['optimal']:>10.4f} {'100.0':>13s}%")
    print("=" * 65)

    # Save full results
    output_path = "docs/data/eval_llm_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "T": 15, "lam": 0.10, "alpha": 0.30, "kappa": 0.5,
                "n_instances": n_instances,
            },
            "baselines": baseline_j,
            "models": all_results,
        }, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
