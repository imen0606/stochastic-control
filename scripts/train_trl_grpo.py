#!/usr/bin/env python3
"""
GRPO training on the regime switching gym using TRL.

Uses rollout_func (not environment_factory) to avoid tool-calling
requirements. We control the multi-turn generation loop directly.

Usage on RunPod (single A100):
    python scripts/train_trl_grpo.py

    # With custom model:
    python scripts/train_trl_grpo.py --model Qwen/Qwen2.5-7B-Instruct

    # Quick test:
    python scripts/train_trl_grpo.py --test
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    _compute_z_grid,
)
from financial_gym.problems.regime_switching.verifier import (
    _parse_decision,
    _compute_realized_utility,
    _linear_utility,
)
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt


# ---------------------------------------------------------------------------
# Episode generation + reward
# ---------------------------------------------------------------------------

def _has_plannable_hard(problem, min_t=2, trend_thr=0.6):
    """Check if episode has at least one plannable hard decision."""
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
                tf = sum(1 for z in history if z > 0) / len(history)
            else:
                tf = sum(1 for z in history if z < 0) / len(history)
            if tf >= trend_thr:
                return True
        prev = oa
    return False


def compute_reward(completions: list[str], problem) -> float:
    """Compute regret-normalised reward for one episode."""
    decisions = np.array([_parse_decision(c) for c in completions], dtype=np.int8)

    j_model = _compute_realized_utility(
        decisions, problem.x_path, problem.lam,
        problem.initial_regime, _linear_utility,
    )

    # Random baseline
    rng = np.random.default_rng(problem.seed + 99999)
    j_randoms = []
    for _ in range(20):
        rd = rng.integers(0, 2, size=problem.T).astype(np.int8)
        j_randoms.append(_compute_realized_utility(
            rd, problem.x_path, problem.lam,
            problem.initial_regime, _linear_utility,
        ))
    j_random = np.mean(j_randoms)

    opt_d = OptimalAgent().decide(problem)
    j_optimal = _compute_realized_utility(
        opt_d, problem.x_path, problem.lam,
        problem.initial_regime, _linear_utility,
    )

    denom = j_optimal - j_random
    if abs(denom) < 0.001:
        return 1.0 if abs(j_model - j_optimal) < 0.001 else 0.0
    return float(np.clip((j_model - j_random) / denom, -2.0, 2.0))


# ---------------------------------------------------------------------------
# Reward function for GRPO (single-turn wrapper)
# ---------------------------------------------------------------------------

def reward_fn(completions, **kwargs) -> list[float]:
    """Reward function compatible with TRL GRPOTrainer.

    Each completion is a single-turn response containing ALL decisions
    for one episode (the model outputs T decisions in one response).
    """
    prompts = kwargs.get("prompts", [])
    rewards = []
    for i, completion in enumerate(completions):
        # Extract the problem seed from the prompt
        try:
            # The seed is embedded in the prompt as metadata
            seed = kwargs["seed"][i] if "seed" in kwargs else i
            problem = _problem_cache.get(seed)
            if problem is None:
                rewards.append(0.0)
                continue

            # Parse all T decisions from the completion text
            text = completion[0]["content"] if isinstance(completion, list) else completion
            # Split by decision markers
            lines = text.split("\n")
            decisions_text = [l for l in lines if "s_" in l or "s =" in l]

            if len(decisions_text) >= problem.T:
                completions_list = decisions_text[:problem.T]
            else:
                completions_list = decisions_text + ["s_t = 0"] * (problem.T - len(decisions_text))

            reward = compute_reward(completions_list, problem)
            rewards.append(reward)
        except Exception as e:
            rewards.append(0.0)

    return rewards


# Global problem cache
_problem_cache = {}


# ---------------------------------------------------------------------------
# Build dataset with full episode prompts
# ---------------------------------------------------------------------------

def build_dataset(num_episodes: int, gen: RegimeSwitchingGenerator) -> Dataset:
    """Build dataset where each prompt contains the full episode context.

    Since we can't do true multi-turn with rollout_func easily,
    we present the ENTIRE episode in a single prompt: all T signal
    values at once, and ask the model to output T decisions.
    """
    prompts = []
    seeds = []

    seed = 0
    count = 0
    while count < num_episodes:
        seed += 1
        problem = gen.sample(seed=seed)

        # Filter for episodes with plannable hard decisions
        if not _has_plannable_hard(problem):
            continue

        # Cache the problem for reward computation
        _problem_cache[seed] = problem

        # Build the full episode prompt
        system = setup_prompt(problem)
        signal_lines = []
        for t in range(problem.T):
            signal_lines.append(
                f"t={t} | Z_t = {problem.z_path[t]:+.4f}"
            )
        signals_str = "\n".join(signal_lines)

        prompt_text = (
            f"{system}\n\n"
            f"Here are ALL the signal observations for the episode:\n"
            f"{signals_str}\n\n"
            f"For each time step, decide s_t = 0 (OFF) or s_t = 1 (ON).\n"
            f"Consider signal trends and whether switching costs can be "
            f"amortised over future steps.\n"
            f"Output your decisions as a list, one per line:\n"
            f"s_0 = ...\n"
            f"s_1 = ...\n"
            f"...\n"
            f"s_{problem.T - 1} = ..."
        )

        prompts.append([{"role": "user", "content": prompt_text}])
        seeds.append(seed)
        count += 1

    return Dataset.from_dict({"prompt": prompts, "seed": seeds})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="output/grpo_planning")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--num-generations", type=int, default=8,
                        help="K completions per episode for GRPO")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 64 episodes, 1 epoch")
    args = parser.parse_args()

    if args.test:
        args.num_episodes = 64
        args.epochs = 1
        args.num_generations = 4

    print(f"Building dataset ({args.num_episodes} filtered episodes)...")
    gen = RegimeSwitchingGenerator(GeneratorConfig.planning_zone())
    dataset = build_dataset(args.num_episodes, gen)
    print(f"  Dataset built: {len(dataset)} episodes, {len(_problem_cache)} problems cached")

    print(f"\nStarting GRPO training:")
    print(f"  Model: {args.model}")
    print(f"  Episodes: {len(dataset)}")
    print(f"  K (generations per episode): {args.num_generations}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Output: {args.output_dir}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=args.output_dir,

            # vLLM for fast generation
            use_vllm=True,
            vllm_mode="colocate",

            # GRPO hyperparameters
            num_generations=args.num_generations,
            max_completion_length=2048,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,

            # Logging
            logging_steps=5,
            save_steps=200,
            log_completions=True,
            report_to="none",  # set to "wandb" if wandb configured
        ),
    )

    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"\nTraining complete. Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
