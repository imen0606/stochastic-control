"""GRPO training on the Financial RLVR Gym — multi-turn.

Each training episode is a multi-turn conversation:
  System: problem setup (rules, parameters)
  User:   t=0 signal → Assistant: decision
  User:   t=1 signal → Assistant: decision
  ...
  → reward computed from full decision sequence vs Bellman optimal

The model sees one signal at a time and commits before seeing the next.
This preserves the information constraint: s_t ∈ σ(Z_0,...,Z_t, s_{t-1}).

Requirements:
    pip install trl>=0.15 peft transformers torch accelerate

Usage:
    python scripts/train_grpo.py                    # Mac M5 Pro (~4-8 hours)
    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py  # GPU (~30-60 min)
"""
import sys
import os
import json
import re
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
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
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.agents.greedy_agent import GreedyAgent


# ================================================================
# 1. Multi-turn conversation generation
# ================================================================

def build_conversation(problem, tokenizer) -> list[dict]:
    """Build the initial conversation (system + first user turn).

    The rest of the conversation is generated during GRPO rollout:
    the model responds, we parse its decision, then construct the
    next user turn with the model's actual previous decision.

    Returns a list of message dicts for the chat template.
    """
    system_msg = setup_prompt(problem)
    first_user_msg = step_prompt(0, problem.z_path[0], problem.initial_regime)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": first_user_msg},
    ]


def generate_episode(model, tokenizer, problem, max_new_tokens=150) -> tuple[list[dict], np.ndarray]:
    """Run one full multi-turn episode.

    At each step:
    1. Format current conversation as chat template
    2. Generate model response
    3. Parse decision from response
    4. Construct next user turn with model's actual decision
    5. Repeat for T steps

    Returns:
        conversation: full list of message dicts
        decisions: np.array of shape (T,)
    """
    T = problem.T
    conversation = [
        {"role": "system", "content": setup_prompt(problem)},
    ]
    decisions = []
    prev_regime = problem.initial_regime

    for t in range(T):
        # User turn: present signal and current state
        user_msg = step_prompt(t, problem.z_path[t], prev_regime)
        conversation.append({"role": "user", "content": user_msg})

        # Format for model
        input_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse decision
        decision = _parse_decision(response)
        decisions.append(decision)
        prev_regime = decision

        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response})

    return conversation, np.array(decisions, dtype=np.int8)


def score_decisions(decisions: np.ndarray, problem) -> dict:
    """Score a decision sequence against optimal and greedy.

    Returns dict with reward and diagnostic info.
    """
    j_model = _compute_realized_utility(
        decisions, problem.x_path, problem.lam,
        problem.initial_regime, _linear_utility,
    )
    opt_d = OptimalAgent().decide(problem)
    j_opt = _compute_realized_utility(
        opt_d, problem.x_path, problem.lam,
        problem.initial_regime, _linear_utility,
    )
    gre_d = GreedyAgent().decide(problem)
    j_gre = _compute_realized_utility(
        gre_d, problem.x_path, problem.lam,
        problem.initial_regime, _linear_utility,
    )

    # Two-bucket classification
    z_grid = _compute_z_grid(
        problem.theta, problem.sigma_z, problem.kappa,
        problem.optimal_policy_table.shape[1],
    )
    easy_correct = 0
    easy_total = 0
    hard_matches_opt = 0
    hard_total = 0
    model_prev = problem.initial_regime

    for t in range(problem.T):
        z_idx = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
        opt_from_model = int(problem.optimal_policy_table[t, z_idx, model_prev])
        q_off = 0.0 - (problem.lam if 0 != model_prev else 0.0)
        q_on = problem.alpha * problem.z_path[t] - (problem.lam if 1 != model_prev else 0.0)
        gre_from_model = 1 if q_on > q_off else 0

        if opt_from_model == gre_from_model:
            easy_total += 1
            if decisions[t] == opt_from_model:
                easy_correct += 1
        else:
            hard_total += 1
            if decisions[t] == opt_from_model:
                hard_matches_opt += 1

        model_prev = int(decisions[t])

    # Reward: J_model normalized
    # Use simple scaling: reward = J_model (raw profit)
    # GRPO normalizes within groups, so absolute scale doesn't matter
    reward = float(j_model)

    return {
        "reward": reward,
        "j_model": float(j_model),
        "j_optimal": float(j_opt),
        "j_greedy": float(j_gre),
        "easy_accuracy": easy_correct / max(easy_total, 1),
        "hard_planning_rate": hard_matches_opt / max(hard_total, 1),
        "easy_total": easy_total,
        "hard_total": hard_total,
        "num_switches": int(sum(
            1 for t in range(1, len(decisions))
            if decisions[t] != decisions[t - 1]
        ) + (1 if decisions[0] != problem.initial_regime else 0)),
    }


# ================================================================
# 2. GRPO training loop (custom, since multi-turn needs manual rollout)
# ================================================================

def run_grpo_training(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_train_problems: int = 500,
    n_eval_problems: int = 50,
    grpo_steps: int = 50,
    group_size: int = 4,
    learning_rate: float = 1e-5,
    output_dir: str = "output/grpo_financial_gym",
):
    """Custom GRPO training loop for multi-turn financial gym.

    At each GRPO step:
    1. Sample a batch of problems
    2. For each problem, generate `group_size` complete episodes
    3. Score each episode's decisions against Bellman optimal
    4. Compute GRPO advantage (within-group normalization)
    5. Update model on high-reward episodes
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("GRPO Training — Multi-Turn Financial RLVR Gym")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Training problems: {n_train_problems}")
    print(f"GRPO steps: {grpo_steps}")
    print(f"Group size: {group_size}")
    print("=" * 60)

    # Load model with LoRA
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type == "mps" else torch.bfloat16,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "mps":
        model = model.to(device)

    # Apply LoRA
    from peft import get_peft_model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Generate problems
    print("\nGenerating problems...")
    gen = RegimeSwitchingGenerator(GeneratorConfig())
    train_problems = [gen.sample(seed=i) for i in range(n_train_problems)]
    eval_problems = [gen.sample(seed=10000 + i) for i in range(n_eval_problems)]

    # Baseline evaluation
    print("\nBaseline evaluation (before training)...")
    baseline_stats = evaluate(model, tokenizer, eval_problems[:20])
    print_eval_stats("BASELINE", baseline_stats)

    # Training loop
    print("\nStarting GRPO training...")
    os.makedirs(output_dir, exist_ok=True)

    for step in range(grpo_steps):
        # Sample a batch of problems
        batch_indices = np.random.choice(len(train_problems), size=4, replace=False)
        batch_problems = [train_problems[i] for i in batch_indices]

        step_rewards = []
        step_losses = []

        for problem in batch_problems:
            # Generate group_size episodes for this problem
            group_rewards = []
            group_log_probs = []

            for g in range(group_size):
                # Run episode
                conversation, decisions = generate_episode(
                    model, tokenizer, problem, max_new_tokens=100
                )
                score = score_decisions(decisions, problem)
                group_rewards.append(score["reward"])

            # GRPO: normalize rewards within group
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            if rewards_tensor.std() > 1e-8:
                advantages = (rewards_tensor - rewards_tensor.mean()) / rewards_tensor.std()
            else:
                advantages = torch.zeros_like(rewards_tensor)

            step_rewards.extend(group_rewards)

            # For now, log the rewards — full policy gradient update
            # requires computing log-probs during generation, which
            # is more complex. This skeleton shows the structure.

        mean_reward = np.mean(step_rewards)
        if (step + 1) % 5 == 0 or step == 0:
            print(f"  Step {step+1}/{grpo_steps}: mean_reward={mean_reward:+.4f}")

        # Periodic evaluation
        if (step + 1) % 25 == 0:
            eval_stats = evaluate(model, tokenizer, eval_problems[:20])
            print_eval_stats(f"STEP {step+1}", eval_stats)

    # Final evaluation
    print("\nFinal evaluation (after training)...")
    final_stats = evaluate(model, tokenizer, eval_problems[:20])
    print_eval_stats("FINAL", final_stats)

    # Save
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"\nModel saved to {output_dir}/final")

    # Save results
    results = {
        "baseline": baseline_stats,
        "final": final_stats,
        "config": {
            "model": model_name,
            "n_train": n_train_problems,
            "grpo_steps": grpo_steps,
            "group_size": group_size,
            "learning_rate": learning_rate,
        },
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/results.json")


def evaluate(model, tokenizer, problems: list) -> dict:
    """Two-bucket evaluation on a set of problems."""
    total_easy_correct = 0
    total_easy = 0
    total_hard_opt = 0
    total_hard = 0
    j_models = []
    j_opts = []
    all_switches = []

    for p in problems:
        _, decisions = generate_episode(model, tokenizer, p, max_new_tokens=100)
        score = score_decisions(decisions, p)
        total_easy_correct += int(score["easy_accuracy"] * score["easy_total"])
        total_easy += score["easy_total"]
        total_hard_opt += int(score["hard_planning_rate"] * score["hard_total"])
        total_hard += score["hard_total"]
        j_models.append(score["j_model"])
        j_opts.append(score["j_optimal"])
        all_switches.append(score["num_switches"])

    return {
        "easy_accuracy": total_easy_correct / max(total_easy, 1),
        "hard_planning_rate": total_hard_opt / max(total_hard, 1),
        "easy_total": total_easy,
        "hard_total": total_hard,
        "mean_j_model": float(np.mean(j_models)),
        "mean_j_optimal": float(np.mean(j_opts)),
        "mean_switches": float(np.mean(all_switches)),
        "n_problems": len(problems),
    }


def print_eval_stats(label: str, stats: dict):
    """Print evaluation statistics."""
    print(f"\n  [{label}]")
    print(f"    Easy accuracy:    {stats['easy_accuracy']:.1%} ({stats['easy_total']} decisions)")
    print(f"    Hard planning:    {stats['hard_planning_rate']:.1%} ({stats['hard_total']} decisions)")
    print(f"    Mean J (model):   {stats['mean_j_model']:+.4f}")
    print(f"    Mean J (optimal): {stats['mean_j_optimal']:+.4f}")
    print(f"    Mean switches:    {stats['mean_switches']:.1f}")


# ================================================================
# 3. Entry point
# ================================================================

if __name__ == "__main__":
    run_grpo_training(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        n_train_problems=500,
        n_eval_problems=50,
        grpo_steps=50,
        group_size=4,
        learning_rate=1e-5,
    )
