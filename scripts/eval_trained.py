#!/usr/bin/env python3
"""
Evaluate a trained model on fresh episodes.

Compares pre-training baseline to post-training performance
on episodes never seen during training.

Usage:
    # Evaluate the trained model
    python scripts/eval_trained.py --model output/grpo_planning/final_model

    # Compare with base model
    python scripts/eval_trained.py --model Qwen/Qwen2.5-3B-Instruct --tag baseline

    # Quick test
    python scripts/eval_trained.py --model output/grpo_planning/final_model -N 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt


def run_episode(model, tokenizer, problem, max_new_tokens=300):
    """Run one episode with a local model."""
    system_msg = setup_prompt(problem)
    messages = [{"role": "system", "content": system_msg}]
    prev = problem.initial_regime
    decisions = []
    raw_texts = []

    for t in range(problem.T):
        user_msg = step_prompt(t, problem.z_path[t], prev)
        messages.append({"role": "user", "content": user_msg})

        # Tokenize and generate
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        messages.append({"role": "assistant", "content": text})
        d = _parse_decision(text)
        decisions.append(d)
        raw_texts.append(text)
        prev = d

    return decisions, raw_texts


def evaluate(model, tokenizer, N=100, seed_start=10000):
    """Evaluate model on N fresh episodes."""
    config = GeneratorConfig.planning_zone()
    gen = RegimeSwitchingGenerator(config)

    results = []
    total_easy_c = total_easy_t = total_hard_o = total_hard_t = 0

    for i in range(N):
        seed = seed_start + i
        p = gen.sample(seed=seed)

        t0 = time.time()
        decisions, raw_texts = run_episode(model, tokenizer, p)
        elapsed = time.time() - t0

        model_d = np.array(decisions, dtype=np.int8)
        z_grid = _compute_z_grid(p.theta, p.sigma_z, p.kappa, 200)

        # Two-bucket analysis
        ec = et = ho = ht = 0
        op = p.initial_regime
        for t in range(p.T):
            zi = int(np.argmin(np.abs(z_grid - p.z_path[t])))
            oa = int(p.optimal_policy_table[t, zi, op])
            qoff = 0.0 - (p.lam if 0 != op else 0.0)
            qon = p.alpha * p.z_path[t] - (p.lam if 1 != op else 0.0)
            ga = 1 if qon > qoff else 0
            if oa == ga:
                et += 1
                if model_d[t] == oa:
                    ec += 1
            else:
                ht += 1
                if model_d[t] == oa:
                    ho += 1
            op = int(model_d[t])

        # J values
        j_model = _compute_realized_utility(
            model_d, p.x_path, p.lam, p.initial_regime, _linear_utility
        )
        j_opt = _compute_realized_utility(
            OptimalAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility
        )
        j_gre = _compute_realized_utility(
            GreedyAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility
        )

        total_easy_c += ec
        total_easy_t += et
        total_hard_o += ho
        total_hard_t += ht

        results.append({
            "seed": seed, "kappa": p.kappa, "T": p.T,
            "easy_correct": ec, "easy_total": et,
            "hard_opt": ho, "hard_total": ht,
            "j_model": j_model, "j_opt": j_opt, "j_gre": j_gre,
            "decisions": decisions, "raw_texts": raw_texts,
        })

        print(
            f"  [{i+1}/{N}] seed={seed} T={p.T} "
            f"easy={ec}/{et} hard={ho}/{ht} "
            f"J={j_model:+.3f} ({elapsed:.0f}s)",
            flush=True,
        )

    easy_acc = total_easy_c / max(total_easy_t, 1)
    hard_rate = total_hard_o / max(total_hard_t, 1)

    print(f"\n{'='*50}")
    print(f"RESULTS (N={N}):")
    print(f"  Easy: {total_easy_c}/{total_easy_t} ({easy_acc:.1%})")
    print(f"  Hard: {total_hard_o}/{total_hard_t} ({hard_rate:.1%})")
    print(f"{'='*50}")

    return {
        "easy": {"correct": total_easy_c, "total": total_easy_t, "accuracy": easy_acc},
        "hard": {"opt": total_hard_o, "total": total_hard_t, "rate": hard_rate},
        "instances": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("-N", type=int, default=100, help="Number of episodes")
    parser.add_argument("--seed-start", type=int, default=10000,
                        help="Starting seed (use >5000 to avoid training data)")
    parser.add_argument("--tag", default="trained", help="Tag for output filename")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    print(f"Evaluating on {args.N} episodes (seeds {args.seed_start}-{args.seed_start + args.N - 1})")
    result = evaluate(model, tokenizer, N=args.N, seed_start=args.seed_start)

    output_path = f"docs/data/eval_{args.tag}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
