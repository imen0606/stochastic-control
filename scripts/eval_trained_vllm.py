#!/usr/bin/env python3
"""
Evaluate a trained model on fresh episodes using vLLM for fast inference.

Usage:
    # Evaluate the trained model (LoRA adapter)
    python scripts/eval_trained_vllm.py --model output/grpo_planning/final_model -N 50 --tag trained

    # Compare with base model
    python scripts/eval_trained_vllm.py --model Qwen/Qwen3-8B -N 50 --tag baseline

    # Quick test
    python scripts/eval_trained_vllm.py --model output/grpo_planning/final_model -N 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

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


def load_vllm_model(model_path, base_model=None):
    """Load model with vLLM, handling LoRA adapters."""
    from vllm import LLM, SamplingParams

    # Check if it's a LoRA adapter
    adapter_path = None
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftConfig
        cfg = PeftConfig.from_pretrained(model_path)
        adapter_path = model_path
        model_path = base_model or cfg.base_model_name_or_path
        print(f"  LoRA adapter: {adapter_path}")
        print(f"  Base model: {model_path}")

    if adapter_path:
        from vllm.lora.request import LoRARequest
        llm = LLM(
            model=model_path,
            enable_lora=True,
            max_lora_rank=64,
            max_model_len=8192,
            gpu_memory_utilization=0.8,
        )
        lora_req = LoRARequest("trained", 1, adapter_path)
    else:
        llm = LLM(
            model=model_path,
            max_model_len=8192,
            gpu_memory_utilization=0.8,
        )
        lora_req = None

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=300,
    )

    return llm, sampling, lora_req


def run_episode(llm, sampling, lora_req, tokenizer, problem):
    """Run one episode step-by-step with vLLM."""
    from vllm import SamplingParams

    system_msg = setup_prompt(problem)
    messages = [{"role": "system", "content": system_msg}]
    prev = problem.initial_regime
    decisions = []
    raw_texts = []

    for t in range(problem.T):
        user_msg = step_prompt(t, problem.z_path[t], prev)
        messages.append({"role": "user", "content": user_msg})

        # Build prompt via chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        # Generate with vLLM
        kwargs = {}
        if lora_req is not None:
            kwargs["lora_request"] = lora_req
        outputs = llm.generate([prompt], sampling, **kwargs)
        text = outputs[0].outputs[0].text

        messages.append({"role": "assistant", "content": text})
        d = _parse_decision(text)
        decisions.append(d)
        raw_texts.append(text)
        prev = d

    return decisions, raw_texts


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


def evaluate(llm, sampling, lora_req, tokenizer, N=50, seed_start=10000):
    """Evaluate model on N filtered episodes."""
    config = GeneratorConfig.planning_zone()
    gen = RegimeSwitchingGenerator(config)

    results = []
    total_easy_c = total_easy_t = total_hard_o = total_hard_t = 0
    j_models = []
    j_opts = []
    j_gres = []

    seed = seed_start
    count = 0
    while count < N:
        p = gen.sample(seed=seed)
        seed += 1
        if not _has_plannable_hard(p):
            continue

        t0 = time.time()
        decisions, raw_texts = run_episode(llm, sampling, lora_req, tokenizer, p)
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
        j_models.append(j_model)
        j_opts.append(j_opt)
        j_gres.append(j_gre)

        results.append({
            "seed": p.seed, "kappa": p.kappa, "T": p.T,
            "easy_correct": ec, "easy_total": et,
            "hard_opt": ho, "hard_total": ht,
            "j_model": j_model, "j_opt": j_opt, "j_gre": j_gre,
        })

        count += 1
        print(
            f"  [{count}/{N}] seed={p.seed} T={p.T} "
            f"easy={ec}/{et} hard={ho}/{ht} "
            f"J_model={j_model:+.3f} J_opt={j_opt:+.3f} J_gre={j_gre:+.3f} ({elapsed:.1f}s)",
            flush=True,
        )

    easy_acc = total_easy_c / max(total_easy_t, 1)
    hard_rate = total_hard_o / max(total_hard_t, 1)

    print(f"\n{'='*60}")
    print(f"RESULTS (N={N}):")
    print(f"  Easy accuracy:    {total_easy_c}/{total_easy_t} ({easy_acc:.1%})")
    print(f"  Hard match rate:  {total_hard_o}/{total_hard_t} ({hard_rate:.1%})")
    print(f"  J_model (mean):   {np.mean(j_models):+.3f} +/- {np.std(j_models):.3f}")
    print(f"  J_optimal (mean): {np.mean(j_opts):+.3f} +/- {np.std(j_opts):.3f}")
    print(f"  J_greedy (mean):  {np.mean(j_gres):+.3f} +/- {np.std(j_gres):.3f}")
    print(f"{'='*60}")

    return {
        "easy": {"correct": total_easy_c, "total": total_easy_t, "accuracy": easy_acc},
        "hard": {"opt": total_hard_o, "total": total_hard_t, "rate": hard_rate},
        "j_model_mean": float(np.mean(j_models)),
        "j_opt_mean": float(np.mean(j_opts)),
        "j_gre_mean": float(np.mean(j_gres)),
        "instances": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--base-model", default=None,
                        help="Base model (if --model is a LoRA adapter and auto-detect fails)")
    parser.add_argument("-N", type=int, default=50, help="Number of episodes")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--tag", default="trained", help="Tag for output filename")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm, sampling, lora_req = load_vllm_model(args.model, args.base_model)

    # Load tokenizer separately for chat template
    from transformers import AutoTokenizer
    tok_path = args.model
    if lora_req is not None:
        # For LoRA, tokenizer is in the base model
        from peft import PeftConfig
        tok_path = PeftConfig.from_pretrained(args.model).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    print(f"Evaluating on {args.N} episodes (seeds {args.seed_start}+)")
    result = evaluate(llm, sampling, lora_req, tokenizer, N=args.N, seed_start=args.seed_start)

    os.makedirs("docs/data", exist_ok=True)
    output_path = f"docs/data/eval_{args.tag}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
