#!/usr/bin/env python3
"""
Parallel multi-model evaluation with prompt caching.

Splits seed ranges across workers for faster execution.
Supports prompt caching for Anthropic models (90% input cost reduction).

Usage:
  # Test run: 2 workers, seeds 30-39 (10 episodes)
  python scripts/eval_parallel.py --model opus --start 30 --end 39 --workers 2

  # Full run: 5 workers, seeds 30-499 (470 episodes)
  python scripts/eval_parallel.py --model opus --start 30 --end 499 --workers 5

  # All models in sequence
  python scripts/eval_parallel.py --model all --start 30 --end 499 --workers 5
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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


# ================================================================
# API calls with caching
# ================================================================

def call_anthropic_cached(api_key, model_id, system_msg, messages, max_tokens=300):
    """Call Anthropic API with prompt caching enabled."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system_msg, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
            )
            usage = resp.usage
            cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
            cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            return resp.content[0].text, {
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'cache_read': cache_read,
                'cache_write': cache_write,
            }
        except Exception as e:
            if 'overloaded' in str(e).lower() and attempt < 2:
                time.sleep(30)
                continue
            return f"Error: {e}", {'input_tokens': 0, 'output_tokens': 0, 'cache_read': 0, 'cache_write': 0}


def call_openai_cached(api_key, model_id, system_msg, messages, max_tokens=300):
    """Call OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=api_key)

    oai_messages = [{"role": "system", "content": system_msg}]
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=oai_messages,
            )
            usage = resp.usage
            return resp.choices[0].message.content, {
                'input_tokens': usage.prompt_tokens,
                'output_tokens': usage.completion_tokens,
                'cache_read': getattr(usage, 'prompt_tokens_details', {}).get('cached_tokens', 0) if hasattr(usage, 'prompt_tokens_details') else 0,
                'cache_write': 0,
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(10)
                continue
            return f"Error: {e}", {'input_tokens': 0, 'output_tokens': 0, 'cache_read': 0, 'cache_write': 0}


# ================================================================
# Single worker: evaluate a range of seeds
# ================================================================

def evaluate_seed_range(worker_id, seed_start, seed_end, model_name, model_id, api_key, provider):
    """Evaluate episodes for a range of seeds. Runs in a subprocess."""
    config = GeneratorConfig.planning_zone()
    gen = RegimeSwitchingGenerator(config)

    call_fn = call_anthropic_cached if provider == 'anthropic' else call_openai_cached

    instances = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0, 'cache_read': 0, 'cache_write': 0}

    for seed in range(seed_start, seed_end + 1):
        p = gen.sample(seed=seed)
        T = p.T
        system_msg = setup_prompt(p)
        messages = []
        prev = p.initial_regime
        decisions = []
        raw_texts = []

        t0 = time.time()
        for t in range(T):
            user_msg = step_prompt(t, p.z_path[t], prev)
            messages.append({"role": "user", "content": user_msg})
            text, usage = call_fn(api_key, model_id, system_msg, messages)
            messages.append({"role": "assistant", "content": text})

            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            d = _parse_decision(text)
            decisions.append(d)
            raw_texts.append(text)
            prev = d

        model_d = np.array(decisions, dtype=np.int8)
        elapsed = time.time() - t0

        # Two-bucket
        z_grid = _compute_z_grid(p.theta, p.sigma_z, p.kappa, 200)
        ec = et = ho = hg = ht = 0
        op = p.initial_regime
        for t in range(T):
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
                elif model_d[t] == ga:
                    hg += 1
            op = int(model_d[t])

        # J values
        j_model = _compute_realized_utility(model_d, p.x_path, p.lam, p.initial_regime, _linear_utility)
        j_opt = _compute_realized_utility(OptimalAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility)
        j_gre = _compute_realized_utility(GreedyAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility)

        instances.append({
            "seed": seed, "kappa": p.kappa, "alpha": p.alpha, "lam": p.lam,
            "sigma_z": p.sigma_z, "T": T,
            "decisions": decisions, "raw_texts": raw_texts,
            "easy_correct": ec, "easy_total": et,
            "hard_opt": ho, "hard_total": ht,
            "j_model": j_model, "j_opt": j_opt, "j_gre": j_gre,
        })

        idx = seed - seed_start + 1
        total = seed_end - seed_start + 1
        print(f"  [W{worker_id}] [{idx}/{total}] seed={seed} kappa={p.kappa:.2f} T={T} easy={ec}/{et} hard={ho}/{ht} J={j_model:+.3f} ({elapsed:.0f}s)", flush=True)

    return {
        "worker_id": worker_id,
        "seed_range": [seed_start, seed_end],
        "instances": instances,
        "usage": total_usage,
    }


# ================================================================
# Merge results
# ================================================================

def merge_results(worker_results, model_name, model_id, existing_path=None):
    """Merge worker results and optionally combine with existing N=30 data."""
    all_instances = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0, 'cache_read': 0, 'cache_write': 0}

    # Load existing data if present
    if existing_path and os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        all_instances.extend(existing['instances'])
        print(f"Loaded {len(existing['instances'])} existing instances from {existing_path}")

    # Add new results
    for wr in sorted(worker_results, key=lambda x: x['seed_range'][0]):
        all_instances.extend(wr['instances'])
        for k in total_usage:
            total_usage[k] += wr['usage'].get(k, 0)

    # Sort by seed
    all_instances.sort(key=lambda x: x['seed'])

    # Aggregate
    te = sum(x["easy_total"] for x in all_instances)
    tec = sum(x["easy_correct"] for x in all_instances)
    th = sum(x["hard_total"] for x in all_instances)
    tho = sum(x["hard_opt"] for x in all_instances)

    config = GeneratorConfig.planning_zone()

    result = {
        "condition": f"planning_{model_name}",
        "model": model_id,
        "config": {
            "kappa_range": list(config.kappa_range),
            "lam_alpha_ratio_range": list(config.lam_alpha_ratio_range),
        },
        "easy": {"total": te, "correct": tec, "accuracy": tec / max(te, 1)},
        "hard": {"total": th, "opt": tho, "planning_rate": tho / max(th, 1)},
        "usage": total_usage,
        "instances": all_instances,
    }

    print(f"\nMERGED RESULTS ({len(all_instances)} episodes):")
    print(f"  Easy: {tec}/{te} ({tec/max(te,1):.1%})")
    print(f"  Hard: {tho}/{th} ({tho/max(th,1):.1%})")
    print(f"  Usage: {total_usage}")

    return result


# ================================================================
# Main
# ================================================================

MODEL_CONFIGS = {
    'opus': ('claude-opus-4-20250514', 'anthropic', 'ANTHROPIC_API_KEY'),
    'haiku': ('claude-haiku-4-5-20251001', 'anthropic', 'ANTHROPIC_API_KEY'),
    'sonnet': ('claude-sonnet-4-20250514', 'anthropic', 'ANTHROPIC_API_KEY'),
    'gpt4o': ('gpt-4o', 'openai', 'OPENAI_API_KEY'),
}

# Map model names to existing N=30 files
EXISTING_FILES = {
    'opus': 'docs/data/eval_planning_original.json',
    'haiku': 'docs/data/eval_planning_haiku.json',
    'sonnet': 'docs/data/eval_planning_sonnet.json',
    'gpt4o': 'docs/data/eval_planning_gpt4o.json',
}


def run_model(model_name, seed_start, seed_end, num_workers, api_key):
    """Run evaluation for one model with parallel workers."""
    model_id, provider, _ = MODEL_CONFIGS[model_name]

    total_seeds = seed_end - seed_start + 1
    seeds_per_worker = total_seeds // num_workers
    remainder = total_seeds % num_workers

    # Split seed ranges
    ranges = []
    current = seed_start
    for w in range(num_workers):
        count = seeds_per_worker + (1 if w < remainder else 0)
        ranges.append((current, current + count - 1))
        current += count

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({model_id})")
    print(f"Seeds: {seed_start}-{seed_end} ({total_seeds} episodes)")
    print(f"Workers: {num_workers}, ranges: {ranges}")
    print(f"{'='*60}", flush=True)

    if num_workers == 1:
        # Single process — no subprocess overhead
        result = evaluate_seed_range(0, seed_start, seed_end, model_name, model_id, api_key, provider)
        worker_results = [result]
    else:
        # Parallel workers
        worker_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for w, (s_start, s_end) in enumerate(ranges):
                f = executor.submit(
                    evaluate_seed_range, w, s_start, s_end,
                    model_name, model_id, api_key, provider
                )
                futures[f] = w

            for f in as_completed(futures):
                w = futures[f]
                try:
                    result = f.result()
                    worker_results.append(result)
                    n = len(result['instances'])
                    print(f"\n  Worker {w} completed: {n} episodes", flush=True)
                except Exception as e:
                    print(f"\n  Worker {w} FAILED: {e}", flush=True)

    # Merge with existing N=30
    existing_path = EXISTING_FILES.get(model_name)
    merged = merge_results(worker_results, model_name, model_id, existing_path)

    # Save
    output_path = f"docs/data/eval_planning_{model_name}_n500.json"
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Saved to {output_path}", flush=True)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='opus', choices=list(MODEL_CONFIGS.keys()) + ['all'])
    parser.add_argument('--start', type=int, default=30)
    parser.add_argument('--end', type=int, default=499)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()

    from dotenv import dotenv_values
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "financial_gym", ".env"
    )
    vals = dotenv_values(env_path)

    models_to_run = [args.model] if args.model != 'all' else list(MODEL_CONFIGS.keys())

    for model_name in models_to_run:
        _, _, key_name = MODEL_CONFIGS[model_name]
        api_key = vals.get(key_name, '')
        if not api_key or api_key.startswith('YOUR_'):
            print(f"ERROR: Set {key_name} in financial_gym/.env")
            continue
        run_model(model_name, args.start, args.end, args.workers, api_key)


if __name__ == "__main__":
    main()
