#!/usr/bin/env python3
"""
Multi-model evaluation: Haiku, GPT-4o, Opus on the planning zone.

Same seeds, same parameters, same verifier — only the model changes.
Tests whether planning capability scales with model size.

Run:  python scripts/eval_multi_model.py [model]
  haiku   = Claude Haiku (planning zone, N=30)
  gpt4o   = GPT-4o (planning zone, N=30)
  opus    = Claude Opus (planning zone, N=30) — replication
  all     = All three sequentially
"""
import json
import os
import sys
import time

import numpy as np
from dotenv import dotenv_values

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
# Model clients
# ================================================================

def call_anthropic(client, model_id, system_msg, messages, max_tokens=300):
    """Call Anthropic API (Opus or Haiku)."""
    import anthropic
    try:
        resp = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system_msg,
            messages=messages,
        )
        return resp.content[0].text
    except anthropic._exceptions.OverloadedError:
        time.sleep(30)
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_msg,
                messages=messages,
            )
            return resp.content[0].text
        except Exception as e:
            return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def call_openai(client, model_id, system_msg, messages, max_tokens=300):
    """Call OpenAI API (GPT-4o)."""
    oai_messages = [{"role": "system", "content": system_msg}]
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})
    try:
        resp = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return resp.choices[0].message.content
    except Exception as e:
        time.sleep(10)
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=oai_messages,
            )
            return resp.choices[0].message.content
        except Exception as e2:
            return f"Error: {e2}"


# ================================================================
# Evaluation
# ================================================================

def run_eval(condition_name, config, call_fn, client, model_id, N=30, seed_start=0):
    """Run one evaluation condition."""
    gen = RegimeSwitchingGenerator(config)

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name} (model: {model_id})")
    print(f"N={N}, config: kappa_range={config.kappa_range}")
    print(f"{'='*60}", flush=True)

    instances = []
    for i in range(seed_start, seed_start + N):
        p = gen.sample(seed=i)
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
            text = call_fn(client, model_id, system_msg, messages)
            messages.append({"role": "assistant", "content": text})
            d = _parse_decision(text)
            decisions.append(d)
            raw_texts.append(text)
            prev = d

        opus_d = np.array(decisions, dtype=np.int8)
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
                if opus_d[t] == oa:
                    ec += 1
            else:
                ht += 1
                if opus_d[t] == oa:
                    ho += 1
                elif opus_d[t] == ga:
                    hg += 1
            op = int(opus_d[t])

        # J values
        j_model = _compute_realized_utility(opus_d, p.x_path, p.lam, p.initial_regime, _linear_utility)
        j_opt = _compute_realized_utility(OptimalAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility)
        j_gre = _compute_realized_utility(GreedyAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility)

        instances.append({
            "seed": i, "kappa": p.kappa, "alpha": p.alpha, "lam": p.lam,
            "sigma_z": p.sigma_z, "T": T,
            "decisions": decisions, "raw_texts": raw_texts,
            "easy_correct": ec, "easy_total": et,
            "hard_opt": ho, "hard_total": ht,
            "j_model": j_model, "j_opt": j_opt, "j_gre": j_gre,
        })
        print(f"  [{i+1}/{N}] kappa={p.kappa:.2f} T={T} easy={ec}/{et} hard={ho}/{ht} J={j_model:+.3f} ({elapsed:.0f}s)", flush=True)

    # Aggregate
    te = sum(x["easy_total"] for x in instances)
    tec = sum(x["easy_correct"] for x in instances)
    th = sum(x["hard_total"] for x in instances)
    tho = sum(x["hard_opt"] for x in instances)

    print(f"\n  RESULTS: Easy {tec}/{te} ({tec/max(te,1):.1%}) | Hard {tho}/{th} ({tho/max(th,1):.1%})", flush=True)

    return {
        "condition": condition_name,
        "model": model_id,
        "config": {
            "kappa_range": list(config.kappa_range),
            "lam_alpha_ratio_range": list(config.lam_alpha_ratio_range) if config.lam_alpha_ratio_range else None,
        },
        "easy": {"total": te, "correct": tec, "accuracy": tec / max(te, 1)},
        "hard": {"total": th, "opt": tho, "planning_rate": tho / max(th, 1)},
        "instances": instances,
    }


# ================================================================
# Main
# ================================================================

def main():
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "financial_gym", ".env"
    )
    vals = dotenv_values(env_path)

    config = GeneratorConfig.planning_zone()

    models_to_run = []

    if model_arg in ("haiku", "all"):
        import anthropic
        client = anthropic.Anthropic(api_key=vals["ANTHROPIC_API_KEY"])
        models_to_run.append(("haiku", client, "claude-haiku-4-5-20251001", call_anthropic))

    if model_arg in ("gpt4o", "all"):
        import openai
        oai_key = vals.get("OPENAI_API_KEY", "")
        if not oai_key or oai_key == "YOUR_OPENAI_KEY_HERE":
            print("ERROR: Set OPENAI_API_KEY in financial_gym/.env")
            sys.exit(1)
        client = openai.OpenAI(api_key=oai_key)
        models_to_run.append(("gpt4o", client, "gpt-4o", call_openai))

    if model_arg in ("sonnet", "all"):
        import anthropic
        client = anthropic.Anthropic(api_key=vals["ANTHROPIC_API_KEY"])
        models_to_run.append(("sonnet", client, "claude-sonnet-4-20250514", call_anthropic))

    if model_arg in ("opus", "all"):
        import anthropic
        client = anthropic.Anthropic(api_key=vals["ANTHROPIC_API_KEY"])
        models_to_run.append(("opus", client, "claude-opus-4-20250514", call_anthropic))

    if not models_to_run:
        print(f"Unknown model: {model_arg}. Use: haiku, sonnet, gpt4o, opus, all")
        sys.exit(1)

    # Support custom seed range: python script.py gpt4o 30 499
    seed_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seed_end = int(sys.argv[3]) if len(sys.argv) > 3 else None
    N = (seed_end - seed_start + 1) if seed_end is not None else 30

    for name, client, model_id, call_fn in models_to_run:
        result = run_eval(f"planning_{name}", config, call_fn, client, model_id, N=N, seed_start=seed_start)
        # Merge with existing data if we're extending
        output_path = f"docs/data/eval_planning_{name}_n500.json"
        existing_path = f"docs/data/eval_planning_{name}.json"
        if seed_start > 0 and os.path.exists(existing_path):
            with open(existing_path) as f:
                existing = json.load(f)
            result['instances'] = existing['instances'] + result['instances']
            result['instances'].sort(key=lambda x: x['seed'])
            te = sum(x['easy_total'] for x in result['instances'])
            tec = sum(x['easy_correct'] for x in result['instances'])
            th = sum(x['hard_total'] for x in result['instances'])
            tho = sum(x['hard_opt'] for x in result['instances'])
            result['easy'] = {'total': te, 'correct': tec, 'accuracy': tec / max(te, 1)}
            result['hard'] = {'total': th, 'opt': tho, 'planning_rate': tho / max(th, 1)}
            print(f"\nMerged with existing N={len(existing['instances'])} → total N={len(result['instances'])}")
            print(f"  Easy: {tec}/{te} ({tec/max(te,1):.1%}) | Hard: {tho}/{th} ({tho/max(th,1):.1%})")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
