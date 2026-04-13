#!/usr/bin/env python3
"""
Evaluate energy dispatch prompt on the same gym infrastructure.

Same OU process, same solver, same verifier — only the prompt changes.
Tests whether domain framing affects planning capability.

Run:  python scripts/eval_energy_dispatch.py [condition]
  1 = Planning zone, energy prompt
  2 = Planning zone, original (finance) prompt (replication)
  0 = Both
"""
import json
import os
import sys
import time

import anthropic
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
from financial_gym.problems.regime_switching.prompts import step_prompt, setup_prompt


# ================================================================
# Prompts
# ================================================================

def setup_prompt_energy(problem):
    """Energy dispatch prompt — same math, different domain."""
    ir_label = "ON" if problem.initial_regime == 1 else "OFF"
    return (
        f"You are operating a power generator over T={problem.T} time periods.\n"
        f"\n"
        f"RULES:\n"
        f"- At each period t, you choose s_t = 1 (ON = generator running) "
        f"or s_t = 0 (OFF = generator idle).\n"
        f"- If ON (s_t=1): you sell electricity and earn revenue for that period. "
        f"Expected revenue = {problem.alpha:.4f} x D_t, where D_t is the "
        f"demand/price signal.\n"
        f"- If OFF (s_t=0): the generator is idle, you earn nothing.\n"
        f"- Every time you SWITCH (start up or shut down the generator), "
        f"you pay a startup/shutdown cost of {problem.lam:.4f}.\n"
        f"- If the generator stays in the same state, no switching cost.\n"
        f"\n"
        f"YOUR GOAL: Maximize total profit = sum of revenue earned "
        f"minus startup/shutdown costs paid.\n"
        f"\n"
        f"PARAMETERS:\n"
        f"  Revenue factor:       alpha = {problem.alpha:.4f}\n"
        f"  Startup/shutdown cost: lambda = {problem.lam:.4f}\n"
        f"  Starting state:       s_{{-1}} = {problem.initial_regime} ({ir_label})\n"
        f"  Horizon:              T = {problem.T} periods\n"
        f"\n"
        f"You will receive one demand/price observation D_t at a time.\n"
        f"State your decision as: s_t = 0 or s_t = 1"
    )


def step_prompt_energy(t, z_t, prev_regime):
    """Step prompt with energy framing."""
    state = "ON" if prev_regime == 1 else "OFF"
    return (
        f"t={t} | D_t = {z_t:+.4f} | "
        f"Generator is currently {state} (s_prev = {prev_regime})"
    )


# ================================================================
# Evaluation (same as eval_4_conditions.py)
# ================================================================

def run_eval(condition_name, config, prompt_fn, step_fn, client, N=30):
    """Run one evaluation condition."""
    gen = RegimeSwitchingGenerator(config)

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name}")
    print(f"N={N}, config: kappa_range={config.kappa_range}")
    print(f"{'='*60}", flush=True)

    instances = []
    for i in range(N):
        p = gen.sample(seed=i)
        T = p.T
        system_msg = prompt_fn(p)
        messages = []
        prev = p.initial_regime
        decisions = []
        raw_texts = []

        t0 = time.time()
        for t in range(T):
            user_msg = step_fn(t, p.z_path[t], prev)
            messages.append({"role": "user", "content": user_msg})
            try:
                resp = client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=300,
                    system=system_msg,
                    messages=messages,
                )
                text = resp.content[0].text
            except anthropic._exceptions.OverloadedError:
                time.sleep(30)
                try:
                    resp = client.messages.create(
                        model="claude-opus-4-20250514",
                        max_tokens=300,
                        system=system_msg,
                        messages=messages,
                    )
                    text = resp.content[0].text
                except Exception as e:
                    text = f"Error: {e}"
            except Exception as e:
                text = f"Error: {e}"
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
        j_opus = _compute_realized_utility(opus_d, p.x_path, p.lam, p.initial_regime, _linear_utility)
        j_opt = _compute_realized_utility(OptimalAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility)
        j_gre = _compute_realized_utility(GreedyAgent().decide(p), p.x_path, p.lam, p.initial_regime, _linear_utility)

        instances.append({
            "seed": i, "kappa": p.kappa, "alpha": p.alpha, "lam": p.lam,
            "sigma_z": p.sigma_z, "T": T,
            "decisions": decisions, "raw_texts": raw_texts,
            "easy_correct": ec, "easy_total": et,
            "hard_opt": ho, "hard_total": ht,
            "j_opus": j_opus, "j_opt": j_opt, "j_gre": j_gre,
        })
        print(f"  [{i+1}/{N}] kappa={p.kappa:.2f} T={T} easy={ec}/{et} hard={ho}/{ht} J={j_opus:+.3f} ({elapsed:.0f}s)", flush=True)

    # Aggregate
    te = sum(x["easy_total"] for x in instances)
    tec = sum(x["easy_correct"] for x in instances)
    th = sum(x["hard_total"] for x in instances)
    tho = sum(x["hard_opt"] for x in instances)

    print(f"\n  RESULTS: Easy {tec}/{te} ({tec/max(te,1):.1%}) | Hard {tho}/{th} ({tho/max(th,1):.1%})", flush=True)

    return {
        "condition": condition_name,
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
    condition = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    vals = dotenv_values(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "financial_gym", ".env"
    ))
    client = anthropic.Anthropic(api_key=vals["ANTHROPIC_API_KEY"])

    conditions = {
        1: ("energy_planning", GeneratorConfig.planning_zone(), setup_prompt_energy, step_prompt_energy),
        2: ("finance_planning_replication", GeneratorConfig.planning_zone(), setup_prompt, step_prompt),
    }

    if condition in conditions:
        name, config, prompt_fn, step_fn = conditions[condition]
        result = run_eval(name, config, prompt_fn, step_fn, client, N=30)
        output_path = f"docs/data/eval_{name}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {output_path}", flush=True)
    elif condition == 0:
        for cond_id in [1, 2]:
            name, config, prompt_fn, step_fn = conditions[cond_id]
            result = run_eval(name, config, prompt_fn, step_fn, client, N=30)
            output_path = f"docs/data/eval_{name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nSaved to {output_path}", flush=True)
    else:
        print(f"Unknown condition {condition}. Use 0 (all), 1 (energy), 2 (finance replication)")


if __name__ == "__main__":
    main()
