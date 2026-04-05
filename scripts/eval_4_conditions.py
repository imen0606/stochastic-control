"""Run 4 evaluation conditions: 2 zones × 2 prompts.

Run 1: Planning zone + original prompt
Run 2: Planning zone + context prompt
Run 3: Control zone + original prompt
Run 4: Control zone + context prompt

Usage:
    python -u scripts/eval_4_conditions.py [condition]
    condition: 1, 2, 3, or 4 (run one at a time for parallel execution)
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import dotenv_values
import anthropic

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
# Prompts
# ================================================================

def setup_prompt_context(problem):
    """Setup prompt WITH strategy context (mean-reversion)."""
    ir_label = "ON" if problem.initial_regime == 1 else "OFF"
    return (
        f"You are managing a mean-reversion trading strategy over T={problem.T} steps. "
        f"The signal Z_t represents a quantity (such as a spread or deviation) that "
        f"tends to revert toward zero over time. You do not know how fast it reverts "
        f"or how noisy it is — you must estimate this from the observations.\n"
        f"\n"
        f"RULES:\n"
        f"- At each step t, you choose s_t = 1 (ON) or s_t = 0 (OFF).\n"
        f"- If ON (s_t=1): you earn the PnL for that step. "
        f"Expected PnL = {problem.alpha:.4f} x Z_t.\n"
        f"- If OFF (s_t=0): you earn nothing (PnL = 0).\n"
        f"- Every time you SWITCH (change from ON to OFF or OFF to ON), "
        f"you pay a cost of {problem.lam:.4f}.\n"
        f"- If you stay in the same state, no switching cost.\n"
        f"\n"
        f"YOUR GOAL: Maximize total profit = sum of PnL earned "
        f"minus switching costs paid.\n"
        f"\n"
        f"PARAMETERS:\n"
        f"  Signal strength: alpha = {problem.alpha:.4f}\n"
        f"  Switching cost:  lambda = {problem.lam:.4f}\n"
        f"  Starting state:  s_{{-1}} = {problem.initial_regime} ({ir_label})\n"
        f"  Horizon:         T = {problem.T} steps\n"
        f"\n"
        f"You will receive one signal observation Z_t at a time.\n"
        f"State your decision as: s_t = 0 or s_t = 1"
    )


# ================================================================
# Evaluation
# ================================================================

def run_eval(condition_name, config, prompt_fn, client, N=30):
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
            user_msg = step_prompt(t, p.z_path[t], prev)
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
        1: ("planning_original", GeneratorConfig.planning_zone(), setup_prompt),
        2: ("planning_context", GeneratorConfig.planning_zone(), setup_prompt_context),
        3: ("control_original", GeneratorConfig.control_zone(), setup_prompt),
        4: ("control_context", GeneratorConfig.control_zone(), setup_prompt_context),
    }

    if condition in conditions:
        name, config, prompt_fn = conditions[condition]
        result = run_eval(name, config, prompt_fn, client, N=30)
        output_path = f"docs/data/eval_{name}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {output_path}", flush=True)
    elif condition == 0:
        # Run all sequentially
        for cond_id in [1, 2, 3, 4]:
            name, config, prompt_fn = conditions[cond_id]
            result = run_eval(name, config, prompt_fn, client, N=30)
            output_path = f"docs/data/eval_{name}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
