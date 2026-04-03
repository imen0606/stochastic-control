"""Opus evaluation with two-bucket analysis (easy vs hard decisions).

Config: κ=0.1, λ=0.15, α=0.30, T=25 (λ/α=0.50, greedy actively switches)
N=30 instances, ~750 decisions total, ~75 hard decisions.

Classifies each decision as:
  Easy: greedy and optimal agree (answer is obvious)
  Hard: greedy and optimal disagree (planning matters)

Then measures Opus accuracy on each bucket separately.
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import dotenv_values
import anthropic

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig, RegimeSwitchingGenerator, _compute_z_grid,
)
from financial_gym.problems.regime_switching.verifier import (
    _compute_realized_utility, _linear_utility, _parse_decision,
)
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.agents.greedy_agent import GreedyAgent


def main():
    vals = dotenv_values(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "financial_gym", ".env"
    ))
    client = anthropic.Anthropic(api_key=vals["ANTHROPIC_API_KEY"])

    config = GeneratorConfig(
        T_range=(25, 25), lam_range=(0.15, 0.15), alpha_range=(0.30, 0.30),
        kappa_range=(0.1, 0.1), sigma_z_range=(0.2, 0.2),
        grid_size=200, n_quad_nodes=20,
    )
    gen = RegimeSwitchingGenerator(config)
    N = 30
    T = 25

    print("=" * 70)
    print("OPUS EVALUATION — Two-Bucket Analysis")
    print(f"Config: kappa=0.1, lambda=0.15, alpha=0.30, T=25, N={N}")
    print(f"Greedy threshold: lambda/alpha = 0.50")
    print("=" * 70)
    print(flush=True)

    # Pre-compute all problems and baseline agents
    problems = []
    opt_decisions = []
    gre_decisions = []

    for seed in range(N):
        p = gen.sample(seed=seed)
        problems.append(p)
        opt_decisions.append(OptimalAgent().decide(p))
        gre_decisions.append(GreedyAgent().decide(p))

    # Run Opus
    opus_decisions = []
    total_start = time.time()

    for i, p in enumerate(problems):
        start = time.time()
        system_msg = setup_prompt(p)
        messages = []
        prev = p.initial_regime
        decisions = []

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
            except Exception as e:
                text = f"Error: {e}"
            messages.append({"role": "assistant", "content": text})
            d = _parse_decision(text)
            decisions.append(d)
            prev = d

        opus_decisions.append(np.array(decisions, dtype=np.int8))
        elapsed = time.time() - start

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Instance {i+1}/{N} ({elapsed:.0f}s)", flush=True)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} minutes", flush=True)

    # ================================================================
    # TWO-BUCKET ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("TWO-BUCKET ANALYSIS")
    print("=" * 70 + "\n")

    easy_opus_correct = 0
    easy_total = 0
    hard_opus_matches_optimal = 0
    hard_opus_matches_greedy = 0
    hard_opus_matches_neither = 0
    hard_total = 0

    # Also track per-instance for reporting
    instance_details = []

    for i in range(N):
        p = problems[i]
        z_grid = _compute_z_grid(p.theta, p.sigma_z, p.kappa, 200)
        opus_d = opus_decisions[i]
        opt_d = opt_decisions[i]
        gre_d = gre_decisions[i]

        opus_prev = p.initial_regime
        gre_prev = p.initial_regime
        instance_easy_correct = 0
        instance_easy_total = 0
        instance_hard_opt = 0
        instance_hard_gre = 0
        instance_hard_neither = 0
        instance_hard_total = 0

        for t in range(T):
            z_idx = int(np.argmin(np.abs(z_grid - p.z_path[t])))

            # What would optimal do from OPUS's state?
            opt_from_opus = int(p.optimal_policy_table[t, z_idx, opus_prev])

            # What would greedy do from OPUS's state?
            q_off = 0.0 - (p.lam if 0 != opus_prev else 0.0)
            q_on = p.alpha * p.z_path[t] - (p.lam if 1 != opus_prev else 0.0)
            gre_from_opus = 1 if q_on > q_off else 0

            # Classify this decision
            if opt_from_opus == gre_from_opus:
                # Easy: both agree
                easy_total += 1
                instance_easy_total += 1
                if opus_d[t] == opt_from_opus:
                    easy_opus_correct += 1
                    instance_easy_correct += 1
            else:
                # Hard: they disagree
                hard_total += 1
                instance_hard_total += 1
                if opus_d[t] == opt_from_opus:
                    hard_opus_matches_optimal += 1
                    instance_hard_opt += 1
                elif opus_d[t] == gre_from_opus:
                    hard_opus_matches_greedy += 1
                    instance_hard_gre += 1
                else:
                    hard_opus_matches_neither += 1
                    instance_hard_neither += 1

            opus_prev = int(opus_d[t])

        instance_details.append({
            "seed": i,
            "easy_total": instance_easy_total,
            "easy_correct": instance_easy_correct,
            "hard_total": instance_hard_total,
            "hard_matches_optimal": instance_hard_opt,
            "hard_matches_greedy": instance_hard_gre,
            "hard_matches_neither": instance_hard_neither,
            "opus_decisions": opus_d.tolist(),
        })

    # Results
    easy_acc = easy_opus_correct / easy_total if easy_total > 0 else 0
    hard_opt_rate = hard_opus_matches_optimal / hard_total if hard_total > 0 else 0
    hard_gre_rate = hard_opus_matches_greedy / hard_total if hard_total > 0 else 0
    hard_neither_rate = hard_opus_matches_neither / hard_total if hard_total > 0 else 0

    print(f"EASY DECISIONS (greedy = optimal, answer is obvious):")
    print(f"  Total: {easy_total}")
    print(f"  Opus correct: {easy_opus_correct}/{easy_total} = {easy_acc:.1%}")
    print()
    print(f"HARD DECISIONS (greedy != optimal, planning matters):")
    print(f"  Total: {hard_total}")
    print(f"  Opus matches OPTIMAL (planned):     {hard_opus_matches_optimal}/{hard_total} = {hard_opt_rate:.1%}")
    print(f"  Opus matches GREEDY  (reacted):     {hard_opus_matches_greedy}/{hard_total} = {hard_gre_rate:.1%}")
    print(f"  Opus matches NEITHER (other error): {hard_opus_matches_neither}/{hard_total} = {hard_neither_rate:.1%}")
    print()

    # Statistical test: is Opus's hard accuracy > 0? (better than pure greedy)
    from scipy.stats import binomtest
    binom_result = binomtest(hard_opus_matches_optimal, hard_total, 0.5, alternative="two-sided")
    print(f"STATISTICAL TEST: Is Opus planning rate ({hard_opt_rate:.1%}) different from chance (50%)?")
    print(f"  Binomial test p-value: {binom_result.pvalue:.4f}")
    print(f"  95% CI: [{binom_result.proportion_ci().low:.1%}, {binom_result.proportion_ci().high:.1%}]")
    print()

    # J-value comparison
    print("J-VALUE COMPARISON:")
    for name, decisions_list in [("Optimal", opt_decisions), ("Greedy", gre_decisions), ("Opus", opus_decisions)]:
        j_vals = []
        for idx in range(N):
            j = _compute_realized_utility(
                decisions_list[idx], problems[idx].x_path,
                problems[idx].lam, problems[idx].initial_regime, _linear_utility,
            )
            j_vals.append(j)
        print(f"  {name:>8}: mean J = {np.mean(j_vals):+.4f} (std = {np.std(j_vals):.4f})")

    # Save
    results = {
        "config": {"kappa": 0.1, "lam": 0.15, "alpha": 0.30, "T": 25, "N": N},
        "easy": {"total": easy_total, "opus_correct": easy_opus_correct, "accuracy": easy_acc},
        "hard": {
            "total": hard_total,
            "matches_optimal": hard_opus_matches_optimal,
            "matches_greedy": hard_opus_matches_greedy,
            "matches_neither": hard_opus_matches_neither,
            "planning_rate": hard_opt_rate,
        },
        "instances": instance_details,
    }
    output_path = "docs/eval_opus_buckets.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
