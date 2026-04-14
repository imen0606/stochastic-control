#!/usr/bin/env python3
"""
Offline reward simulation: validates old vs composite reward functions.

Generates episodes locally (no GPU), runs 5 strategies through both
reward functions, and prints a comparison table. The composite reward
must rank: optimal > planning > greedy > always-ON/OFF > random.

Usage:
    python scripts/test_reward.py
    python scripts/test_reward.py --episodes 20
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    _compute_z_grid,
)
from financial_gym.problems.regime_switching.verifier import (
    _compute_realized_utility,
    _linear_utility,
    _parse_decision,
)
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.agents.greedy_agent import GreedyAgent


# ── Strategy generators ──────────────────────────────────────────────

def strategy_optimal(problem):
    return OptimalAgent().decide(problem)

def strategy_greedy(problem):
    return GreedyAgent().decide(problem)

def strategy_always_on(problem):
    return np.ones(problem.T, dtype=np.int8)

def strategy_always_off(problem):
    return np.zeros(problem.T, dtype=np.int8)

def strategy_random(problem, seed_offset=0):
    rng = np.random.default_rng(problem.seed + seed_offset)
    return rng.integers(0, 2, size=problem.T).astype(np.int8)

def strategy_planning(problem):
    """Greedy everywhere except hard decisions, where it matches optimal.

    This simulates a model that has learned to plan on hard decisions
    but otherwise behaves greedily. It's the target behaviour for training.
    """
    greedy_d = strategy_greedy(problem)
    optimal_d = strategy_optimal(problem)
    # Start with greedy, override on hard decisions
    decisions = greedy_d.copy()
    z_grid = _compute_z_grid(problem.theta, problem.sigma_z, problem.kappa, 200)
    prev_greedy = problem.initial_regime
    prev_optimal = problem.initial_regime
    for t in range(problem.T):
        # Recompute greedy action from greedy's own trajectory
        z = problem.z_path[t]
        q_off = 0.0 - (problem.lam if 0 != prev_greedy else 0.0)
        q_on = problem.alpha * z - (problem.lam if 1 != prev_greedy else 0.0)
        ga = 1 if q_on > q_off else 0

        # Optimal action from optimal's own trajectory
        zi = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
        oa = int(problem.optimal_policy_table[t, zi, prev_optimal])

        if ga != oa:
            # Hard decision: override with optimal
            decisions[t] = oa
        else:
            decisions[t] = ga

        prev_greedy = ga
        prev_optimal = oa
    return decisions


# ── Reward functions ─────────────────────────────────────────────────

def compute_j(decisions, problem):
    """Compute realized utility J for a decision sequence."""
    return _compute_realized_utility(
        decisions, problem.x_path, problem.lam,
        problem.initial_regime, _linear_utility,
    )


def random_baseline_j(problem, n=20):
    """Average J over n random policies."""
    rng = np.random.default_rng(problem.seed + 99999)
    js = []
    for _ in range(n):
        d = rng.integers(0, 2, size=problem.T).astype(np.int8)
        js.append(compute_j(d, problem))
    return np.mean(js)


def old_reward(j_model, j_random, j_optimal):
    """Original: (J_model - J_random) / (J_optimal - J_random), clipped [-2, 2]."""
    denom = j_optimal - j_random
    if abs(denom) < 0.001:
        return 1.0 if abs(j_model - j_optimal) < 0.001 else 0.0
    return float(np.clip((j_model - j_random) / denom, -2.0, 2.0))


def _classify_decisions(decisions, problem):
    """Classify each step as easy/hard and check if model matches optimal.

    Returns (easy_matches, easy_total, hard_matches, hard_total).
    """
    z_grid = _compute_z_grid(problem.theta, problem.sigma_z, problem.kappa, 200)
    prev_greedy = problem.initial_regime
    prev_optimal = problem.initial_regime
    easy_matches = easy_total = hard_matches = hard_total = 0

    for t in range(problem.T):
        z = problem.z_path[t]
        # Greedy action from greedy's trajectory
        q_off = 0.0 - (problem.lam if 0 != prev_greedy else 0.0)
        q_on = problem.alpha * z - (problem.lam if 1 != prev_greedy else 0.0)
        ga = 1 if q_on > q_off else 0

        # Optimal action from optimal's trajectory
        zi = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
        oa = int(problem.optimal_policy_table[t, zi, prev_optimal])

        if ga != oa:
            hard_total += 1
            if decisions[t] == oa:
                hard_matches += 1
        else:
            easy_total += 1
            if decisions[t] == oa:
                easy_matches += 1

        prev_greedy = ga
        prev_optimal = oa

    return easy_matches, easy_total, hard_matches, hard_total


def composite_reward(j_model, j_random, j_optimal, decisions, problem):
    """Decision-level reward with comprehension gate.

    reward = easy_accuracy + hard_accuracy * 1.5  (if easy_acc >= 0.9)
    reward = easy_accuracy                         (if easy_acc < 0.9)

    The gate ensures degenerate strategies (always_on/off, random) with
    ~50% easy accuracy never get the hard bonus, even if they accidentally
    match some hard decisions. Only models with basic comprehension
    (greedy-level) can earn the planning bonus.

    Range: [0, 2.5] — 1.0 for comprehension + up to 1.5 for planning.
    """
    em, et, hm, ht = _classify_decisions(decisions, problem)

    easy_acc = em / max(et, 1)
    hard_acc = hm / max(ht, 1)
    bonus_weight = 1.5
    gate_threshold = 0.9

    if easy_acc >= gate_threshold:
        bonus = hard_acc * bonus_weight
    else:
        bonus = 0.0

    reward = easy_acc + bonus
    return reward, easy_acc, bonus, ht, hm


# ── Main simulation ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    args = parser.parse_args()

    gen = RegimeSwitchingGenerator(GeneratorConfig.planning_zone())

    strategies = {
        "optimal":    strategy_optimal,
        "planning":   strategy_planning,
        "greedy":     strategy_greedy,
        "always_on":  strategy_always_on,
        "always_off": strategy_always_off,
        "random":     lambda p: strategy_random(p, seed_offset=42),
    }

    # Accumulate per-strategy results
    results = {name: {"old": [], "composite": [], "base": [], "bonus": [],
                       "hard_total": [], "hard_matches": []}
               for name in strategies}

    episodes_used = 0
    seed = args.seed_start

    print(f"Generating {args.episodes} episodes from planning zone...\n")

    while episodes_used < args.episodes:
        problem = gen.sample(seed=seed)
        seed += 1

        # Only use episodes with hard decisions (same filter as training)
        greedy_d = strategy_greedy(problem)
        optimal_d = strategy_optimal(problem)
        z_grid = _compute_z_grid(problem.theta, problem.sigma_z, problem.kappa, 200)
        prev_g = problem.initial_regime
        prev_o = problem.initial_regime
        has_hard = False
        for t in range(problem.T):
            z = problem.z_path[t]
            q_off = 0.0 - (problem.lam if 0 != prev_g else 0.0)
            q_on = problem.alpha * z - (problem.lam if 1 != prev_g else 0.0)
            ga = 1 if q_on > q_off else 0
            zi = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
            oa = int(problem.optimal_policy_table[t, zi, prev_o])
            if ga != oa:
                has_hard = True
                break
            prev_g = ga
            prev_o = oa

        if not has_hard:
            continue

        episodes_used += 1
        j_random = random_baseline_j(problem)
        j_optimal = compute_j(strategy_optimal(problem), problem)

        for name, strat_fn in strategies.items():
            d = strat_fn(problem)
            j = compute_j(d, problem)

            r_old = old_reward(j, j_random, j_optimal)
            r_comp, base, bonus, ht, hm = composite_reward(
                j, j_random, j_optimal, d, problem
            )

            results[name]["old"].append(r_old)
            results[name]["composite"].append(r_comp)
            results[name]["base"].append(base)
            results[name]["bonus"].append(bonus)
            results[name]["hard_total"].append(ht)
            results[name]["hard_matches"].append(hm)

    # ── Print results ────────────────────────────────────────────────

    print(f"{'='*80}")
    print(f"REWARD COMPARISON ({episodes_used} episodes with hard decisions)")
    print(f"{'='*80}\n")

    # Summary table
    header = f"{'Strategy':<14} {'Old Reward':>12} {'Composite':>12} {'Base':>8} {'Bonus':>8} {'Hard':>8}"
    print(header)
    print("-" * len(header))

    for name in ["optimal", "planning", "greedy", "always_on", "always_off", "random"]:
        r = results[name]
        avg_old = np.mean(r["old"])
        avg_comp = np.mean(r["composite"])
        avg_base = np.mean(r["base"])
        avg_bonus = np.mean(r["bonus"])
        avg_hard = f"{sum(r['hard_matches'])}/{sum(r['hard_total'])}"
        print(f"{name:<14} {avg_old:>12.3f} {avg_comp:>12.3f} {avg_base:>8.3f} {avg_bonus:>8.3f} {avg_hard:>8}")

    # Check ranking
    print(f"\n{'='*80}")
    print("RANKING CHECK")
    print(f"{'='*80}\n")

    def avg(name, key):
        return np.mean(results[name][key])

    print("Old reward ranking (BROKEN — should be optimal > planning > greedy > always_* > random):")
    old_ranking = sorted(strategies.keys(), key=lambda n: avg(n, "old"), reverse=True)
    for i, name in enumerate(old_ranking, 1):
        print(f"  {i}. {name:<14} {avg(name, 'old'):>8.3f}")

    print("\nComposite reward ranking (should be: optimal > planning > greedy > always_* > random):")
    comp_ranking = sorted(strategies.keys(), key=lambda n: avg(n, "composite"), reverse=True)
    for i, name in enumerate(comp_ranking, 1):
        print(f"  {i}. {name:<14} {avg(name, 'composite'):>8.3f}")

    # Verify key invariants
    print(f"\n{'='*80}")
    print("INVARIANT CHECKS")
    print(f"{'='*80}\n")

    checks = [
        ("optimal > greedy (composite)",
         avg("optimal", "composite") > avg("greedy", "composite")),
        ("planning > greedy (composite)",
         avg("planning", "composite") > avg("greedy", "composite")),
        ("optimal > always_on (composite)",
         avg("optimal", "composite") > avg("always_on", "composite")),
        ("optimal > always_off (composite)",
         avg("optimal", "composite") > avg("always_off", "composite")),
        ("planning > always_on (composite)",
         avg("planning", "composite") > avg("always_on", "composite")),
        ("planning > always_off (composite)",
         avg("planning", "composite") > avg("always_off", "composite")),
        ("greedy > always_on (composite) OR close",
         avg("greedy", "composite") >= avg("always_on", "composite") - 0.1),
        ("OLD reward is broken: always_off > greedy (expected, demonstrates bug)",
         avg("always_off", "old") > avg("greedy", "old") - 0.05),
    ]

    all_pass = True
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {desc}")

    print(f"\n{'='*80}")
    if all_pass:
        print("ALL CHECKS PASSED — composite reward fixes the ranking.")
    else:
        print("SOME CHECKS FAILED — composite reward needs adjustment.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
