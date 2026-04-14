#!/usr/bin/env python3
"""
Generate SFT warmup data for GRPO training cold-start.

Produces synthetic chain-of-thought traces showing Bellman-optimal decisions
with varied reasoning templates. The goal is to teach the model the *concept*
of switching (paying lambda to capture a trend) so GRPO's K completions
diverge under temperature sampling.

Usage:
    python scripts/generate_sft_data.py                    # 300 episodes
    python scripts/generate_sft_data.py -n 500 -o my.jsonl # custom
    python scripts/generate_sft_data.py --inspect 5        # inspect 5 episodes
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.problems.regime_switching.verifier import (
    _compute_realized_utility,
    _linear_utility,
)


# ---------------------------------------------------------------------------
# Chain-of-thought reasoning templates
# ---------------------------------------------------------------------------

def _trend_description(z_history: np.ndarray) -> tuple[str, int]:
    """Describe the recent signal trend."""
    if len(z_history) < 2:
        return "unclear", 0
    # Count consecutive same-sign steps from the end
    last_sign = 1 if z_history[-1] > 0 else -1
    count = 0
    for z in reversed(z_history):
        if (z > 0 and last_sign > 0) or (z <= 0 and last_sign <= 0):
            count += 1
        else:
            break
    direction = "positive" if last_sign > 0 else "negative"
    return direction, count


def _generate_cot(
    problem, t: int, action: int, prev: int, rng: np.random.Generator
) -> str:
    """Generate chain-of-thought reasoning for one optimal decision."""
    z = problem.z_path[t]
    alpha = problem.alpha
    lam = problem.lam
    pnl = alpha * z
    switching = action != prev
    remaining = problem.T - t - 1

    state_label = "ON" if prev == 1 else "OFF"
    action_label = "ON" if action == 1 else "OFF"

    trend_dir, trend_len = _trend_description(problem.z_path[:t + 1])
    z_history = problem.z_path[:t + 1]

    # Pick reasoning style
    if switching:
        return _cot_switch(z, pnl, lam, alpha, state_label, action_label,
                           action, prev, trend_dir, trend_len, remaining, rng)
    elif action == prev:
        return _cot_stay(z, pnl, lam, alpha, state_label, action_label,
                         action, prev, trend_dir, trend_len, remaining, rng)
    else:
        # Should not happen (action != prev is switching)
        return f"s_t = {action}"


def _cot_switch(z, pnl, lam, alpha, state_label, action_label,
                action, prev, trend_dir, trend_len, remaining, rng) -> str:
    """Reasoning for switching state (paying lambda)."""
    templates = [
        # Template 1: Signal-trend based
        lambda: (
            f"The signal Z_t = {z:+.4f} gives expected PnL = {pnl:+.4f} if ON. "
            f"I'm currently {state_label}. Switching to {action_label} costs lambda = {lam:.4f}. "
            f"The signal has been {trend_dir} for {trend_len} step{'s' if trend_len != 1 else ''}, "
            f"and with {remaining} steps remaining, paying the switching cost now should be "
            f"recovered over the coming steps.\n\ns_t = {action}"
        ),
        # Template 2: Cost-benefit
        lambda: (
            f"Z_t = {z:+.4f}, so expected PnL if ON = {alpha:.4f} * {z:+.4f} = {pnl:+.4f}. "
            f"Currently {state_label}. To switch to {action_label}, I pay {lam:.4f}. "
            f"With the signal trending {trend_dir} and {remaining} steps left, "
            f"the cumulative benefit of being {action_label} outweighs the one-time cost.\n\n"
            f"s_t = {action}"
        ),
        # Template 3: Amortization reasoning
        lambda: (
            f"Signal: {z:+.4f}. Expected PnL if ON: {pnl:+.4f}. "
            f"Switching from {state_label} to {action_label} costs {lam:.4f}. "
            f"If the signal stays around this level for even a few more steps, "
            f"the gains will cover the switching cost. "
            f"With {remaining} steps remaining, this is worth the switch.\n\n"
            f"s_t = {action}"
        ),
        # Template 4: Direct calculation
        lambda: (
            f"At t with Z_t = {z:+.4f}, the expected per-step PnL if ON is {pnl:+.4f}. "
            f"Switching cost is {lam:.4f}. "
            f"The break-even is {lam / max(abs(pnl), 0.001):.1f} steps. "
            f"I have {remaining} steps left, so switching to {action_label} is justified.\n\n"
            f"s_t = {action}"
        ),
        # Template 5: Trend momentum
        lambda: (
            f"Z_t = {z:+.4f}. The signal has maintained a {trend_dir} trend "
            f"({trend_len} steps). Currently {state_label} — switching to {action_label} "
            f"costs {lam:.4f}, but the persistent trend suggests this will pay off "
            f"over the remaining {remaining} steps.\n\n"
            f"s_t = {action}"
        ),
    ]
    return rng.choice(templates)()


def _cot_stay(z, pnl, lam, alpha, state_label, action_label,
              action, prev, trend_dir, trend_len, remaining, rng) -> str:
    """Reasoning for staying in current state."""
    # Determine if this is "stay because it's obviously right" or "stay despite temptation"
    if prev == 1 and pnl > 0:
        # ON and signal is positive — easy stay
        return _cot_stay_easy(z, pnl, lam, alpha, state_label, action, remaining, rng)
    elif prev == 0 and pnl <= 0:
        # OFF and signal is negative — easy stay
        return _cot_stay_easy(z, pnl, lam, alpha, state_label, action, remaining, rng)
    else:
        # Staying despite the signal suggesting otherwise — planning insight
        return _cot_stay_planning(z, pnl, lam, alpha, state_label, action, remaining, rng)


def _cot_stay_easy(z, pnl, lam, alpha, state_label, action, remaining, rng) -> str:
    """Easy decision: staying in current state is obviously correct."""
    templates = [
        lambda: (
            f"Z_t = {z:+.4f}, expected PnL if ON = {pnl:+.4f}. "
            f"Currently {state_label}, no reason to switch. Staying.\n\n"
            f"s_t = {action}"
        ),
        lambda: (
            f"Signal {z:+.4f}. PnL if ON = {pnl:+.4f}. "
            f"Already {state_label} — staying avoids the switching cost of {lam:.4f}. "
            f"No change needed.\n\ns_t = {action}"
        ),
        lambda: (
            f"Z_t = {z:+.4f}. Staying {state_label} costs nothing and "
            f"{'earns ' + f'{pnl:+.4f}' if action == 1 else 'avoids negative PnL'}. "
            f"s_t = {action}"
        ),
        lambda: (
            f"The signal is {z:+.4f}, giving expected PnL = {pnl:+.4f} if ON. "
            f"I'm {state_label} and the current position is correct. "
            f"No switching cost needed.\n\ns_t = {action}"
        ),
        lambda: (
            f"Z_t = {z:+.4f}. Currently {state_label}. "
            f"{'Positive' if pnl > 0 else 'Negative'} signal "
            f"{'supports' if (action == 1 and pnl > 0) or (action == 0 and pnl <= 0) else 'confirms'} "
            f"staying. s_t = {action}"
        ),
    ]
    return rng.choice(templates)()


def _cot_stay_planning(z, pnl, lam, alpha, state_label, action, remaining, rng) -> str:
    """Planning insight: staying despite signal suggesting otherwise."""
    templates = [
        lambda: (
            f"Z_t = {z:+.4f}, expected PnL if ON = {pnl:+.4f}. "
            f"Switching would cost {lam:.4f}. Although the signal "
            f"{'looks positive' if pnl > 0 else 'looks negative'}, "
            f"the switching cost is high relative to the expected gain. "
            f"Better to wait for a stronger signal.\n\ns_t = {action}"
        ),
        lambda: (
            f"Signal is {z:+.4f} ({'positive' if z > 0 else 'negative'}). "
            f"Tempting to switch, but lambda = {lam:.4f} is costly. "
            f"The immediate {'gain' if pnl > 0 else 'loss'} of {abs(pnl):.4f} "
            f"doesn't justify the switch. Staying {state_label}.\n\n"
            f"s_t = {action}"
        ),
        lambda: (
            f"Z_t = {z:+.4f}. The {'positive signal suggests switching ON' if pnl > 0 and action == 0 else 'negative signal suggests switching OFF'}, "
            f"but at a cost of {lam:.4f}, I need the signal to be stronger "
            f"or more persistent to justify the switch. Holding position.\n\n"
            f"s_t = {action}"
        ),
        lambda: (
            f"Expected PnL if ON: {pnl:+.4f}. Switching cost: {lam:.4f}. "
            f"The break-even requires {lam / max(abs(pnl), 0.001):.1f} steps at this signal level. "
            f"Not confident enough to commit — staying {state_label}.\n\n"
            f"s_t = {action}"
        ),
        lambda: (
            f"Z_t = {z:+.4f}. Currently {state_label}. "
            f"With switching cost = {lam:.4f} and only {abs(pnl):.4f} expected PnL per step, "
            f"the risk-reward doesn't favour switching now. Patience.\n\n"
            f"s_t = {action}"
        ),
    ]
    return rng.choice(templates)()


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_sft_data(num_episodes: int, output_path: str, inspect: int = 0):
    """Generate SFT warmup JSONL data."""
    cfg = GeneratorConfig.j_gap_zone()
    gen = RegimeSwitchingGenerator(cfg)
    optimal_agent = OptimalAgent()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_turns = 0
    total_switches = 0
    j_values = []

    with open(output_path, "w") as f:
        for seed in range(1, num_episodes + 1):
            problem = gen.sample(seed=seed)
            decisions = optimal_agent.decide(problem)
            rng = np.random.default_rng(seed + 777)

            messages = [{"role": "system", "content": setup_prompt(problem)}]

            prev = problem.initial_regime
            ep_switches = 0
            for t in range(problem.T):
                # User turn
                messages.append({
                    "role": "user",
                    "content": step_prompt(t, problem.z_path[t], prev),
                })

                # Assistant turn with CoT
                a = int(decisions[t])
                cot = _generate_cot(problem, t, a, prev, rng)
                messages.append({"role": "assistant", "content": cot})

                if a != prev:
                    ep_switches += 1
                prev = a

            # Compute J for this episode
            j = _compute_realized_utility(
                decisions, problem.x_path, problem.lam,
                problem.initial_regime, _linear_utility,
            )
            j_values.append(j)
            total_turns += problem.T
            total_switches += ep_switches

            f.write(json.dumps({"messages": messages}) + "\n")

            if inspect > 0 and seed <= inspect:
                _inspect_episode(seed, problem, decisions, messages, j, ep_switches)

    # Summary statistics
    print(f"\nGenerated {num_episodes} episodes -> {output_path}")
    print(f"  Total turns:    {total_turns}")
    print(f"  Avg T:          {total_turns / num_episodes:.0f}")
    print(f"  Total switches: {total_switches} ({total_switches / total_turns * 100:.1f}% of turns)")
    print(f"  Avg switches/ep:{total_switches / num_episodes:.1f}")
    print(f"  Mean J:         {np.mean(j_values):.2f}")
    print(f"  Std J:          {np.std(j_values):.2f}")

    # Estimate token count (~4 chars per token)
    file_size = os.path.getsize(output_path)
    est_tokens = file_size / 4
    print(f"  File size:      {file_size / 1e6:.1f} MB")
    print(f"  Est. tokens:    {est_tokens / 1e6:.1f}M")


def _inspect_episode(seed, problem, decisions, messages, j, switches):
    """Print a human-readable summary of one episode."""
    print(f"\n{'=' * 60}")
    print(f"Episode seed={seed}, T={problem.T}, kappa={problem.kappa:.3f}, "
          f"alpha={problem.alpha:.3f}, lambda={problem.lam:.3f}")
    print(f"J={j:.2f}, switches={switches}")
    print(f"{'=' * 60}")

    # Show first 5 and last 2 turns
    turns_to_show = list(range(min(5, problem.T))) + list(range(max(5, problem.T - 2), problem.T))
    turns_to_show = sorted(set(turns_to_show))

    for t in turns_to_show:
        user_msg = messages[1 + 2 * t]["content"]
        asst_msg = messages[2 + 2 * t]["content"]
        d = decisions[t]
        # Truncate assistant message
        if len(asst_msg) > 150:
            asst_msg = asst_msg[:147] + "..."
        print(f"  t={t:>3d} | d={d} | {user_msg}")
        print(f"         CoT: {asst_msg}")

    if problem.T > 7:
        print(f"  ... ({problem.T - len(turns_to_show)} turns omitted)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT warmup data")
    parser.add_argument("-n", "--num-episodes", type=int, default=300)
    parser.add_argument("-o", "--output", default="data/sft_warmup.jsonl")
    parser.add_argument("--inspect", type=int, default=0,
                        help="Print detailed output for first N episodes")
    args = parser.parse_args()

    generate_sft_data(args.num_episodes, args.output, args.inspect)
