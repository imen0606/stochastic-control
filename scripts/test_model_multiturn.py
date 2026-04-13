#!/usr/bin/env python3
"""
Quick test: can a model follow our multi-turn tool-calling format?

Run this BEFORE committing to training. Tests 3 episodes and prints
full CoT so you can verify the model:
1. Understands which time step it's on
2. Makes a decision (s_t = 0 or 1) each turn
3. Doesn't re-read old turns or get confused

Usage:
    python scripts/test_model_multiturn.py --model Qwen/Qwen3-7B
    python scripts/test_model_multiturn.py --model Qwen/Qwen2.5-7B-Instruct

Cost: ~$0.20 (10 minutes on A100)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.problems.regime_switching.verifier import _parse_decision


def test_with_transformers(model_name, n_episodes=3):
    """Test multi-turn with HuggingFace transformers (slow but simple)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    gen = RegimeSwitchingGenerator(GeneratorConfig.planning_zone())

    # Use seeds known to have short episodes
    test_seeds = [4, 10, 22]  # T=9, T=7, T=7 approximately

    for seed in test_seeds[:n_episodes]:
        p = gen.sample(seed=seed)
        print(f"\n{'='*60}")
        print(f"EPISODE seed={seed}, T={p.T}, kappa={p.kappa:.3f}")
        print(f"Signal: {' '.join(f'{z:+.3f}' for z in p.z_path[:p.T])}")
        print(f"{'='*60}")

        system_msg = setup_prompt(p)
        messages = [{"role": "system", "content": system_msg}]
        prev = p.initial_regime
        decisions = []

        for t in range(p.T):
            user_msg = step_prompt(t, p.z_path[t], prev)
            messages.append({"role": "user", "content": user_msg})

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,  # short — we just need the decision
                    do_sample=False,
                )
            elapsed = time.time() - t0

            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            d = _parse_decision(text)
            decisions.append(d)
            messages.append({"role": "assistant", "content": text})
            prev = d

            # Print condensed output
            status = "ON" if d == 1 else "OFF"
            print(f"\n  t={t} | Z={p.z_path[t]:+.4f} | Decision: {status} | ({elapsed:.1f}s)")
            # Print first 200 chars of reasoning
            clean_text = text.replace("<think>", "").replace("</think>", "").strip()
            if len(clean_text) > 200:
                print(f"    CoT: {clean_text[:200]}...")
            else:
                print(f"    CoT: {clean_text}")

        print(f"\n  DECISIONS: {decisions}")
        print(f"  All OFF: {all(d == 0 for d in decisions)}")
        print(f"  All ON:  {all(d == 1 for d in decisions)}")

        # Check for red flags
        all_same = all(d == decisions[0] for d in decisions)
        if all_same:
            print(f"  ⚠️  WARNING: Model chose {decisions[0]} every step — may not be reasoning")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print("Check above for:")
    print("  1. Does the model reference the CURRENT time step (not t=0)?")
    print("  2. Does it compute PnL correctly?")
    print("  3. Does it ever switch (not all same decision)?")
    print("  4. Is the CoT coherent per step?")
    print(f"{'='*60}")


def test_with_trl_env(model_name):
    """Test using the actual TRL environment (tool-calling format)."""
    print("\n--- Testing with TRL environment (tool-calling) ---")
    print("This tests the exact format GRPO training will use.")

    # Import and run the env test
    from scripts.trl_env import RegimeSwitchingEnv

    env = RegimeSwitchingEnv()
    obs = env.reset()
    print(f"Reset observation (first 150 chars): {obs[:150]}...")
    print(f"Problem: T={env.problem.T}, kappa={env.problem.kappa:.3f}")

    # Simulate a model that alternates ON/OFF
    for t in range(min(env.problem.T, 5)):
        decision = f"At t={t}, the signal is noted. s_t = {t % 2}"
        try:
            result = env.decide(decision)
            print(f"  t={t}: decided {t%2}, got: {result[:80]}")
        except ValueError as e:
            print(f"  t={t}: Episode ended: {e}")
            break

    print(f"  Reward: {env.reward:.3f}")
    print("  TRL env works correctly.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-7B",
                        help="Model to test")
    parser.add_argument("-n", type=int, default=3,
                        help="Number of test episodes")
    parser.add_argument("--skip-env-test", action="store_true",
                        help="Skip TRL environment test")
    args = parser.parse_args()

    if not args.skip_env_test:
        test_with_trl_env(args.model)

    test_with_transformers(args.model, n_episodes=args.n)
