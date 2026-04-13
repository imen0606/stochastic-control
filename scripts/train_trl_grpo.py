#!/usr/bin/env python3
"""
GRPO training on the regime switching gym using TRL.

Uses environment_factory for true multi-turn sequential decisions.
The model sees one signal at a time and must decide before the next.

Usage on RunPod (single A100):
    python scripts/train_trl_grpo.py

    # Quick test:
    python scripts/train_trl_grpo.py --test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from scripts.trl_env import RegimeSwitchingEnv, reward_func


def main():
    parser = argparse.ArgumentParser()
    # Qwen3 has proper tool-calling support required by environment_factory
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="output/grpo_planning")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--num-generations", type=int, default=8,
                        help="K completions per episode for GRPO")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 64 episodes, 1 epoch")
    args = parser.parse_args()

    if args.test:
        args.num_episodes = 64
        args.epochs = 1
        args.num_generations = 4

    # Dataset: each row is a prompt that starts an episode.
    # The environment handles the multi-turn interaction via tool calling.
    SYSTEM_PROMPT = (
        "You are a trading strategy agent. At each time step, you receive "
        "a signal observation. Use the `decide` tool to submit your reasoning "
        "and decision (s_t = 0 for OFF, s_t = 1 for ON). Consider the signal "
        "trend, remaining steps, and whether paying the switching cost now will "
        "be recovered by future gains."
    )

    dataset = Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": SYSTEM_PROMPT}]
        ] * args.num_episodes
    })

    print(f"Starting GRPO training:")
    print(f"  Model: {args.model}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  K (generations per episode): {args.num_generations}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Output: {args.output_dir}")
    print(f"  Test mode: {args.test}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=args.output_dir,

            # vLLM for fast generation (colocate = single GPU)
            use_vllm=True,
            vllm_mode="colocate",

            # GRPO hyperparameters
            num_generations=args.num_generations,
            max_completion_length=4096,  # multi-turn needs space
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,

            # Logging
            logging_steps=5,
            save_steps=200,
            log_completions=True,
            report_to="none",

            # Disable thinking mode for simpler outputs
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=RegimeSwitchingEnv,
    )

    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"\nTraining complete. Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
