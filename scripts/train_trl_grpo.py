#!/usr/bin/env python3
"""
GRPO training on the regime switching gym using TRL.

Uses environment_factory for true multi-turn sequential decisions.
The model sees one signal at a time and must decide before the next.

Training Run 2 changes (J-gap zone):
  - Zone: kappa=[0.01,0.10], ratio=[2.0,5.0], T=[50,100]
    (greedy captures only 30-60% of J_optimal, vs 95% in old zone)
  - Reward: raw J (realized utility), no normalization
    (GRPO advantage within K group handles per-episode scale)
  - K=4 (was 8) and max_completion_length=16384 (was 4096)
    to accommodate longer episodes (50-100 steps)

Designed for 7B models on single A100 80GB with LoRA to fit in memory.

Usage on RunPod:
    # Step 1: Test model can follow multi-turn (ALWAYS DO THIS FIRST)
    python scripts/test_model_multiturn.py --model Qwen/Qwen3-8B

    # Step 2: Quick training test (32 episodes, ~30 min)
    python scripts/train_trl_grpo.py --test

    # Step 3: Full training (5000 episodes, ~8-12 hours)
    python scripts/train_trl_grpo.py
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
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", default="output/grpo_planning")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="K completions per episode for GRPO (reduced from 8 for longer episodes)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 64 episodes, 1 epoch, K=4")
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="Use LoRA for memory efficiency (default: True)")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    args = parser.parse_args()

    if args.test:
        args.num_episodes = 32
        args.epochs = 1
        args.num_generations = 2

    # System prompt for the agent
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

    # LoRA config for memory efficiency on 7B models
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    print(f"Starting GRPO training:")
    print(f"  Model: {args.model}")
    print(f"  LoRA: {args.use_lora}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  K (generations per episode): {args.num_generations}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Output: {args.output_dir}")
    print(f"  Test mode: {args.test}")

    trainer_kwargs = dict(
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
            max_completion_length=16384,  # J-gap zone: T=50-100 steps, ~100 tokens/step
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,

            # Logging
            logging_steps=5,
            save_steps=500,  # save less frequently to save disk
            save_total_limit=2,  # keep only last 2 checkpoints
            log_completions=True,
            report_to="none",

            # Disable thinking mode for cleaner outputs
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=RegimeSwitchingEnv,
    )

    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)
    print(f"\nTraining complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
