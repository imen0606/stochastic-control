#!/usr/bin/env python3
"""
SFT warmup for GRPO cold-start.

Fine-tunes a base model on synthetic optimal traces so it learns
the concept of switching. Light touch — 2 epochs, LoRA only.
The goal is NOT to make the model optimal, but to give it enough
planning capability that GRPO's K completions diverge.

Usage:
    # On RunPod (after generating data locally and pushing to repo):
    python scripts/train_sft_warmup.py

    # Quick test:
    python scripts/train_sft_warmup.py --test

    # Then GRPO:
    python scripts/train_trl_grpo.py --model output/sft_warmup/final_model
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="SFT warmup training")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data", default="data/sft_warmup.jsonl")
    parser.add_argument("--output-dir", default="output/sft_warmup")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 10 episodes, 1 epoch")
    args = parser.parse_args()

    # Lazy imports (heavy deps)
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Load data
    print(f"Loading SFT data from {args.data}...")
    with open(args.data) as f:
        data = [json.loads(line) for line in f]

    if args.test:
        data = data[:10]
        args.epochs = 1

    print(f"  Episodes: {len(data)}")
    print(f"  Avg turns: {sum(len(d['messages']) for d in data) / len(data) / 2:.0f}")

    dataset = Dataset.from_list(data)

    # LoRA config — same as GRPO for compatibility
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    print(f"\nStarting SFT warmup:")
    print(f"  Model: {args.model}")
    print(f"  Episodes: {len(data)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  LoRA: r=16, alpha=32")
    print(f"  Output: {args.output_dir}")

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            max_seq_length=16384,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            report_to="none",
            bf16=True,

            # Disable thinking mode for cleaner outputs
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )

    trainer.train()

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)
    print(f"\nSFT warmup complete. Model saved to {final_path}")
    print(f"\nNext step: GRPO training")
    print(f"  python scripts/train_trl_grpo.py --model {final_path}")


if __name__ == "__main__":
    main()
