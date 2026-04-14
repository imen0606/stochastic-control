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
        args.max_completion_length = 4096
    else:
        args.max_completion_length = 6144

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

    # Check if --model is a LoRA adapter checkpoint (from SFT warmup)
    sft_adapter_path = None
    if os.path.isdir(args.model) and os.path.exists(os.path.join(args.model, "adapter_config.json")):
        from peft import PeftConfig
        peft_cfg = PeftConfig.from_pretrained(args.model)
        sft_adapter_path = args.model
        args.model = peft_cfg.base_model_name_or_path
        print(f"Detected LoRA adapter at {sft_adapter_path}")
        print(f"  Base model: {args.model}")

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
    print(f"  SFT adapter: {sft_adapter_path or 'None'}")
    print(f"  LoRA: {args.use_lora}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  K (generations per episode): {args.num_generations}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Max completion length: {args.max_completion_length}")
    print(f"  Output: {args.output_dir}")
    print(f"  Test mode: {args.test}")

    # If SFT adapter exists, merge it and save as a full model on disk
    # (avoids keeping two copies in memory during GRPO init)
    model_ref = args.model
    if sft_adapter_path:
        import torch
        merged_path = os.path.join(os.path.dirname(sft_adapter_path), "merged_for_grpo")
        if os.path.exists(os.path.join(merged_path, "config.json")):
            print(f"Using previously merged model at {merged_path}")
            model_ref = merged_path
        else:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            print(f"Loading base model + SFT adapter...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
            )
            merged = PeftModel.from_pretrained(base_model, sft_adapter_path)
            merged = merged.merge_and_unload()
            print(f"  Merged SFT adapter. Saving to {merged_path}...")
            merged.save_pretrained(merged_path)
            # Also copy tokenizer from base model cache
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
            tok.save_pretrained(merged_path)
            # Free memory before GRPO loads its own copies
            del merged, base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()
            print(f"  Saved merged model and freed memory")
            model_ref = merged_path

    trainer_kwargs = dict(
        model=model_ref,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=args.output_dir,

            # vLLM for fast generation (colocate = single GPU)
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.4,  # limit vLLM to 40% GPU (rest for training)

            # GRPO hyperparameters
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,  # reduced from 4 for lower peak memory
            gradient_checkpointing=True,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,

            # Logging
            logging_steps=5,
            save_steps=500,
            save_total_limit=2,
            log_completions=False,  # save memory (completions can be very long)
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
