#!/bin/bash
# RunPod setup script for GRPO training on the regime switching gym.
#
# Usage:
#   1. Rent a RunPod A100 (80GB) instance with PyTorch template
#   2. SSH in
#   3. Run: bash runpod_setup.sh
#
# This will:
#   - Install dependencies
#   - Clone the repo
#   - Run a quick test (100 episodes, ~10 min)
#   - If test passes, run full training (5000 episodes, ~4-8 hours)

set -e

echo "========================================="
echo "GRPO Training Setup for Planning Gym"
echo "========================================="

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
pip install -q trl transformers datasets vllm scipy numpy wandb

# 2. Clone repo
echo "[2/4] Cloning repository..."
if [ ! -d "stochastic_control" ]; then
    git clone https://github.com/YOUR_REPO/stochastic_control.git
fi
cd stochastic_control
pip install -q -e .

# 3. Quick test
echo "[3/4] Running quick test (100 episodes, ~10 min)..."
python scripts/train_trl_grpo.py --test

echo ""
echo "========================================="
echo "Quick test complete!"
echo "Check output/grpo_planning for results."
echo ""
echo "To run full training:"
echo "  python scripts/train_trl_grpo.py"
echo ""
echo "To evaluate after training:"
echo "  python scripts/eval_trained.py --model output/grpo_planning/final_model"
echo "  python scripts/eval_trained.py --model Qwen/Qwen2.5-3B-Instruct --tag baseline"
echo "========================================="
