#!/bin/bash
# RunPod setup for GRPO training on the regime switching gym.
#
# TRAINING RUN 2: J-gap zone + SFT warmup
#   - Zone: kappa=[0.01,0.10], ratio=[2.0,5.0], T=[50,100]
#   - Reward: raw J (realized utility)
#   - SFT warmup first (300 episodes, ~1-2 hours)
#   - Then GRPO (5000 episodes, K=4, ~8-12 hours)
#
# REQUIREMENTS:
#   - H100 80GB or A100 80GB
#   - 150GB disk
#   - PyTorch template
#
# Usage:
#   1. Rent GPU pod, SSH in
#   2. Run: bash scripts/runpod_setup.sh
#   3. Follow the prompts
#
# Total cost estimate: $15-25 on H100 ($1.50-3/hr x 8-12 hours)

set -e

echo "========================================="
echo "GRPO Training Setup (Run 2: SFT + GRPO)"
echo "========================================="

# Check disk space
AVAIL=$(df -h /root | awk 'NR==2{print $4}')
echo "Available disk: $AVAIL"
echo "(Need at least 80GB free)"
echo ""

# 1. Install dependencies
# IMPORTANT: transformers@main is required for environment_factory (needs >=5.2.0)
# peft>=0.15.0 is required for compatibility with transformers@main
echo "[1/4] Installing dependencies..."
pip install -q git+https://github.com/huggingface/transformers.git@main
pip install -q "vllm>=0.17.0,<0.18.0" "trl>=0.20.0" "peft>=0.15.0" datasets scipy numpy

# 2. Clone/update repo
echo "[2/4] Setting up repository..."
if [ ! -d "/root/stochastic_control" ]; then
    git clone https://github.com/imen0606/stochastic-control.git /root/stochastic_control
else
    cd /root/stochastic_control && git pull
fi
cd /root/stochastic_control

# 3. SFT warmup
echo ""
echo "========================================="
echo "[3/4] SFT warmup (300 episodes, ~1-2 hours)"
echo "========================================="
echo "Training on synthetic optimal traces to break"
echo "the cold-start problem (model plays always-OFF)."
echo ""

if [ -d "output/sft_warmup/final_model" ]; then
    echo "SFT model already exists at output/sft_warmup/final_model"
    echo "Skipping SFT. Delete the directory to re-run."
else
    python scripts/train_sft_warmup.py
fi

# 4. Verify SFT worked
echo ""
echo "========================================="
echo "[4/4] Verify SFT model switches"
echo "========================================="
echo "Testing if the SFT model now switches (not always OFF)..."
echo ""
python scripts/test_model_multiturn.py --model output/sft_warmup/final_model -n 2

echo ""
echo "========================================="
echo "CHECK: Does the model ever switch?"
echo "  - If yes: proceed to GRPO"
echo "  - If still always OFF: SFT didn't work, need more data or epochs"
echo ""
echo "TO RUN GRPO (~8-12 hours):"
echo "  python scripts/train_trl_grpo.py --model output/sft_warmup/final_model"
echo ""
echo "TO QUICK-TEST GRPO FIRST (32 episodes, ~30 min):"
echo "  python scripts/train_trl_grpo.py --test --model output/sft_warmup/final_model"
echo ""
echo "TO EVALUATE AFTER TRAINING:"
echo "  python scripts/eval_trained.py --model output/grpo_planning/final_model -N 50 --tag trained"
echo "  python scripts/eval_trained.py --model Qwen/Qwen3-8B -N 50 --tag baseline"
echo "========================================="
