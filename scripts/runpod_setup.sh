#!/bin/bash
# RunPod setup for GRPO training on the regime switching gym.
#
# TRAINING RUN 2: J-gap zone
#   - Zone: kappa=[0.01,0.10], ratio=[2.0,5.0], T=[50,100]
#   - Reward: raw J (realized utility)
#   - K=4, max_completion_length=16384
#
# REQUIREMENTS:
#   - A100 80GB GPU (or 2xA100 40GB)
#   - 150GB disk (NOT 50GB)
#   - PyTorch template
#
# Usage:
#   1. Rent RunPod: A100 80GB, 150GB disk, PyTorch template
#   2. SSH or web terminal in
#   3. Run: bash scripts/runpod_setup.sh
#   4. Follow the prompts
#
# Total cost estimate: $20-35 (8-12 hours on A100, episodes are longer now)

set -e

echo "========================================="
echo "GRPO Training Setup (Run 2: J-gap zone)"
echo "========================================="

# Check disk space
AVAIL=$(df -h /root | awk 'NR==2{print $4}')
echo "Available disk: $AVAIL"
echo "(Need at least 80GB free — if less, increase pod disk to 150GB)"
echo ""

# 1. Install dependencies with pinned versions
echo "[1/5] Installing dependencies..."
pip install -q "vllm>=0.17.0,<0.18.0" trl transformers datasets scipy numpy peft
pip install -q git+https://github.com/huggingface/transformers.git@main

# 2. Clone repo
echo "[2/5] Setting up repository..."
if [ ! -d "/root/stochastic_control" ]; then
    git clone https://github.com/imen0606/stochastic-control.git /root/stochastic_control
else
    cd /root/stochastic_control && git pull
fi
cd /root/stochastic_control

# 3. Test multi-turn FIRST
echo ""
echo "========================================="
echo "[3/5] CRITICAL: Testing multi-turn format"
echo "========================================="
echo "Running 2 test episodes to verify the model"
echo "can follow multi-turn tool-calling..."
echo "(NOTE: Episodes are now 50-100 steps, much longer than before)"
echo ""
python scripts/test_model_multiturn.py --model Qwen/Qwen3-7B -n 2

echo ""
echo "========================================="
echo "CHECK THE OUTPUT ABOVE:"
echo "  1. Does the model reference the current time step?"
echo "  2. Does it compute PnL (not repeat t=0)?"
echo "  3. Does it ever switch (not all same)?"
echo ""
read -p "Does the model follow multi-turn correctly? (y/n) " ANSWER
if [ "$ANSWER" != "y" ]; then
    echo "Model failed multi-turn test. Try a different model:"
    echo "  python scripts/test_model_multiturn.py --model Qwen/Qwen2.5-7B-Instruct"
    echo "  python scripts/test_model_multiturn.py --model Qwen/Qwen3-4B"
    exit 1
fi

# 4. Quick training test
echo ""
echo "========================================="
echo "[4/5] Quick training test (32 episodes, K=2, ~30-45 min)"
echo "========================================="
echo "WHAT TO WATCH FOR:"
echo "  - Rewards should NOT all be 0 (some episodes have signal)"
echo "  - Rewards should NOT all be identical (need variance across K)"
echo "  - If all rewards identical: cold start problem -> need SFT warmup"
echo ""
python scripts/train_trl_grpo.py --test

echo ""
echo "========================================="
echo "Quick test complete! Check output above."
echo ""
echo "KEY CHECKS before full training:"
echo "  1. Are rewards varied across completions? (need variance for GRPO)"
echo "  2. Are any rewards > 0? (model earning positive J)"
echo "  3. Is 'never switch' NOT dominating? (check completion logs)"
echo ""
echo "[5/5] To run full training (~8-12 hours):"
echo "  python scripts/train_trl_grpo.py"
echo ""
echo "To evaluate after training:"
echo "  python scripts/eval_trained.py --model output/grpo_planning/final_model -N 50 --tag trained"
echo "  python scripts/eval_trained.py --model Qwen/Qwen3-7B -N 50 --tag baseline"
echo ""
echo "IF COLD START (all K completions identical):"
echo "  Need SFT warmup first. See scripts/sft_warmup.py (TODO)"
echo "========================================="
