#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/HRNet-Semantic-Segmentation/lib"
mkdir -p /root/tf-logs/logs

# =============================================
# Stage 1: Train on Deepglobe dataset
# =============================================
echo "============================================"
echo "Stage 1: Training on Deepglobe Road Dataset"
echo "============================================"

python tools/train_road.py \
    --cfg experiments/deepglobe_road/seg_hrnet_w48_focal_dice_512x512_sgd_lr1e-2_wd5e-4_bs_4_epoch100.yaml \
    --focal_weight 1.0 \
    --dice_weight 1.0 \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    --seed 42

DEEPGLOBE_BEST="output/deepglobe_road/seg_hrnet_w48_focal_dice_512x512_sgd_lr1e-2_wd5e-4_bs_4_epoch100/best_model.pth"

if [ ! -f "$DEEPGLOBE_BEST" ]; then
    echo "ERROR: Deepglobe training failed - best model not found"
    exit 1
fi

echo ""
echo "Deepglobe training complete. Best model at: $DEEPGLOBE_BEST"
echo ""

# =============================================
# Stage 2: Fine-tune on Dataset 2
# =============================================
echo "============================================"
echo "Stage 2: Fine-tuning on Road Dataset 2"
echo "============================================"

python tools/train_road.py \
    --cfg experiments/road_dataset2/seg_hrnet_w48_focal_dice_512x512_sgd_lr1e-2_wd5e-4_bs_4_epoch100.yaml \
    --focal_weight 1.0 \
    --dice_weight 1.0 \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    --pretrained "$DEEPGLOBE_BEST" \
    --seed 42

echo ""
echo "All training complete!"
echo "Stage 1 best model: $DEEPGLOBE_BEST"
echo "Stage 2 best model: output/road_dataset2/seg_hrnet_w48_focal_dice_512x512_sgd_lr1e-2_wd5e-4_bs_4_epoch100/best_model.pth"
