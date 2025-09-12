#\!/bin/bash

echo "================================================"
echo "Testing SigLIP + SmolLM2-135M Configuration"
echo "================================================"

python train.py \
    --vision_encoder siglip \
    --language_model HuggingFaceTB/SmolLM2-135M-Instruct \
    --save_checkpoint_steps 500 \
    --max_training_steps 20 \
    --validation_steps 10 \
    --max_validation_samples 100 \
    --no_log_wandb \
    --console_log_interval 5 \
    --num_workers 4 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 2>&1 | grep -E "(nanoVLM initialized|Step [0-9]+/|Val Loss|Average time)" 

echo ""
echo "================================================"
echo "Testing DINOv3 + SmolLM2-135M Configuration"  
echo "================================================"

python train.py \
    --vision_encoder dinov3 \
    --language_model HuggingFaceTB/SmolLM2-135M-Instruct \
    --freeze_vision_encoder \
    --save_checkpoint_steps 500 \
    --max_training_steps 20 \
    --validation_steps 10 \
    --max_validation_samples 100 \
    --max_img_size 512 \
    --no_log_wandb \
    --console_log_interval 5 \
    --num_workers 4 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 2>&1 | grep -E "(nanoVLM initialized|Step [0-9]+/|Val Loss|Average time)"

echo ""
echo "Both configurations completed successfully\!"
