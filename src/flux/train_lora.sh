export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="cat"
export OUTPUT_DIR="outputs/textflux-beta-lora"
# export CUDA_VISIBLE_DEVICES=0

# single line
accelerate launch --main_process_port 29503 --config_file accelerate_config_lora.yaml scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_inpaint_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \
  --pretrained_lora_path="weights/pytorch_lora_weights.safetensors" \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="prodigy" \
  --learning_rate=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --seed="42" \
  --max_sequence_length=512  \
  --checkpointing_steps=5000  \
  --train_base_model \
  --train_lora \
  --lora_rank 128\
  --report_to="wandb" \


# multi-line
# accelerate launch --main_process_port 29503 --config_file accelerate_config_lora.yaml train_flux_inpaint_lora_diffuser.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --pretrained_inpaint_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --data_path=datasets/multi-lingual \
#   --resolution 512 640 768 896 1024\
#   --caption_type="txt" \
#   --mixed_precision="bf16" \
#   --train_batch_size=1 \
#   --guidance_scale=1 \
#   --gradient_accumulation_steps=8 \
#   --optimizer="adamw" \
#   --use_8bit_adam \
#   --learning_rate=2e-5 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=30000 \
#   --seed="42" \
#   --max_sequence_length=512  \
#   --checkpointing_steps=10000  \
#   --report_to="wandb" \
#   --train_lora \
#   --lora_rank 128\
#   --multi_dataset \
#   --multi_line

