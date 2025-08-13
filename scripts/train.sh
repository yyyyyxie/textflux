export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="cat"
export OUTPUT_DIR="outputs/textflux-beta"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# single line train
# mkdir -p weights/transformer && mv yyyyyxie/textflux/* weights/transformer
accelerate launch --config_file accelerate_config.yaml scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_inpaint_model_name_or_path="weights" \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=2e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --seed="42" \
  --max_sequence_length=512  \
  --checkpointing_steps=5000  \
  --train_base_model \
  --report_to="wandb" \


# multi-line
# accelerate launch --config_file accelerate_config.yaml train_flux_inpaint_finetune_diffuser.py \
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
#   --gradient_accumulation_steps=4 \
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
#   --train_base_model \
#   --multi_dataset \
#   --multi_line