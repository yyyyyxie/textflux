export CUDA_VISIBLE_DEVICES=0,1,2,3

python scripts/run_eval_lora.py \
    --json_path textflux_benchmark_all_mask_1024_single_line/ReCTS_ori.json \
    --original_images_dir textflux_benchmark_all_mask_1024_single_line/textflux_data/processed_ReCTS_test_images/original \
    --lora_weights_path weights/textflux-beta-lora/pytorch_lora_weights.safetensors \
    --output_dir ./results_lora/ReCTS_output \
    --font_path ./resource/font/Arial-Unicode-Regular.ttf \
    --num_gpus 4 \
    --text_height_ratio 0.15625 \
    --scheduler "overshoot"  # optional
