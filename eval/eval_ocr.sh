# python eval/eval_dgocr.py pred_imgs gt.json


python eval/eval_dgocr.py \
    --img_dir results/ReCTS_output/cropped_images \
    --input_json textflux_benchmark_all_mask_1024_single_line/ReCTS_ori.json


python eval/eval_dgocr.py \
    --img_dir results_other/ReCTS_output/cropped_images \
    --input_json textflux_benchmark_all_mask_1024_single_line/ReCTS_other.json


python eval/eval_dgocr.py \
    --img_dir results_lora_Japan/ReCTS_output/cropped_images \
    --input_json textflux_benchmark_all_mask_1024_single_line/ReCTS_Japan.json


python eval/eval_dgocr.py \
    --img_dir results_lora/ReCTS_output/cropped_images \
    --input_json textflux_benchmark_all_mask_1024_single_line/ReCTS_other.json



