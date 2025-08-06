#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import multiprocessing
import torch
import numpy as np
import json
from diffusers import FluxFillPipeline, FluxTransformer2DModel
# from diffusers import FluxFillPipeline, FluxTransformer2DModel, StochasticRFOvershotDiscreteScheduler
from diffusers.utils import check_min_version, load_image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2


def draw_glyph_flexible(font, text, width, height, max_font_size=140):

    img = Image.new(mode='RGB', size=(width, height), color='black')
    if not text or not text.strip():
        return img
    draw = ImageDraw.Draw(img)
    g_size = 50
    new_font = font.font_variant(size=g_size)
    left, top, right, bottom = new_font.getbbox(text)
    text_width_initial = max(right - left, 1)
    text_height_initial = max(bottom - top, 1)
    width_ratio = width * 0.9 / text_width_initial
    height_ratio = height * 0.9 / text_height_initial
    ratio = min(width_ratio, height_ratio)
    final_font_size = int(g_size * ratio)

    if width > 1280:
        max_font_size = 200
        
    final_font_size = min(final_font_size, max_font_size)
    new_font = font.font_variant(size=max(final_font_size, 10))
    draw.text((width / 2, height / 2), text, font=new_font, fill='white', anchor='mm')
    return img


def load_data_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        data_list = all_data.get('data_list', [])
        if not data_list:
            print(f"Warning: The 'data_list' in JSON file '{json_path}' is empty.")
        return data_list
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_path}'.")
        return []
    except json.JSONDecodeError:
        print(f"Error: JSON file '{json_path}' has an invalid format.")
        return []

def generate_prompt(words):
    words_str = ', '.join(f"'{word}'" for word in words)
    prompt_template = (
        "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
        "[IMAGE1] is a template image rendering the text, with the words {words}; "
        "[IMAGE2] shows the text content {words} naturally and correspondingly integrated into the image."
    )
    return prompt_template.format(words=words_str)

prompt_template2 = (
    "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
    "[IMAGE1] is a template image rendering the text, with the words; "
    "[IMAGE2] shows the text content naturally and correspondingly integrated into the image."
)


def run_inference(
    item_data, 
    original_images_dir,
    font, 
    pipe,
    args
):

    img_name = item_data['img_name']
    annotation = item_data['annotations'][0]
    text = annotation['text']
    polygon = np.array(annotation['polygon'], dtype=np.int32)
    original_img_path = os.path.join(original_images_dir, img_name)
    original_img_pil = load_image(original_img_path).convert("RGB")
    w, h = original_img_pil.size

    text_render_height = int(w * args.text_height_ratio)
    
    text_render_pil = draw_glyph_flexible(font, text, width=w, height=text_render_height)
    
    original_mask_np = np.zeros((h, w, 3), dtype=np.uint8)
    
    cv2.fillPoly(original_mask_np, [polygon], (255, 255, 255))
    original_mask_pil = Image.fromarray(original_mask_np)

    text_mask_pil = Image.new("RGB", text_render_pil.size, "black")
    
    inpaint_image = Image.fromarray(np.vstack((np.array(text_render_pil), np.array(original_img_pil))))
    extended_mask = Image.fromarray(np.vstack((np.array(text_mask_pil), np.array(original_mask_pil))))


    new_width = (w // 32) * 32
    new_height = ((h + text_render_height) // 32) * 32
    
    inpaint_image = inpaint_image.resize((new_width, new_height))
    extended_mask = extended_mask.resize((new_width, new_height))

    prompt = generate_prompt([text])
    print(f"Processing {img_name} (WxH: {w}x{h}), text line height: {text_render_height}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    if args.scheduler == 'overshoot':
        from diffusers import StochasticRFOvershotDiscreteScheduler
        scheduler_config = pipe.scheduler.config
        scheduler = StochasticRFOvershotDiscreteScheduler.from_config(scheduler_config)
        overshot_func = lambda t, dt: t+dt
        
        pipe.scheduler = scheduler
        pipe.scheduler.set_c(2.0)
        pipe.scheduler.set_overshot_func(overshot_func)

    result = pipe(
        height=new_height,
        width=new_width,
        image=inpaint_image,
        mask_image=extended_mask,
        num_inference_steps=args.steps,
        generator=generator,
        max_sequence_length=512,
        guidance_scale=args.guidance_scale,
        prompt=prompt_template2,
        prompt_2=prompt,
    ).images[0]
    
    return result, new_width, new_height, h, text_render_height



def worker(gpu_id, task_queue, args):
    torch.cuda.set_device(gpu_id)
    print(f"Worker {gpu_id} is loading the model onto GPU {gpu_id}...")
    
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev/transformer",
        torch_dtype=torch.bfloat16
    )

    state_dict, network_alphas = FluxFillPipeline.lora_state_dict(
        pretrained_model_name_or_path_or_dict=args.lora_weights_path,     
        return_alphas=True
    )
        
    is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
    if not is_correct_format:
        raise ValueError("Invalid LoRA checkpoint.")


    FluxFillPipeline.load_lora_into_transformer(
        state_dict=state_dict,
        network_alphas=network_alphas,
        transformer=transformer,
    )

    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)

    try:
        font = ImageFont.truetype(args.font_path, size=60)
    except IOError:
        font = ImageFont.load_default()
        print(f"Worker {gpu_id}: Font '{args.font_path}' not found, using default font.")
    print(f"Worker {gpu_id}: Model loaded, starting task processing...")
    
    while True:
        task = task_queue.get()
        if task is None:
            print(f"Worker {gpu_id} received shutdown signal, exiting.")
            break
        
        item_data = task
        
        try:
            full_result_image, res_w, res_h, orig_h, text_render_height = run_inference(
                item_data=item_data,
                original_images_dir=args.original_images_dir,
                font=font,
                pipe=pipe,
                args=args 
            )
            
            base_filename = os.path.basename(item_data['img_name'])
            
            full_save_path = os.path.join(args.output_dir, 'full_images', base_filename)
            full_result_image.save(full_save_path)

            # Calculate the cropped region 
            crop_top_edge = int(res_h * (text_render_height / (orig_h + text_render_height)))
            cropped_image = full_result_image.crop((0, crop_top_edge, res_w, res_h))
            
            final_save_path = os.path.join(args.output_dir, 'cropped_images', base_filename)
            cropped_image.save(final_save_path)
            print(f"Worker {gpu_id}: ✔️ {item_data['img_name']} -> Saved to 'full_images' and 'cropped_images'")

        except Exception as e:
            print(f"Worker {gpu_id}: Error processing {item_data.get('img_name')}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Multi-process Real-time Stitching and FLUX Inference')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the annos.json file containing annotation information')
    parser.add_argument('--original_images_dir', type=str, required=True, help='Path to the folder containing original images (e.g., .../original)')
    parser.add_argument('--output_dir', type=str,  default="visualization_results", help='Main output folder for results')
    parser.add_argument('--lora_weights_path', type=str, required=True, help='Path to lora weights')
    parser.add_argument('--font_path', type=str, default="./resource/font/Arial-Unicode-Regular.ttf", help='Path to the font file (.ttf or .ttc)')
    parser.add_argument('--text_height_ratio', type=float, default=0.1667, help='Ratio of top text line height to image width (default: 1/6)')
    parser.add_argument('--steps', type=int, default=30, help='Inference steps')
    parser.add_argument('--guidance_scale', type=float, default=30, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--scheduler', type=str, default="overshoot", help='Sampler, None or "overshoot"')
    
    args = parser.parse_args()

    check_min_version("0.30.1")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'full_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'cropped_images'), exist_ok=True)
    
    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    
    data_list = load_data_from_json(args.json_path)

    if not data_list:
        print("Data list is empty, exiting program.")
        return

    print(f"Found {len(data_list)} tasks, distributing them to the queue...")
    for item_data in tqdm(data_list, desc="Distributing tasks"):
        if not item_data.get('annotations') or not item_data['annotations'][0].get('text') or not item_data['annotations'][0].get('polygon'):
            print(f"Skipping {item_data.get('img_name')}: Incomplete annotation information.")
            continue 

        task_queue.put(item_data)
    
    for _ in range(args.num_gpus):
        task_queue.put(None)
    
    processes = []
    for gpu_id in range(args.num_gpus):
        p = multiprocessing.Process(target=worker, args=(gpu_id, task_queue, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("All tasks processed.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    multiprocessing.set_start_method("spawn", force=True)
    main()