import argparse
import os

import cv2
import numpy as np
import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import check_min_version, load_image
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def read_words_from_text(input_text):
    if isinstance(input_text, str) and os.path.exists(input_text):
        with open(input_text, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        words = [line.strip() for line in input_text.splitlines() if line.strip()]
    return words

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


PIPE = None
def load_flux_pipeline():
    global PIPE
    if PIPE is None:
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev/transformer",
            torch_dtype=torch.bfloat16
        )

        state_dict, network_alphas = FluxFillPipeline.lora_state_dict(
            pretrained_model_name_or_path_or_dict="yyyyyxie/textflux-lora",     ## The tryon Lora weights
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

        PIPE = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        PIPE.transformer.to(torch.bfloat16)
    return PIPE

def run_inference(image_input, mask_input, words_input, num_steps=50, guidance_scale=30, seed=42):
    # Load images
    inpaint_image = load_image(image_input).convert("RGB") if isinstance(image_input, str) else image_input.convert("RGB")
    extended_mask = load_image(mask_input).convert("RGB") if isinstance(mask_input, str) else mask_input.convert("RGB")
    
    # Adjust dimensions
    width, height = inpaint_image.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    inpaint_image = inpaint_image.resize((new_width, new_height))
    extended_mask = extended_mask.resize((new_width, new_height))
    
    # Process text
    words = read_words_from_text(words_input)
    prompt = generate_prompt(words)
    print("Generated prompt:", prompt)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(inpaint_image)
    mask_tensor = mask_transform(extended_mask)
    
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    pipe = load_flux_pipeline()
    result = pipe(
        height=new_height,
        width=new_width,
        image=inpaint_image,
        mask_image=extended_mask,
        num_inference_steps=num_steps,
        generator=generator,
        max_sequence_length=512,
        guidance_scale=guidance_scale,
        prompt=prompt_template2,
        prompt_2=prompt,
    ).images[0]

    return result

# Keep functions related to custom mode
# =============================================================================
# 4. Custom mode: Preprocess Mask, Render text in regions, Concatenate composite image
# =============================================================================
def extract_mask(original, drawn, threshold=30):
    """
    Extract a binary mask from the original image and the user-drawn image:
        - If drawn is a dict and contains a "mask" key, directly binarize this mask;
        - Otherwise, use inversion and difference methods to extract the mask.
    """
    if isinstance(drawn, dict):
        if "mask" in drawn and drawn["mask"] is not None:
            drawn_mask = np.array(drawn["mask"]).astype(np.uint8)
            if drawn_mask.ndim == 3:
                drawn_mask = cv2.cvtColor(drawn_mask, cv2.COLOR_RGB2GRAY)
            _, binary_mask = cv2.threshold(drawn_mask, 50, 255, cv2.THRESH_BINARY)
            return Image.fromarray(binary_mask).convert("RGB")
        else:
            drawn_img = np.array(drawn["image"]).astype(np.uint8)
            drawn = 255 - drawn_img
    orig_arr = np.array(original).astype(np.int16)
    drawn_arr = np.array(drawn).astype(np.int16)
    diff = np.abs(drawn_arr - orig_arr)
    diff_gray = np.mean(diff, axis=-1)
    binary_mask = (diff_gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary_mask).convert("RGB")

def insert_spaces(text, num_spaces):
    """
    Insert a specified number of spaces between each character to adjust spacing during text rendering.
    """
    if len(text) <= 1:
        return text
    return (' ' * num_spaces).join(list(text))

def draw_glyph2(
    font,
    text,
    polygon,
    vertAng=10,
    scale=1,
    width=512,
    height=512,
    add_space=True,
    scale_factor=2,
    rotate_resample=Image.BICUBIC,
    downsample_resample=Image.Resampling.LANCZOS
):
    """
    Render tilted/curved text within a specified region (defined by polygon):
        - First upscale (supersample), then rotate, then downsample to ensure high quality;
        - Dynamically adjust font size and whether to insert spaces between characters based on the region's shape.
    Return the final downsampled RGBA numpy array to the target dimensions (height, width).
    """
    big_w = width * scale_factor
    big_h = height * scale_factor

    # Upscale polygon coordinates
    big_polygon = polygon * scale_factor * scale
    rect = cv2.minAreaRect(big_polygon.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if (abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng):
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    # Create large image and temporary white background image
    big_img = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
    tmp = Image.new("RGB", big_img.size, "white")
    tmp_draw = ImageDraw.Draw(tmp)

    _, _, _tw, _th = tmp_draw.textbbox((0, 0), text, font=font)
    if _th == 0:
        text_w = 0
    else:
        w_f, h_f = float(w), float(h)
        text_w = min(w_f, h_f) * (_tw / _th)

    if text_w <= max(w, h):
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_sp = insert_spaces(text, i)
                _, _, tw2, th2 = tmp_draw.textbbox((0, 0), text_sp, font=font)
                if th2 != 0:
                    if min(w, h) * (tw2 / th2) > max(w, h):
                        break
            text = insert_spaces(text, i-1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.75 if vert else 0.85
        if text_w != 0:
            font_size = min(w, h) / (text_w / max(w, h)) * shrink
        else:
            font_size = min(w, h) * 0.80

    new_font = font.font_variant(size=int(font_size))
    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    # Create transparent text rendering layer
    layer = Image.new("RGBA", big_img.size, (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(layer)
    cx, cy = rect[0]
    if not vert:
        draw_layer.text(
            (cx - text_width // 2, cy - text_height // 2 - top),
            text,
            font=new_font,
            fill=(255, 255, 255, 255)
        )
    else:
        _w_ = max(box[:, 0]) - min(box[:, 0])
        x_s = min(box[:, 0]) + _w_ // 2 - text_height // 2
        y_s = min(box[:, 1])
        for c in text:
            draw_layer.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(
        angle,
        expand=True,
        center=(cx, cy),
        resample=rotate_resample
    )

    xo = int((big_img.width - rotated_layer.width) // 2)
    yo = int((big_img.height - rotated_layer.height) // 2)
    big_img.paste(rotated_layer, (xo, yo), rotated_layer)

    final_img = big_img.resize((width, height), downsample_resample)
    final_np = np.array(final_img)
    return final_np

def render_glyph_multi(original, computed_mask, texts):
    """
    For each independent region in computed_mask:
        - Extract region positions using contours and sort them from top to bottom, left to right;
        - Call draw_glyph2 to render corresponding text in each region (supports tilt/curve);
        - Overlay the rendering results of each region onto a transparent black background image, and output the final rendered image.
    """
    mask_np = np.array(computed_mask.convert("L"))
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 50:
            continue
        regions.append((x, y, w, h, cnt))
    regions = sorted(regions, key=lambda r: (r[1], r[0]))
    
    render_img = Image.new("RGBA", original.size, (0, 0, 0, 0))
    try:
        base_font = ImageFont.truetype("resource/font/Arial-Unicode-Regular.ttf", 40)
    except:
        base_font = ImageFont.load_default()
    
    for i, region in enumerate(regions):
        if i >= len(texts):
            break
        text = texts[i].strip()
        if not text:
            continue
        cnt = region[4]
        polygon = cnt.reshape(-1, 2)
        rendered_np = draw_glyph2(
            font=base_font,
            text=text,
            polygon=polygon,
            vertAng=10,
            scale=1,
            width=original.size[0],
            height=original.size[1],
            add_space=True,
            scale_factor=1,
            rotate_resample=Image.BICUBIC,
            downsample_resample=Image.Resampling.LANCZOS
        )
        rendered_img = Image.fromarray(rendered_np, mode="RGBA")
        render_img = Image.alpha_composite(render_img, rendered_img)
    return render_img.convert("RGB")

def choose_concat_direction(height, width):
    """
    Choose concatenation direction based on the original image's aspect ratio:
        - If height is greater than width, use horizontal concatenation;
        - Otherwise, use vertical concatenation.
    """
    return 'horizontal' if height > width else 'vertical'

def get_next_seq_number():
    counter = 1
    while True:
        seq_str = f"{counter:04d}"
        result_path = os.path.join("outputs_my", f"result_{seq_str}.png")
        if not os.path.exists(result_path):
            return seq_str
        counter += 1

def process_normal_mode(image_path, mask_path, words_path, steps, guidance_scale, seed):
    """Normal mode processing function"""
    # Load original image and mask
    original_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("RGB")
    
    # Read text and render font
    texts = read_words_from_text(words_path)
    rendered_text = render_glyph_multi(original_image, mask_image, texts)
    
    # Determine concatenation direction
    width, height = original_image.size
    direction = 'horizontal' if height > width else 'vertical'
    
    # Concatenate image and mask based on direction
    if direction == 'horizontal':
        # Horizontal concatenation [Rendered Image | Original Image]
        combined_image = Image.fromarray(np.hstack((np.array(rendered_text), np.array(original_image))))
        combined_mask = Image.fromarray(np.hstack((np.array(Image.new("RGB", original_image.size, (0, 0, 0))), np.array(mask_image))))
    else:
        # Vertical concatenation [Rendered Image / Original Image]
        combined_image = Image.fromarray(np.vstack((np.array(rendered_text), np.array(original_image))))
        combined_mask = Image.fromarray(np.vstack((np.array(Image.new("RGB", original_image.size, (0, 0, 0))), np.array(mask_image))))
    
    print("Starting inference...")
    result = run_inference(combined_image, combined_mask, words_path, 
                            num_steps=steps, guidance_scale=guidance_scale, seed=seed)
    
    # Crop result (keep only the original image part)
    width, height = result.size
    if direction == 'horizontal':
        cropped_result = result.crop((width // 2, 0, width, height))
    else:
        cropped_result = result.crop((0, height // 2, width, height))
    
    # Create output directories
    os.makedirs("outputs_my", exist_ok=True)
    os.makedirs("outputs_my/crop", exist_ok=True)
    os.makedirs("outputs_my/mask", exist_ok=True)
    os.makedirs("outputs_my/ori", exist_ok=True)
    os.makedirs("outputs_my/txt", exist_ok=True)
    os.makedirs("outputs_my/rendered", exist_ok=True)  # Added directory for rendered text results
    
    # Get sequence number and save all related files
    seq = get_next_seq_number()
    
    # Save files
    result_paths = {
        "full_result": os.path.join("outputs_my", f"result_{seq}.png"),
        "cropped_result": os.path.join("outputs_my/crop", f"crop_{seq}.png"),
        "mask": os.path.join("outputs_my/mask", f"mask_{seq}.png"),
        "original_image": os.path.join("outputs_my/ori", f"ori_{seq}.png"),
        "text_file": os.path.join("outputs_my/txt", f"words_{seq}.txt"),
        "rendered_text_image": os.path.join("outputs_my/rendered", f"rendered_{seq}.png"), # Added saving for rendered text result
    }
    
    result.save(result_paths["full_result"])
    cropped_result.save(result_paths["cropped_result"])
    mask_image.save(result_paths["mask"])
    original_image.save(result_paths["original_image"])
    rendered_text.save(result_paths["rendered_text_image"]) # Save the rendered text image
    
    # Copy text file
    import shutil
    shutil.copy2(words_path, result_paths["text_file"])
    
    print("\nFiles saved:")
    for desc, path in result_paths.items():
        print(f"{desc.replace('_', ' ').title()}: {path}") # Making the description more readable
    
    return cropped_result


# python run_inference_lora.py --image resource/example/ori/ori_0001.png --mask resource/example/mask/mask_0001.png --words resource/example/txt/words_0001.txt
def main():
    parser = argparse.ArgumentParser(description='Flux Text Generation CLI')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--mask', type=str, required=True,
                        help='Path to mask image')
    parser.add_argument('--words', type=str, required=True,
                        help='Path to text file containing words')
    parser.add_argument('--steps', type=int, default=30,
                        help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=30,
                        help='Guidance scale value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    try:
        process_normal_mode(args.image, args.mask, args.words, 
                            args.steps, args.guidance_scale, args.seed)
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    check_min_version("0.32.1")
    main()
