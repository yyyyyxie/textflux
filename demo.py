import os
import torch
import numpy as np
import cv2
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import check_min_version, load_image
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import uuid
# from gradio_canvas import Canvas


def read_words_from_text(input_text):
    """
    Reads words/list of words:
      - If input_text is a file path, it reads all non-empty lines from the file.
      - Otherwise, it directly splits the input by newlines into a list.
    """
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
            "yyyyyxie/textflux",
            # "black-forest-labs/FLUX.1-Fill-dev/transformer",
            torch_dtype=torch.bfloat16
        )
        PIPE = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        PIPE.transformer.to(torch.bfloat16)
    return PIPE

def run_inference(image_input, mask_input, words_input, num_steps=50, guidance_scale=30, seed=42):
    """
    Invokes the Flux model pipeline for inference:
      - Both image_input and mask_input are required to be concatenated composite images.
      - Automatically adjusts image dimensions to be multiples of 32 to meet model input requirements.
      - Generates a prompt based on the word list and passes it to the pipeline for inference execution.
    """

    if isinstance(image_input, str):
        inpaint_image = load_image(image_input).convert("RGB")
    else:
        inpaint_image = image_input.convert("RGB")
    if isinstance(mask_input, str):
        extended_mask = load_image(mask_input).convert("RGB")
    else:
        extended_mask = mask_input.convert("RGB")
    
    width, height = inpaint_image.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    inpaint_image = inpaint_image.resize((new_width, new_height))
    extended_mask = extended_mask.resize((new_width, new_height))
    
    words = read_words_from_text(words_input)
    prompt = generate_prompt(words)
    print("Generated prompt:", prompt)
    
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

# =============================================================================
# 3. Normal Mode: Direct Inference Call
# =============================================================================
def flux_demo_normal(image, mask, words, steps, guidance_scale, seed):
    """
    Gradio main function for normal mode:
      - Directly passes the input image, mask, and word list to run_inference for inference.
      - Returns the generated result image.
    """
    result = run_inference(image, mask, words, num_steps=steps, guidance_scale=guidance_scale, seed=seed)
    return result

# =============================================================================
# 4. Custom Mode: Mask Preprocessing, Region Text Rendering, and Composite Image Concatenation
# =============================================================================
def extract_mask(original, drawn, threshold=30):
    """
    Extracts a binary mask from the original image and the user-drawn image:
      - If 'drawn' is a dictionary and contains a "mask" key, that mask is directly binarized.
      - Otherwise, the mask is extracted using inversion and differentiation methods.
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
    Inserts a specified number of spaces between each character to adjust the spacing during text rendering.
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
    Renders skewed/curved text within a specified region (defined by polygon):
      - Upsamples (supersamples) then rotates, then downsamples to ensure high quality.
      - Dynamically adjusts font size and whether to insert spaces between characters based on the region's shape.
    Returns the final downsampled RGBA numpy array to the target dimensions (height, width).
    """
    big_w = width * scale_factor
    big_h = height * scale_factor

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
      - Extracts region locations via contour detection and sorts them from top to bottom, then left to right.
      - Calls draw_glyph2 to render corresponding text within each region (supports skewing/curving).
      - Overlays the rendering results of each region onto a transparent black background, outputting the final rendered image.
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
    Selects the concatenation direction based on the original image's aspect ratio:
      - If height is greater than width, horizontal concatenation is used.
      - Otherwise, vertical concatenation is used.
    """
    return 'horizontal' if height > width else 'vertical'

def get_next_seq_number():
    """
    Finds the next available sequential number (format: 0001, 0002,...) in the 'outputs_my' directory.
    When 'result_XXXX.png' does not exist, that number is considered available, and the formatted string XXXX is returned.
    """
    counter = 1
    while True:
        seq_str = f"{counter:04d}"  
        result_path = os.path.join("outputs_my", f"result_{seq_str}.png")
        if not os.path.exists(result_path):
            return seq_str
        counter += 1

def flux_demo_custom(original_image, drawn_mask, words, steps, guidance_scale, seed):
    """
    Gradio main function for custom mode:
      1. Extracts a binary mask from the original image and user-drawn data.
      2. Splits the user-input text into a list by line, with each line corresponding to a mask region.
      3. Calls render_glyph_multi for each independent region to render skewed/curved text, generating a rendered image.
      4. Selects the concatenation direction based on the original image's dimensions, concatenating [rendered_image, original_image] and [pure_black_mask, computed_mask] into composite images respectively.
      5. Passes the concatenated images to run_inference, returning the generated result and a concatenated preview image.
    """
    computed_mask = extract_mask(original_image, drawn_mask)
    texts = read_words_from_text(words)
    render_img = render_glyph_multi(original_image, computed_mask, texts)
    
    width, height = original_image.size
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    
    direction = choose_concat_direction(height, width)
    if direction == 'horizontal':
        combined_image = np.hstack((np.array(render_img), np.array(original_image)))
        combined_mask = np.hstack((empty_mask, np.array(computed_mask.convert("L"))))
    else:
        combined_image = np.vstack((np.array(render_img), np.array(original_image)))
        combined_mask = np.vstack((empty_mask, np.array(computed_mask.convert("L"))))
    
    combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
    composite_image = Image.fromarray(combined_image)
    composite_mask = Image.fromarray(combined_mask)
    
    result = run_inference(composite_image, composite_mask, words, num_steps=steps, guidance_scale=guidance_scale, seed=seed)

    # Crop the result, keeping only the scene image portion.
    width, height = result.size
    if direction == 'horizontal': 
        cropped_result = result.crop((width // 2, 0, width, height))
    else: 
        cropped_result = result.crop((0, height // 2, width, height))
    
    # Save results
    os.makedirs("outputs_my", exist_ok=True)
    os.makedirs("outputs_my/crop", exist_ok=True)
    os.makedirs("outputs_my/mask", exist_ok=True)
    os.makedirs("outputs_my/ori", exist_ok=True)
    # os.makedirs("outputs_my/composite", exist_ok=True)  
    os.makedirs("outputs_my/txt", exist_ok=True) 

    seq = get_next_seq_number()

    result_filename = os.path.join("outputs_my", f"result_{seq}.png")
    crop_filename = os.path.join("outputs_my", "crop", f"crop_{seq}.png")
    mask_filename = os.path.join("outputs_my", "mask", f"mask_{seq}.png")
    ori_filename = os.path.join("outputs_my", "ori", f"ori_{seq}.png")  
    # composite_filename = os.path.join("outputs_my", "composite", f"composite_{seq}.png") 
    txt_filename = os.path.join("outputs_my", "txt", f"words_{seq}.txt") 


    # Save images
    result.save(result_filename)
    cropped_result.save(crop_filename)
    computed_mask.save(mask_filename)
    original_image.save(ori_filename)  
    # composite_image.save(composite_filename) 
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(words)

    return cropped_result, composite_image, composite_mask


# =============================================================================
# 5. Gradio Interface (using Tabs to differentiate between two modes)
# =============================================================================
with gr.Blocks(title="Flux Inference Demo") as demo:
    gr.Markdown("## Flux Inference Demo")
    
    with gr.Tabs():
        with gr.TabItem("Normal Mode"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Image Input")
                    image_normal = gr.Image(type="pil", label="Image Input")
                    gr.Markdown("### Mask Input")
                    mask_normal = gr.Image(type="pil", label="Mask Input")
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Parameter Settings")
                    words_normal = gr.Textbox(lines=5, placeholder="Please enter words here, one per line", label="Text List")
                    steps_normal = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Inference Step")
                    guidance_scale_normal = gr.Slider(minimum=1, maximum=50, step=1, value=30, label="Guidance Scale")
                    seed_normal = gr.Number(value=42, label="Random Seed")
                    run_normal = gr.Button("Generated Results")
            output_normal = gr.Image(type="pil", label="Generated Results")
            run_normal.click(fn=flux_demo_normal, 
                             inputs=[image_normal, mask_normal, words_normal, steps_normal, guidance_scale_normal, seed_normal],
                             outputs=output_normal)
        

        with gr.TabItem("Custom Mode"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Image Input")
                    original_image_custom = gr.Image(type="pil", label="Upload Original Image")
                    gr.Markdown("### Drawn Mask")
                    mask_drawing_custom = gr.Image(type="pil", label="Draw Mask on Original Image", tool="sketch")

                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Parameter Settings")
                    words_custom = gr.Textbox(lines=5, placeholder="Please enter the corresponding text for each line (corresponding to each mask region)", label="Text List")
                    steps_custom = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Inference Step")
                    guidance_scale_custom = gr.Slider(minimum=1, maximum=50, step=1, value=30, label="Guidance Scale")
                    seed_custom = gr.Number(value=42, label="Random Seed")
                    run_custom = gr.Button("Generated Results")

            with gr.Tabs():
                with gr.TabItem("Generated Results"):
                    output_result_custom = gr.Image(type="pil", label="Generated Results")
                with gr.TabItem("Input Preview"):
                    output_composite_custom = gr.Image(type="pil", label="Concatenated Original Image")
                    output_mask_custom = gr.Image(type="pil", label="Concatenated Mask")

            original_image_custom.change(fn=lambda x: x, inputs=original_image_custom, outputs=mask_drawing_custom)
            run_custom.click(fn=flux_demo_custom, 
                             inputs=[original_image_custom, mask_drawing_custom, words_custom, steps_custom, guidance_scale_custom, seed_custom],
                             outputs=[output_result_custom, output_composite_custom, output_mask_custom])
    
    gr.Markdown(
        """
        ### Instructions
        - **Normal Mode**: Directly upload an image, mask, and a list of words to generate the result image.
        - **Custom Mode**: Upload an original image, then draw a mask on it. Enter the corresponding text for each masked region to generate a composite image with rendered text in those areas, and then perform inference.
        """
    )

if __name__ == "__main__":
    check_min_version("0.32.1")
    demo.launch()

