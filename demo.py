import os
import torch
import numpy as np
import cv2
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import check_min_version, load_image
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
# import uuid # Not used in the final logic shown


def read_words_from_text(input_text):
    """
    Read text/word list:
      - If input_text is a file path, read all non-empty lines from the file;
      - Otherwise, split directly by newline into a list.
    """
    if isinstance(input_text, str) and os.path.exists(input_text):
        with open(input_text, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        words = [line.strip() for line in input_text.splitlines() if line.strip()]
    return words

def generate_prompt(words):
    """
    Generate prompt based on the text list:
      Wrap each text with English single quotes and fill it into the preset template to generate the complete prompt.
    """
    words_str = ', '.join(f"'{word}'" for word in words)
    prompt_template = (
        "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
        "[IMAGE1] is a template image rendering the text, with the words {words}; "
        "[IMAGE2] shows the text content {words} naturally and correspondingly integrated into the image."
    ) #  and the font needs to be bold (Original comment)
    return prompt_template.format(words=words_str)

# Fixed alternative prompt template (without specific words)
prompt_template2 = (
    "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
    "[IMAGE1] is a template image rendering the text, with the words; "
    "[IMAGE2] shows the text content naturally and correspondingly integrated into the image."
)


PIPE = None
def load_flux_pipeline():
    """
    Load Flux model pipeline, move to CUDA, and use bfloat16 data type.
    Use global variable caching to avoid repeated loading.
    """
    global PIPE
    if PIPE is None:
        transformer = FluxTransformer2DModel.from_pretrained(
            "yyyyyxie/textflux", # User-specific path, textflux transformer path
            # "black-forest-labs/FLUX.1-Fill-dev/transformer",
            torch_dtype=torch.bfloat16
        )
        PIPE = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",  # black-forest-labs weights path
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        PIPE.transformer.to(torch.bfloat16)
    return PIPE

def run_inference(image_input, mask_input, words_input, num_steps=50, guidance_scale=30, seed=42):
    """
    Call Flux model pipeline for inference:
      - Both image_input and mask_input are required to be concatenated composite images;
      - Automatically adjust image dimensions to be multiples of 32 to meet model input requirements;
      - Generate prompt based on the word list and pass it to the pipeline for inference.
    """
    # Load image (supports file path or PIL.Image)
    if isinstance(image_input, str):
        inpaint_image = load_image(image_input).convert("RGB")
    else:
        inpaint_image = image_input.convert("RGB")
    if isinstance(mask_input, str):
        extended_mask = load_image(mask_input).convert("RGB")
    else:
        extended_mask = mask_input.convert("RGB")
    
    # Adjust dimensions to be multiples of 32
    width, height = inpaint_image.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    inpaint_image = inpaint_image.resize((new_width, new_height))
    extended_mask = extended_mask.resize((new_width, new_height))
    
    # Process text input
    words = read_words_from_text(words_input)
    prompt = generate_prompt(words)
    print("Generated prompt:", prompt)
    
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    pipe = load_flux_pipeline()
    result = pipe(
        height=new_height,
        width=new_width,
        image=inpaint_image, # Directly pass PIL image
        mask_image=extended_mask, # Directly pass PIL image
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
    Gradio main function for Normal Mode:
      - Directly pass the input image, Mask, and word list to run_inference for inference;
      - Return the generated result image.
    """
    result = run_inference(image, mask, words, num_steps=steps, guidance_scale=guidance_scale, seed=seed)
    return result

# =============================================================================
# 4. Custom Mode: Preprocess Mask, Render text in regions, Concatenate composite image
# =============================================================================
def extract_mask(original, drawn, threshold=30):
    """
    Extract a binary mask from the original image and the user-drawn image:
      - If drawn is a dict and contains a "mask" key, directly binarize this mask;
      - Otherwise, use inversion and difference methods to extract the mask.
    """
    if isinstance(drawn, dict):
        if "mask" in drawn and drawn["mask"] is not None:
            drawn_mask_arr = np.array(drawn["mask"]).astype(np.uint8)
            if drawn_mask_arr.ndim == 3: # Ensure it's grayscale if it has 3 channels (e.g. RGBA from sketch)
                drawn_mask_arr = cv2.cvtColor(drawn_mask_arr, cv2.COLOR_RGB2GRAY)
            _, binary_mask_arr = cv2.threshold(drawn_mask_arr, 50, 255, cv2.THRESH_BINARY)
            return Image.fromarray(binary_mask_arr).convert("RGB")
        else: # Case for when 'image' is provided by the sketch tool
            drawn_img_arr = np.array(drawn["image"]).astype(np.uint8)
            # Assuming the drawing tool uses a white brush on a transparent/original background
            # We are interested in the drawn parts. If black background, invert.
            # If drawn is significantly different from original, it's part of the mask.
            # This part might need adjustment based on how 'drawn["image"]' looks.
            # The provided logic assumes drawn lines are darker or color on white.
            # If drawing white on original, original-drawn could work, or detect non-black pixels if background is black.
            # For sketch tool that returns RGBA with drawing on alpha, the "mask" key path is typical.
            # If "image" is returned, and it's the original with sketches, diff is appropriate.
            drawn_arr_for_diff = 255 - drawn_img_arr # Invert if sketches are black on white
    else: # If 'drawn' is just an image (e.g. from upload)
        drawn_arr_for_diff = np.array(drawn).astype(np.int16)

    original_arr = np.array(original).astype(np.int16)
    # If drawn_arr_for_diff was not set because drawn was a dict without 'mask' and 'image'
    if 'drawn_arr_for_diff' not in locals(): 
        if isinstance(drawn, dict) and "image" in drawn: # This case from original code for "image"
             drawn_img_arr = np.array(drawn["image"]).astype(np.uint8)
             drawn_arr_for_diff = 255 - drawn_img_arr # Assuming black drawing on white for diff
        else: # Fallback or error, this path should ideally not be hit if input is correct
            return Image.new("RGB", original.size, (0,0,0))


    diff = np.abs(drawn_arr_for_diff - original_arr)
    if diff.ndim == 3:
      diff_gray = np.mean(diff, axis=-1)
    else: # If already grayscale
      diff_gray = diff
      
    binary_mask_arr = (diff_gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary_mask_arr).convert("RGB")


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
    scale_factor=2, # Super-sampling factor
    rotate_resample=Image.BICUBIC, # Deprecated in newer Pillow, use Resampling.BICUBIC
    downsample_resample=Image.Resampling.LANCZOS
):
    """
    Render tilted/curved text within a specified region (defined by polygon):
      - First upscale (supersample), then rotate, then downsample to ensure high quality;
      - Dynamically adjust font size and whether to insert spaces between characters based on the region's shape.
    Return the final downsampled RGBA numpy array to the target dimensions (height, width).
    """
    # Handle potential Pillow version differences for resample
    try:
        rotate_resample_enum = rotate_resample if isinstance(rotate_resample, Image.Resampling) else Image.Resampling.BICUBIC
    except AttributeError: # Older Pillow
        rotate_resample_enum = rotate_resample


    big_w = width * scale_factor
    big_h = height * scale_factor

    # Upscale polygon coordinates
    big_polygon = polygon * scale_factor * scale
    rect = cv2.minAreaRect(big_polygon.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    w_rect, h_rect = rect[1] # width and height from minAreaRect
    angle = rect[2]
    if angle < -45:
        angle += 90
        w_rect, h_rect = h_rect, w_rect # Swap width and height
    angle = -angle # Adjust angle for PIL rotation (clockwise)
    # if w_rect < h_rect: # This logic might be redundant if angle adjustment is correct
    #     angle += 90

    vert = False # Vertical text rendering
    # Check if the polygon is oriented vertically
    if (abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng):
        # More robust check for vertical orientation using bounding box of the polygon
        poly_min_x, poly_min_y = np.min(big_polygon, axis=0)
        poly_max_x, poly_max_y = np.max(big_polygon, axis=0)
        poly_w = poly_max_x - poly_min_x
        poly_h = poly_max_y - poly_min_y
        if poly_h > poly_w : # If actual height of polygon is greater than width
            vert = True
            angle = 0 # Force horizontal rendering for vertical text characters

    # Create large image and temporary white background image for text size calculation
    big_img = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
    # tmp = Image.new("RGB", big_img.size, "white") # Not strictly needed if using getbbox
    # tmp_draw = ImageDraw.Draw(tmp)

    # Estimate text size
    # _, _, _tw, _th = tmp_draw.textbbox((0,0), text, font=font) # Pillow 9+
    # For wider compatibility with Pillow versions for textbbox / getsize
    try:
        _left, _top, _right, _bottom = font.getbbox(text)
        _tw = _right - _left
        _th = _bottom - _top
    except AttributeError: # Older Pillow
        _tw, _th = font.getsize(text)


    if _th == 0: # Avoid division by zero
        text_w_ratio = 0
    else:
        text_w_ratio = min(w_rect, h_rect) * (_tw / _th) # Estimated width if text height matches smaller dimension of rect

    # Dynamically adjust font size and spacing
    target_rect_dim_for_text_width = max(w_rect, h_rect)
    target_rect_dim_for_text_height = min(w_rect, h_rect)

    if text_w_ratio <= target_rect_dim_for_text_width:
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100): # Try adding spaces
                text_sp = insert_spaces(text, i)
                try:
                    _, _, tw2, th2_bbox_bottom = font.getbbox(text_sp) # Using getbbox
                    th2 = th2_bbox_bottom - font.getbbox(text_sp)[1]
                except AttributeError:
                    tw2, th2 = font.getsize(text_sp)

                if th2 != 0:
                    if target_rect_dim_for_text_height * (tw2 / th2) > target_rect_dim_for_text_width:
                        text = insert_spaces(text, i - 1 if i > 0 else 0)
                        break
            else: # If loop finished without break
                text = insert_spaces(text, i)


        font_size_ratio = 0.80 # Default fill ratio
        font_size = target_rect_dim_for_text_height * font_size_ratio
    else: # Text is too wide for the area, shrink font
        shrink_factor = 0.75 if vert else 0.85
        if text_w_ratio != 0: # text_w_ratio is actually min(w,h) * text_aspect_ratio
            font_size = (target_rect_dim_for_text_height / (text_w_ratio / target_rect_dim_for_text_width)) * shrink_factor
        else:
            font_size = target_rect_dim_for_text_height * 0.80


    new_font = font.font_variant(size=int(font_size))
    try:
        left, top, right, bottom = new_font.getbbox(text)
        text_pixel_width = right - left
        text_pixel_height = bottom - top
    except AttributeError:
        text_pixel_width, text_pixel_height = new_font.getsize(text)
        top = 0 # Assuming baseline is 0 for getsize


    # Create transparent text rendering layer
    # Determine text layer size based on the rotated rectangle to minimize processing
    # For simplicity, using big_img.size, can be optimized
    layer = Image.new("RGBA", big_img.size, (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(layer)
    
    cx_rect, cy_rect = rect[0] # Center of the minimum area rectangle

    if not vert:
        # Adjust for PIL's text anchor (top-left) and bbox (actual ink)
        text_draw_x = cx_rect - text_pixel_width / 2
        text_draw_y = cy_rect - text_pixel_height / 2 - top # 'top' from getbbox can be negative
        draw_layer.text(
            (text_draw_x, text_draw_y),
            text,
            font=new_font,
            fill=(255, 255, 255, 255)
        )
    else: # Vertical text rendering (character by character)
        # Calculate starting position for vertical text
        # Assuming characters are stacked top to bottom, centered horizontally within the polygon
        current_y = cy_rect - (len(text) * text_pixel_height / len(text.replace(" ",""))) / 2 # Rough estimate
        if text:
            char_height_estimate = text_pixel_height / len(text) if len(text) >0 else 0
            total_text_block_height = 0
            char_widths = []
            char_actual_heights = []

            for char_idx, char_val in enumerate(text):
                try:
                    c_left, c_top, c_right, c_bottom = new_font.getbbox(char_val)
                    char_w = c_right - c_left
                    char_h = c_bottom - c_top
                except AttributeError:
                    char_w, char_h = new_font.getsize(char_val)
                char_widths.append(char_w)
                char_actual_heights.append(char_h)
                total_text_block_height += char_h # Add inter-character spacing if needed

            current_y = cy_rect - total_text_block_height / 2

            for char_idx, char_val in enumerate(text):
                char_w = char_widths[char_idx]
                char_h = char_actual_heights[char_idx]
                try: # getbbox includes descenders/ascenders
                    _, c_top_offset, _, _ = new_font.getbbox(char_val)
                except AttributeError:
                    c_top_offset = 0

                char_x_pos = cx_rect - char_w / 2
                draw_layer.text((char_x_pos, current_y - c_top_offset), char_val, font=new_font, fill=(255,255,255,255))
                current_y += char_h # Move to next line based on character's actual height


    rotated_layer = layer.rotate(
        angle, # Angle from minAreaRect
        expand=True,
        center=(cx_rect, cy_rect), # Rotate around the center of the text
        resample=rotate_resample_enum
    )

    # Paste rotated layer onto the main big_image
    # Calculate offset to paste centered if expand=True changed dimensions
    paste_x = int((big_img.width - rotated_layer.width) / 2)
    paste_y = int((big_img.height - rotated_layer.height) / 2)
    big_img.paste(rotated_layer, (paste_x, paste_y), rotated_layer)

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
        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(cnt)
        if w_bbox * h_bbox < 50: # Filter small regions
            continue
        regions.append({'x': x_bbox, 'y': y_bbox, 'w': w_bbox, 'h': h_bbox, 'contour': cnt})
    
    # Sort regions: primarily by top-y coordinate, secondarily by left-x coordinate
    regions = sorted(regions, key=lambda r: (r['y'], r['x']))
    
    render_img = Image.new("RGBA", original.size, (0, 0, 0, 0))
    try:
        # Ensure the font path is correct for your system
        base_font = ImageFont.truetype("resource/font/Arial-Unicode-Regular.ttf", 40)
    except IOError:
        print("Font not found, using default.")
        base_font = ImageFont.load_default()
    
    for i, region_data in enumerate(regions):
        if i >= len(texts): # Stop if we run out of text lines
            break
        text_to_render = texts[i].strip()
        if not text_to_render: # Skip empty text lines
            continue
        
        contour = region_data['contour']
        polygon = contour.reshape(-1, 2) # Reshape contour for draw_glyph2

        rendered_np = draw_glyph2(
            font=base_font,
            text=text_to_render,
            polygon=polygon, # Pass the contour points as polygon
            vertAng=10,
            scale=1, # Scale of the polygon relative to original image (usually 1)
            width=original.size[0],
            height=original.size[1],
            add_space=True,
            scale_factor=1, # Super-sampling factor for draw_glyph2, 1 means no super-sampling there
            rotate_resample=Image.Resampling.BICUBIC, # Or Image.BICUBIC for older Pillow
            downsample_resample=Image.Resampling.LANCZOS
        )
        rendered_single_glyph_img = Image.fromarray(rendered_np, mode="RGBA")
        render_img = Image.alpha_composite(render_img, rendered_single_glyph_img)
        
    return render_img.convert("RGB")


def choose_concat_direction(height, width):
    """
    Choose concatenation direction based on the original image's aspect ratio:
      - If height is greater than width, use horizontal concatenation;
      - Otherwise, use vertical concatenation.
    """
    return 'horizontal' if height > width else 'vertical'

def get_next_seq_number():
    """
    Find the next available sequential number (format 0001, 0002, ...) in the outputs_my directory.
    When 'result_XXXX.png' does not exist, that number is considered available, and the formatted string XXXX is returned.
    """
    counter = 1
    while True:
        seq_str = f"{counter:04d}"  # Format as a 4-digit number, e.g., "0001"
        result_path = os.path.join("outputs_my", f"result_{seq_str}.png")
        if not os.path.exists(result_path):
            return seq_str
        counter += 1

def flux_demo_custom(original_image_pil, drawn_mask_data, words_multiline, steps, guidance_scale, seed):
    """
    Gradio main function for Custom Mode:
      1. Extract binary mask from the original image and user-drawn data;
      2. Split user-input text by line into a list, each line corresponding to a mask region;
      3. Call render_glyph_multi for each independent region to render tilted/curved text, generating a rendered image;
      4. Choose concatenation direction based on original image size, and concatenate 
         [Rendered Image, Original Image] and [Pure Black Mask, computed_mask] into composite images respectively;
      5. Pass the concatenated images to run_inference, and return the generated result and concatenated preview images.
    """
    if original_image_pil is None:
        # Handle case where no original image is uploaded
        return None, None, None 
        
    computed_mask_pil = extract_mask(original_image_pil, drawn_mask_data)
    texts_list = read_words_from_text(words_multiline)
    rendered_text_pil = render_glyph_multi(original_image_pil, computed_mask_pil, texts_list)
    
    width_orig, height_orig = original_image_pil.size
    # Create an empty (black) image of the same size as the original for the first part of the mask
    empty_half_mask_arr = np.zeros((height_orig, width_orig, 3), dtype=np.uint8) # Ensure 3 channels for RGB
    
    direction = choose_concat_direction(height_orig, width_orig)
    
    original_image_arr = np.array(original_image_pil)
    rendered_text_arr = np.array(rendered_text_pil)
    computed_mask_arr = np.array(computed_mask_pil.convert("RGB")) # Ensure mask is RGB for stacking

    if direction == 'horizontal':
        combined_image_arr = np.hstack((rendered_text_arr, original_image_arr))
        combined_mask_arr = np.hstack((empty_half_mask_arr, computed_mask_arr))
    else: # vertical
        combined_image_arr = np.vstack((rendered_text_arr, original_image_arr))
        combined_mask_arr = np.vstack((empty_half_mask_arr, computed_mask_arr))
    
    composite_image_pil = Image.fromarray(combined_image_arr)
    composite_mask_pil = Image.fromarray(combined_mask_arr)
    
    result_pil = run_inference(composite_image_pil, composite_mask_pil, words_multiline, 
                               num_steps=steps, guidance_scale=guidance_scale, seed=seed)

    # Crop result, keeping only the scene image part
    width_result, height_result = result_pil.size
    if direction == 'horizontal':  # Horizontal concatenation
        cropped_result_pil = result_pil.crop((width_result // 2, 0, width_result, height_result))
    else:  # Vertical concatenation
        cropped_result_pil = result_pil.crop((0, height_result // 2, width_result, height_result))
    
    # Save the cropped result and other intermediate files
    os.makedirs("outputs_my", exist_ok=True)
    os.makedirs("outputs_my/crop", exist_ok=True)
    os.makedirs("outputs_my/mask", exist_ok=True) # For the computed mask from drawing
    os.makedirs("outputs_my/ori", exist_ok=True) # For the original uploaded image
    os.makedirs("outputs_my/rendered_template", exist_ok=True) # For the rendered text template
    # os.makedirs("outputs_my/composite", exist_ok=True) # Optional: directory for saving concatenated input images
    os.makedirs("outputs_my/txt", exist_ok=True) # Directory for saving input text

    seq = get_next_seq_number()

    result_filename = os.path.join("outputs_my", f"result_full_{seq}.png") # Full inference output
    crop_filename = os.path.join("outputs_my", "crop", f"crop_result_{seq}.png")
    mask_filename = os.path.join("outputs_my", "mask", f"computed_mask_{seq}.png")
    ori_filename = os.path.join("outputs_my", "ori", f"original_input_{seq}.png")
    rendered_template_filename = os.path.join("outputs_my", "rendered_template", f"text_template_{seq}.png")
    # composite_img_filename = os.path.join("outputs_my", "composite", f"composite_image_input_{seq}.png")
    # composite_mask_filename = os.path.join("outputs_my", "composite", f"composite_mask_input_{seq}.png")
    txt_filename = os.path.join("outputs_my", "txt", f"input_words_{seq}.txt")

    result_pil.save(result_filename)
    cropped_result_pil.save(crop_filename)
    computed_mask_pil.save(mask_filename)
    original_image_pil.save(ori_filename)
    rendered_text_pil.save(rendered_template_filename)
    # composite_image_pil.save(composite_img_filename) # Optional save
    # composite_mask_pil.save(composite_mask_filename) # Optional save
    
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(words_multiline)
        
    return cropped_result_pil, composite_image_pil, composite_mask_pil


# =============================================================================
# 5. Gradio Interface (Using Tabs to differentiate two modes)
# =============================================================================
with gr.Blocks(title="Flux Inference Demo") as demo:
    gr.Markdown("## Flux Inference Demo")
    
    with gr.Tabs():
        # ---------------- Normal Mode ----------------
        with gr.TabItem("Normal Mode"):
            gr.Markdown(
                "In **Normal Mode**, you provide a complete image that already has the text template on one half "
                "and the scene on the other half. You also provide a corresponding mask."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Image Input")
                    image_normal = gr.Image(type="pil", label="Input Composite Image (Template + Scene)")
                    gr.Markdown("### Mask Input")
                    mask_normal = gr.Image(type="pil", label="Input Composite Mask (Empty + Scene Mask)")
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Parameter Settings")
                    words_normal = gr.Textbox(lines=5, placeholder="Enter words here, one per line, corresponding to the template.", label="Word List")
                    steps_normal = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Inference Steps")
                    guidance_scale_normal = gr.Slider(minimum=1, maximum=50, step=1, value=30, label="Guidance Scale")
                    seed_normal = gr.Number(value=42, label="Random Seed")
                    run_normal = gr.Button("Generate Result")
            output_normal = gr.Image(type="pil", label="Generated Result")
            run_normal.click(fn=flux_demo_normal, 
                             inputs=[image_normal, mask_normal, words_normal, steps_normal, guidance_scale_normal, seed_normal],
                             outputs=output_normal)
        
        # ---------------- Custom Mode ----------------
        with gr.TabItem("Custom Mode"):
            gr.Markdown(
                "In **Custom Mode**, you upload an original scene image, draw masks on it, and provide text. "
                "The script will then create the text template and composite images automatically."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Image Input")
                    original_image_custom = gr.Image(type="pil", label="Upload Original Scene Image")
                    gr.Markdown("### Draw Mask")
                    # The 'sketch' tool provides `{"image": PIL.Image, "mask": PIL.Image}`
                    # where "mask" is the drawn parts.
                    mask_drawing_custom = gr.Image(type="pil", label="Draw Mask(s) on Original Image", tool="sketch", image_mode="RGB")
                    # mask_drawing_custom = gr.Image(type="pil", label="Draw Mask on Original Image", tool="sketch", brush_color="white") # Example if needing specific brush

                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### Parameter Settings")
                    words_custom = gr.Textbox(lines=5, placeholder="Enter text for each mask region, one per line (top-to-bottom order of masks)", label="Text List")
                    steps_custom = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Inference Steps")
                    guidance_scale_custom = gr.Slider(minimum=1, maximum=50, step=1, value=30, label="Guidance Scale")
                    seed_custom = gr.Number(value=42, label="Random Seed")
                    run_custom = gr.Button("Generate Result")
            
            # Output area uses Tabs to display generated results and input preview images
            with gr.Tabs():
                with gr.TabItem("Generated Result (Cropped)"):
                    output_result_custom = gr.Image(type="pil", label="Final Generated Scene with Text")
                with gr.TabItem("Input Preview (Composite Images for Model)"):
                    output_composite_custom = gr.Image(type="pil", label="Concatenated Image (Template + Original)")
                    output_mask_custom = gr.Image(type="pil", label="Concatenated Mask (Empty + Drawn Mask)")
            
            # Synchronize original image to drawing mask (allows user to draw directly on original image as background)
            def sync_image_to_draw(img):
                return img # The image itself will be used as background for sketch tool
            original_image_custom.change(fn=sync_image_to_draw, inputs=original_image_custom, outputs=mask_drawing_custom)
            
            run_custom.click(fn=flux_demo_custom, 
                             inputs=[original_image_custom, mask_drawing_custom, words_custom, steps_custom, guidance_scale_custom, seed_custom],
                             outputs=[output_result_custom, output_composite_custom, output_mask_custom])
    
    gr.Markdown(
        """
        ### Instructions
        - **Normal Mode**: Directly upload a pre-prepared composite image (text template + scene) and its corresponding composite mask. Input the word list. This mode expects you to have already combined the template and scene.
        - **Custom Mode**: Upload an original scene image. Use the sketch tool to draw mask(s) where text should appear. Input the corresponding text for each masked region (one line of text per region, ordered from top-to-bottom based on mask positions). The script will automatically generate the text template, combine it with your scene, and perform inference.
        - **Output Files**: Generated images and inputs are saved in the `outputs_my` directory and its subdirectories (`crop`, `mask`, `ori`, `rendered_template`, `txt`).
        """
    )

if __name__ == "__main__":
    check_min_version("0.32.1") # Diffusers library version check
    demo.launch()