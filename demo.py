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
    读取文本/单词列表：
      - 如果 input_text 为文件路径，则从文件中读取所有非空行；
      - 否则直接按换行拆分成列表。
    """
    if isinstance(input_text, str) and os.path.exists(input_text):
        with open(input_text, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        words = [line.strip() for line in input_text.splitlines() if line.strip()]
    return words

def generate_prompt(words):
    """
    根据文本列表生成提示词：
      将每个文本用英文单引号包裹后填入预设模板中，生成完整提示词。
    """
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
    """
    加载 Flux 模型 pipeline，转移到 CUDA 并采用 bfloat16 数据类型。
    使用全局变量缓存，避免重复加载。
    """
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
    调用 Flux 模型 pipeline 进行推理：
      - 输入 image_input 与 mask_input 均要求为拼接后的复合图像；
      - 自动调整图片尺寸为 32 的倍数以满足模型输入要求；
      - 根据单词列表生成提示词，并传入 pipeline 执行推理。
    """
    # 加载图片（支持文件路径或 PIL.Image）
    if isinstance(image_input, str):
        inpaint_image = load_image(image_input).convert("RGB")
    else:
        inpaint_image = image_input.convert("RGB")
    if isinstance(mask_input, str):
        extended_mask = load_image(mask_input).convert("RGB")
    else:
        extended_mask = mask_input.convert("RGB")
    
    # 调整尺寸为 32 的倍数
    width, height = inpaint_image.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    inpaint_image = inpaint_image.resize((new_width, new_height))
    extended_mask = extended_mask.resize((new_width, new_height))
    
    # 处理文本输入
    words = read_words_from_text(words_input)
    prompt = generate_prompt(words)
    print("生成的提示词:", prompt)
    
    # 图像预处理（转 Tensor 及归一化）
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
# 3. 普通模式：直接调用推理
# =============================================================================
def flux_demo_normal(image, mask, words, steps, guidance_scale, seed):
    """
    普通模式的 Gradio 主函数：
      - 直接将输入的图片、Mask和单词列表传入 run_inference 进行推理；
      - 返回生成的结果图像。
    """
    result = run_inference(image, mask, words, num_steps=steps, guidance_scale=guidance_scale, seed=seed)
    return result

# =============================================================================
# 4. 自定义模式：预处理 Mask、区域文字渲染、拼接复合图像
# =============================================================================
def extract_mask(original, drawn, threshold=30):
    """
    从原始图片与用户手绘后的图像中提取二值 mask：
      - 如果 drawn 为 dict 且包含 "mask" 键，则直接对该 mask 二值化；
      - 否则采用反转与差分方法提取 mask。
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
    在每个字符之间插入指定数量的空格，用于调整文字渲染时的间距。
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
    在指定区域内（由 polygon 定义）渲染倾斜/弯曲文字：
      - 先放大（超采样）后旋转、再下采样以保证高质量；
      - 根据区域形状动态调整字体大小及是否需要在字符间插入空格。
    返回最终下采样到目标尺寸 (height, width) 的 RGBA numpy 数组。
    """
    big_w = width * scale_factor
    big_h = height * scale_factor

    # 放大 polygon 坐标
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

    # 创建大图与临时白底图
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

    # 创建透明文字渲染层
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
    针对 computed_mask 中每个独立区域：
      - 通过轮廓提取区域位置，并按从上到下、从左到右排序；
      - 调用 draw_glyph2 在各区域内渲染对应文本（支持倾斜/弯曲）；
      - 将各区域的渲染结果叠加到一张透明黑底图上，输出最终渲染图。
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
    根据原图宽高比选择拼接方向：
      - 若高度大于宽度，则采用水平拼接；
      - 否则采用垂直拼接。
    """
    return 'horizontal' if height > width else 'vertical'

def get_next_seq_number():
    """
    从 outputs_my 目录中查找下一个可用的顺序编号（格式为 0001,0002,...）。
    当 'result_XXXX.png' 不存在时，即认为该编号可用，返回格式化字符串 XXXX。
    """
    counter = 1
    while True:
        seq_str = f"{counter:04d}"  # 格式化为4位数字，例如 "0001"
        result_path = os.path.join("outputs_my", f"result_{seq_str}.png")
        if not os.path.exists(result_path):
            return seq_str
        counter += 1

def flux_demo_custom(original_image, drawn_mask, words, steps, guidance_scale, seed):
    """
    自定义模式的 Gradio 主函数：
      1. 从原图和用户手绘数据提取二值 mask；
      2. 将用户输入的文本按行拆分成列表，每行对应一个 mask 区域；
      3. 针对每个独立区域调用 render_glyph_multi 渲染倾斜/弯曲文字，生成渲染图；
      4. 根据原图尺寸选择拼接方向，将【渲染图, 原图】与【纯黑 mask, computed_mask】分别拼接成复合图像；
      5. 将拼接后的图像传入 run_inference，返回生成结果及拼接预览图。
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

    # 裁剪结果，只保留场景图片部分
    width, height = result.size
    if direction == 'horizontal':  # 水平拼接
        cropped_result = result.crop((width // 2, 0, width, height))
    else:  # 垂直拼接
        cropped_result = result.crop((0, height // 2, width, height))
    
    # 保存裁剪后的结果
    os.makedirs("outputs_my", exist_ok=True)
    os.makedirs("outputs_my/crop", exist_ok=True)
    os.makedirs("outputs_my/mask", exist_ok=True)
    os.makedirs("outputs_my/ori", exist_ok=True)
    # os.makedirs("outputs_my/composite", exist_ok=True)  # 新增用于保存拼接图片的目录
    os.makedirs("outputs_my/txt", exist_ok=True)  # 新增用于保存文本的目录

    # 获取下一个可用的顺序编号
    seq = get_next_seq_number()

    # 根据编号生成文件名
    result_filename = os.path.join("outputs_my", f"result_{seq}.png")
    crop_filename = os.path.join("outputs_my", "crop", f"crop_{seq}.png")
    mask_filename = os.path.join("outputs_my", "mask", f"mask_{seq}.png")
    ori_filename = os.path.join("outputs_my", "ori", f"ori_{seq}.png")  # 原始图片的文件名
    # composite_filename = os.path.join("outputs_my", "composite", f"composite_{seq}.png")  # 拼接图片
    txt_filename = os.path.join("outputs_my", "txt", f"words_{seq}.txt")  # 输入文本的文件名


    # 保存图片
    result.save(result_filename)
    cropped_result.save(crop_filename)
    computed_mask.save(mask_filename)
    original_image.save(ori_filename)  # 保存输入的原始图片
    # composite_image.save(composite_filename)  # 保存拼接后的图片
    # 保存用户输入的文本到 txt 文件中
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(words)

    # unique_id = uuid.uuid4().hex[:8]
    # result_filename = f"outputs_my/result_{unique_id}.png"
    # crop_filename = f"outputs_my/crop/crop_{unique_id}.png"
    # result.save(result_filename)
    # cropped_result.save(crop_filename)
    # computed_mask.save(f"outputs_my/mask/mask_{unique_id}.png")
    # return result, composite_image, composite_mask
    return cropped_result, composite_image, composite_mask


# =============================================================================
# 5. Gradio 界面（使用 Tabs 分页区分两种模式）
# =============================================================================
with gr.Blocks(title="Flux 推理 Demo") as demo:
    gr.Markdown("## Flux Inference Demo")
    
    with gr.Tabs():
        # ---------------- 普通模式 ----------------
        with gr.TabItem("普通模式"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### 图片输入")
                    image_normal = gr.Image(type="pil", label="输入图片")
                    gr.Markdown("### Mask 输入")
                    mask_normal = gr.Image(type="pil", label="输入 Mask")
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### 参数设置")
                    words_normal = gr.Textbox(lines=5, placeholder="请在此输入单词，每行一个", label="单词列表")
                    steps_normal = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="推理步数")
                    guidance_scale_normal = gr.Slider(minimum=1, maximum=50, step=1, value=30, label="Guidance Scale")
                    seed_normal = gr.Number(value=42, label="随机种子")
                    run_normal = gr.Button("生成结果")
            output_normal = gr.Image(type="pil", label="生成结果")
            run_normal.click(fn=flux_demo_normal, 
                             inputs=[image_normal, mask_normal, words_normal, steps_normal, guidance_scale_normal, seed_normal],
                             outputs=output_normal)
        
        # ---------------- 自定义模式 ----------------
        with gr.TabItem("自定义模式"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### 图片输入")
                    original_image_custom = gr.Image(type="pil", label="上传原始图片")
                    gr.Markdown("### 手绘 Mask")
                    mask_drawing_custom = gr.Image(type="pil", label="在原图上手绘 Mask", tool="sketch")

                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### 参数设置")
                    words_custom = gr.Textbox(lines=5, placeholder="请每行输入对应文本（对应每个 mask 区域）", label="文本列表")
                    steps_custom = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="推理步数")
                    guidance_scale_custom = gr.Slider(minimum=1, maximum=50, step=1, value=30, label="Guidance Scale")
                    seed_custom = gr.Number(value=42, label="随机种子")
                    run_custom = gr.Button("生成结果")
            # 输出区域采用 Tabs 分页展示生成结果及输入预览图
            with gr.Tabs():
                with gr.TabItem("生成结果"):
                    output_result_custom = gr.Image(type="pil", label="生成结果")
                with gr.TabItem("输入预览"):
                    output_composite_custom = gr.Image(type="pil", label="拼接后的原图")
                    output_mask_custom = gr.Image(type="pil", label="拼接后的 Mask")
            # 同步原图到手绘 Mask（方便用户直接在原图上绘制）
            original_image_custom.change(fn=lambda x: x, inputs=original_image_custom, outputs=mask_drawing_custom)
            run_custom.click(fn=flux_demo_custom, 
                             inputs=[original_image_custom, mask_drawing_custom, words_custom, steps_custom, guidance_scale_custom, seed_custom],
                             outputs=[output_result_custom, output_composite_custom, output_mask_custom])
    
    gr.Markdown(
        """
        ### 使用说明
        - 普通模式：直接上传图片、Mask及单词列表，生成结果图像。
        - 自定义模式：上传原始图片后在上面手绘 Mask，再输入对应每个区域的文本，生成区域文字渲染后拼接的复合图像，并进行推理。
        """
    )

if __name__ == "__main__":
    check_min_version("0.32.1")
    demo.launch()

