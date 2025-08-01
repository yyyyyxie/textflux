import os
import json
import random
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import random

PREFERED_RESOLUTIONS = [672, 688, 720, 752, 800, 832, 880, 944, 1024]  

DATA_PATHS = [
    ['datasets/anyword/ArT/data.json', 'datasets/anyword/ArT/rename_artimg_train'],
    ['datasets/anyword/CTW1500/data.json', 'datasets/anyword/CTW1500/ctwtrain_text_image'],
    ['datasets/anyword/ic13/data.json', 'datasets/anyword/ic13/train_images'],
    ['datasets/anyword/LSVT/data.json', 'datasets/anyword/LSVT/rename_lsvtimg_train'],
    ['datasets/anyword/mlt2017/data.json', 'datasets/anyword/mlt2017/images'],
    ['datasets/anyword/MLT19_train/data.json', 'datasets/anyword/MLT19_train/images'],
    ['datasets/anyword/ReCTS/data.json', 'datasets/anyword/ReCTS/ReCTS_train_images'],
    ['datasets/anyword/textocr/data.json', 'datasets/anyword/textocr/train_images'],
    ['datasets/anyword/totaltext/data.json', 'datasets/anyword/totaltext/train_images'],
    ]

FONT_PATH = "./resource/font/Arial-Unicode-Regular.ttf"

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def generate_prompt_for_inpaint(words):
    words_str = ', '.join(f"'{word}'" for word in words)
    prompt_template = (
        "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
        "[IMAGE1] is a template image rendering the text, with the words {words}; "
        "[IMAGE2] shows the text content {words} naturally and correspondingly integrated into the image."
    )
    return prompt_template.format(words=words_str)

def insert_spaces(text, num_spaces):
    if len(text) <= 1:
        return text
    return (' ' * num_spaces).join(list(text))


def draw_glyph_flexible(font, text, width, height, max_font_size=140):
    if width <= 0:
        width = 1

    # height = max(width // 4, height)
    height = min(width // 6, height)
    
    img = Image.new(mode='1', size=(width, height), color=0)

    if not text or not text.strip():
        return img
        
    draw = ImageDraw.Draw(img)

    # --- Adaptive calculation of font size ---
    # Initial guess size
    g_size = 50
    new_font = font.font_variant(size=g_size)

    # get size
    left, top, right, bottom = new_font.getbbox(text)
    text_width_initial = max(right - left, 1)  
    text_height_initial = max(bottom - top, 1)

    width_ratio = width * 0.9 / text_width_initial
    height_ratio = height * 0.9 / text_height_initial
    ratio = min(width_ratio, height_ratio)
    final_font_size = int(g_size * ratio)
    
    # Apply the upper limit of font size
    if width > 1280:
        max_font_size = 180

    if width > 2048:
        max_font_size = 280

    final_font_size = min(final_font_size, max_font_size)
    new_font = font.font_variant(size=max(final_font_size, 10))

    draw.text(
        (width / 2, height / 2), 
        text, 
        font=new_font, 
        fill='white', 
        anchor='mm'  # Middle-Middle anchor
    )
    return img



def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="1:1"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class DynamicConcatDataset(Dataset):
    # 1. Change the constructor to accept a list of paths
    def __init__(self, img_size=1024, max_lines=1, max_chars=35, font_path=FONT_PATH):
        """
        Initializes the dataset.
        :param dataset_paths: A list where each element is a pair of [json_path, img_root].
        """
        print(f"Initializing DynamicConcatDataset with {len(DATA_PATHS)} datasets.")
        # self.img_root is no longer needed
        self.img_size = PREFERED_RESOLUTIONS
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.dataset_paths = DATA_PATHS
        
        try:
            self.font = ImageFont.truetype(font_path, size=60)
        except IOError:
            print(f"Font file not found at {font_path}. Using default font.")
            self.font = ImageFont.load_default()
        
        # Pass the list of paths to _load_data
        self.data_list = self._load_data(self.dataset_paths)
        print(f'All datasets loaded, total valid images = {len(self.data_list)}')

    # 2. Modify _load_data to handle multiple paths
    def _load_data(self, dataset_paths):
        """Loads all JSON files and associates each sample with its img_root."""
        full_data_list = []
        # Loop through each [json_path, img_root] pair
        for json_path, img_root in dataset_paths:
            print(f"  > Loading {json_path}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            for gt in content['data_list']:
                if 'annotations' not in gt or not gt['annotations']: continue
                valid_annotations = [ann for ann in gt['annotations'] if ann.get('polygon') and ann.get('text')]
                if not valid_annotations: continue
                
                info = {
                    'img_name': gt['img_name'],
                    'annotations': valid_annotations,
                    'img_root': img_root  # KEY CHANGE: Store the corresponding img_root with each sample
                }
                full_data_list.append(info)
        return full_data_list

    def __getitem__(self, idx):
        try:
            item_data = self.data_list[idx]
            
            # 3. Modify __getitem__ to use the dynamic img_root
            # KEY CHANGE: Get the corresponding img_root from item_data
            img_path = os.path.join(item_data['img_root'], item_data['img_name'])
            
            real_image_pil = Image.open(img_path).convert('RGB')
            w, h = real_image_pil.size
            
            # ... The rest of the processing logic remains the same ...
            
            annotations = item_data['annotations']
            selected_ann = random.choice(annotations)
            text = selected_ann.get('text', '')[:self.max_chars]
            polygon = selected_ann.get('polygon')
            if not polygon or len(polygon) < 3 or not text or w > 5000 or h > 5000:
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            render_pil_small = draw_glyph_flexible(self.font, text, width=w, height=h)
            
            render_pil_rgb = render_pil_small.convert('RGB')
            render_img_np = np.array(render_pil_rgb)

            mask_np_real = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_np_real, [np.array(polygon, dtype=np.int32)], 255)
            
            random_value = random.random()
            kernel = np.ones((3, 3), dtype=np.uint8)
            if random_value < 0.7: pass
            elif random_value < 0.8: mask_np_real = cv2.dilate(mask_np_real, kernel, iterations=1)
            elif random_value < 0.9: mask_np_real = cv2.erode(mask_np_real, kernel, iterations=1)
            elif random_value < 0.95: mask_np_real = cv2.dilate(mask_np_real, kernel, iterations=2)
            else: mask_np_real = cv2.erode(mask_np_real, kernel, iterations=2)

            empty_mask_np = np.zeros((render_pil_small.height, render_pil_small.width), dtype=np.uint8)
            real_image_np = np.array(real_image_pil)

            combined_image_np = np.vstack((render_img_np, real_image_np))
            combined_mask_np = np.vstack((empty_mask_np, mask_np_real))
                
            selected_texts = [text]
            prompt = generate_prompt_for_inpaint(selected_texts)
            
            # drop_prob = 0.1
            # if random.random() < drop_prob:
            #     prompt = ""
            
            combined_image_pil = Image.fromarray(combined_image_np)
            combined_mask_pil = Image.fromarray(combined_mask_np)
            
            img_size_item = random.choice(self.img_size)
            combined_image_pil = image_resize(combined_image_pil, img_size_item)
            
            old_w, old_h = combined_image_pil.size
            new_w = (old_w // 32) * 32
            new_h = (old_h // 32) * 32

            combined_image_pil = combined_image_pil.resize((new_w, new_h))

            combined_mask_pil = image_resize(combined_mask_pil, img_size_item)
            combined_mask_pil = combined_mask_pil.resize((new_w, new_h))
            image_np = (np.array(combined_image_pil) / 127.5) - 1
            image_tensor = torch.from_numpy(image_np)
            image_tensor = image_tensor.permute(2, 0, 1)

            mask_np_normalized = np.array(combined_mask_pil) / 255.0
            mask_tensor = torch.from_numpy(mask_np_normalized)
            
            return image_tensor, prompt, mask_tensor

        except Exception as e:
            print(f"Error processing index {idx} ({self.data_list[idx].get('img_name', '')}): {e}")
            import traceback
            traceback.print_exc()
            return self.__getitem__(random.randint(0, len(self) - 1))

    def __len__(self):
        return len(self.data_list)



class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=768, caption_type='txt', random_ratio=False, use_mask=True, expand_prompt=True):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.use_mask = use_mask
        self.mask_dir = os.path.join(img_dir, "mask") if use_mask else None
        self.expand_prompt = expand_prompt  # Control whether to expand the prompt word

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')

            if isinstance(self.img_size, list):
                current_size = random.choice(self.img_size)
            else:
                current_size = self.img_size

            if not os.path.exists(self.images[idx]):
                raise FileNotFoundError(f"Image not found: {self.images[idx]}")

            # Random ratio cropping
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)

            # Resize image
            # print("original size:", img.size) 
            img = image_resize(img, current_size)
            # print("after resize:", img.size) 
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            
            # Load caption
            json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            if self.caption_type == "json":
                prompt = json.load(open(json_path))['caption']
            else:
                prompt = open(json_path).read()

            if self.expand_prompt:
                words = [line.strip() for line in prompt.splitlines() if line.strip()]
                prompt = generate_prompt(words)

            # Load mask if required
            if self.use_mask and self.mask_dir:
                # image_basename = os.path.basename(self.images[idx]).split('.')[0] + "_mask.png"
                # import pdb; pdb.set_trace() 
                image_basename = os.path.basename(self.images[idx])
                image_name, image_ext = os.path.splitext(image_basename)
                mask_name = f"{image_name}_mask{image_ext}" 

                mask_path = os.path.join(self.mask_dir, mask_name)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')  # Assuming mask is a grayscale image
                    if self.random_ratio:
                        if ratio != "default":
                            mask = crop_to_aspect_ratio(mask, ratio)
                    mask = image_resize(mask, current_size)
                    mask = mask.resize((new_w, new_h))
                    mask = torch.from_numpy(np.array(mask) / 255.0)  # Normalize mask
                    # mask = mask.unsqueeze(0)
                    return img, prompt, mask
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
            else:
                raise NotImplementedError("Require mask but mask_dir is not set or use_mask is False") # 或者根据你的逻辑调整


        except Exception as e:
            print(f"Error loading data: {e}")
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class ParentDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json', random_ratio=False, use_mask=True):
        self.datasets = []
        self.img_dir = img_dir
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.use_mask = use_mask
        
        # Traverse each subdirectory and create a dataset for it
        for sub_dir in os.listdir(img_dir):
            
            sub_dir_path = os.path.join(img_dir, sub_dir)
            # if sub_dir == 'processed_LSVT_train_images':
            #     import pdb; pdb.set_trace()
            if os.path.isdir(sub_dir_path):  # Ensure it's a directory
                dataset = CustomImageDataset(
                    img_dir=sub_dir_path,
                    img_size=img_size,
                    caption_type=caption_type,
                    random_ratio=random_ratio,
                    use_mask=use_mask
                )
                self.datasets.append(dataset)
        
        # Flatten the image list for indexing
        self.dataset_offsets = [0]
        for dataset in self.datasets:
            self.dataset_offsets.append(self.dataset_offsets[-1] + len(dataset))

    def __len__(self):
        return self.dataset_offsets[-1]

    def __getitem__(self, idx):
        # Determine which dataset this index belongs to
        for i in range(len(self.dataset_offsets) - 1):
            if self.dataset_offsets[i] <= idx < self.dataset_offsets[i + 1]:
                local_idx = idx - self.dataset_offsets[i]
                return self.datasets[i][local_idx]
        raise IndexError("Index out of range in ParentDataset")

