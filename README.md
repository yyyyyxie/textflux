# TextFlux: An OCR-Free DiT Model for High-Fidelity Multilingual Scene Text Synthesis

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2505.17778">
    <img src='https://img.shields.io/badge/arXiv-2505.17778-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/yyyyyxie/textflux'>
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/yyyyyxie/textflux">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="https://huggingface.co/yyyyyxie/textflux" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://yyyyyxie.github.io/textflux-site/'>
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://modelscope.cn/models/xieyu20001003/textflux">
  <img src="https://img.shields.io/badge/🤖_ModelScope-ckpts-ffbd45.svg" alt="ModelScope">
  </a>
</div>
  <p align="left">
    <strong>English</strong> | <a href="./README_CN.md"><strong>中文简体</strong></a>
  </p>

**TextFlux** is an **OCR-free framework** using a Diffusion Transformer (DiT, based on [FLUX.1-Fill-dev](https://github.com/black-forest-labs/flux)) for high-fidelity multilingual scene text synthesis. It simplifies the learning task by providing direct visual glyph guidance through spatial concatenation of rendered glyphs with the scene image, enabling the model to focus on contextual reasoning and visual fusion.

## Key Features

* **OCR-Free:** Simplified architecture without OCR encoders.
* **High-Fidelity & Contextual Styles:** Precise rendering, stylistically consistent with scenes.
* **Multilingual & Low-Resource:** Strong performance across languages, adapts to new languages with minimal data (e.g., <1,000 samples).
* **Zero-Shot Generalization:** Renders characters unseen during training.
* **Controllable Multi-Line Text:** Flexible multi-line synthesis with line-level control.
* **Data Efficient:** Uses a fraction of data (e.g., ~1%) compared to other methods.

<div align="center">
  <img src="resource/abstract_fig.png" width="100%" height="100%"/>
</div>


## Updates

- **`2025/08/02`**: Our full param [**TextFlux-beta**](https://huggingface.co/yyyyyxie/textflux-beta) weights and [**TextFlux-LoRA-beta**](https://huggingface.co/yyyyyxie/textflux-lora-beta) weights are now available! Single-line text generation accuracy performance could be significantly enhanced by **10.9%** and **11.2%** respectively 👋! 
- **`2025/08/02`**: Our [**Training Datasets**](https://huggingface.co/datasets/yyyyyxie/textflux-anyword) and [**Testing Datasets**](https://huggingface.co/datasets/yyyyyxie/textflux-test-datasets) are now available 👋!
- **`2025/08/01`**: Our [**Eval Scripts**](https://huggingface.co/yyyyyxie/textflux) are now available 👋!
- **`2025/05/27`**: Our [**Full-Param Weights**](https://huggingface.co/yyyyyxie/textflux) and [**LoRA Weights**](https://huggingface.co/yyyyyxie/textflux-lora) are now available 👋!
- **`2025/05/25`**: Our [**Paper on ArXiv**](https://arxiv.org/abs/2505.17778) is available 👋!



## TextFlux-beta  

We are excited to release [**TextFlux-beta**](https://huggingface.co/yyyyyxie/textflux-beta) and [**TextFlux-LoRA-beta**](https://huggingface.co/yyyyyxie/textflux-lora-beta), new versions of our model specifically optimized for single-line text editing.

### Key Advantages

- **Significantly improves the quality** of single-line text rendering.
- **Increases inference speed** for single-line text by approximately **1.4x**.
- **Dramatically enhances the accuracy** of small text synthesis.

### How It Works

Considering that single-line editing is a primary use case for many users and generally yields more stable, high-quality results, we have released new weights optimized for this scenario.

Unlike the original model which renders glyphs onto a full-size mask, the beta version utilizes a **single-line image strip** for the glyph condition. This approach not only reduces unnecessary computational overhead but also provides a more stable and high-quality supervisory signal. This leads directly to the significant improvements in both single-line and small text rendering (see example [here](resource/demo_singleline.png)).


To use these new models, please refer to the updated files: demo.py, run_inference.py, and run_inference_lora.py. While the beta models retain the ability to generate multi-line text, we **highly recommend** using them for single-line tasks to achieve the best performance and stability.

### Performance

This table shows that the TextFlux-beta model achieves a significant performance improvement of approximately **11 points** in single-line text editing, while also boosting inference speed by **1.4 times** compared to previous versions! The [**AMO Sampler**](https://github.com/hxixixh/amo-release) contributed approximately 3 points to this increase. The test dataset is [**ReCTS editing**](https://huggingface.co/datasets/yyyyyxie/textflux-test-datasets). 

| Method             | SeqAcc-Editing (%)↑ | NED (%)↑ |  FID ↓   |  LPIPS ↓  | Inference Speed (s/img)↓ |
| ------------------ | :-----------------: | :------: | :------: | :-------: | :----------------------: |
| TextFlux-LoRA      |        37.2         |   58.2   |   4.93   |   0.063   |           16.8           |
| TextFlux           |        40.6         |   60.7   |   4.84   |   0.062   |           15.6           |
| TextFlux-LoRA-beta |        48.4         |   70.5   |   4.69   |   0.062   |           12.0           |
| TextFlux-beta      |      **51.5**       | **72.9** | **4.59** | **0.061** |         **10.9**         |



## Setup

1.  **Clone/Download:** Get the necessary code and model weights.


2. **Dependencies:**

```bash
git clone https://github.com/yyyyyxie/textflux.git
cd textflux
conda create -n textflux python==3.11.4 -y
conda activate textflux
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
cd diffusers
pip install -e .
# Ensure gradio == 3.50.1
```



## Gradio Demo

Provides "Custom Mode" (upload scene image, draw masks, input text for automatic template generation) and "Normal Mode" (for pre-combined inputs).

```bash
# Ensure gradio == 3.50.1
python demo.py 
```



## Training

To reproduce the results from the paper, you'll need to download the [**Multi-line**](https://huggingface.co/datasets/yyyyyxie/textflux-multi-line) dataset and run the multi-line training command found in `scripts/train.sh`.

```
bash scripts/train.sh
```

or

```
bash scripts/train_lora.sh
```

When training the beta version weights optimized for the single-line task, we fine-tuned for an additional 10,000 steps by loading the weights from the multi-line  [**TextFlux**](https://huggingface.co/yyyyyxie/textflux) and [**TextFLux-LoRA**](https://huggingface.co/yyyyyxie/textflux-lora)  models. You'll need to download the [**Single-line**](https://huggingface.co/datasets/yyyyyxie/textflux-anyword) dataset and run the single-line training command in `scripts/train.sh`.

```
bash scripts/train.sh
```

or

```
bash scripts/train_lora.sh
```



## Evaluation

First, use the `scripts/batch_eval.sh` script to perform batch inference on the images in the test set.

```
bash scripts/batch_eval.sh
```

Once inference is complete, use `eval/eval_ocr.sh` to evaluate the OCR accuracy and `eval/eval_fid_lpips.sh` to evaluate FID and LPIPS scores.

```
bash eval/eval_ocr.sh
```

```
bash eval/eval_fid_lpips.sh
```



## TODO

- [x] Release the training datasets and testing datasets
- [x] Release the training scripts
- [x] Release the eval scripts
- [ ] Support comfyui



## Acknowledgement

Our code is modified based on [Diffusers](https://github.com/huggingface/diffusers). We adopt [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) as the base model. Thanks to all the contributors for the helpful discussions! We also sincerely thank the contributors of the following code repositories for their valuable contributions: [AnyText](https://github.com/tyxsspa/AnyText), [AMO](https://github.com/hxixixh/amo-release).



## Citation

```bibtex
@misc{xie2025textfluxocrfreeditmodel,
      title={TextFlux: An OCR-Free DiT Model for High-Fidelity Multilingual Scene Text Synthesis}, 
      author={Yu Xie and Jielei Zhang and Pengyu Chen and Ziyue Wang and Weihang Wang and Longwen Gao and Peiyi Li and Huyang Sun and Qiang Zhang and Qian Qiao and Jiaqing Fan and Zhouhui Lian},
      year={2025},
      eprint={2505.17778},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.17778}, 
}
```
