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
  <a href="https://huggingface.co/yyyyyxie/textflux">
  <img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace">
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

- **`2025/05/27`**: Our [**Full-Param Weights**](https://huggingface.co/yyyyyxie/textflux) and [**LoRA Weights**](https://huggingface.co/yyyyyxie/textflux-lora) are now available 🤗!
- **`2025/05/25`**: Our [**Paper on ArXiv**](https://arxiv.org/abs/2505.17778) is available 🥳!

## Setup

1.  **Clone/Download:** Get the necessary code and model weights.


2. **Dependencies:**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
# Ensure diffusers >= 0.32.1
```



## Gradio Demo

Provides "Normal Mode" (for pre-combined inputs) and "Custom Mode" (upload scene, draw masks, input text for automatic template generation and concatenation).
```bash
python demo.py
```

## TODO

- [ ] Release the training datasets and testing datasets
- [ ] Release the training scripts
- [ ] Release the eval scripts
- [ ] Support comfyui

## Acknowledgement

Our code is modified based on [Diffusers](https://github.com/huggingface/diffusers). We adopt [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) as the base model. Thanks to all the contributors for the helpful discussions!

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
