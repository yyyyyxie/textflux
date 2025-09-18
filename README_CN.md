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
    <strong>中文简体</strong> | <a href="./README.md"><strong>English</strong></a>
  </p>

**TextFlux** 是一个**OCR-free的框架**，它使用 Diffusion Transformer (DiT，基于 [FLUX.1-Fill-dev](https://github.com/black-forest-labs/flux)) 来实现高保真的多语言场景文本合成。它通过将渲染的字形与场景图像进行空间拼接，为模型提供直接的视觉字形指导，从而简化了学习任务，使模型能够专注于上下文推理和视觉融合。

## 主要特性

* **OCR-free：** 简化的架构，无需OCR编码器。
* **高保真且上下文风格一致：** 精确渲染，与场景风格一致。
* **多语言和低资源：** 在各种语言中表现出色，仅需少量数据（例如，少于1000张图片）即可适应新语言。
* **零样本泛化：** 能够渲染训练期间未见过的字符。
* **可控多行文本：** 灵活的多行文本合成，具有行级控制能力。
* **数据高效：** 与其他方法相比，仅使用一小部分数据（例如，约1%）。

<div align="center">
  <img src="resource/abstract_fig.png" width="100%" height="100%"/>
</div>
## 最新动态

-   **`2025/09/18`**: [**TextFlux的ComfyUI**](https://github.com/yyyyyxie/textflux_comfyui)脚本可获得，简单易用，几乎无需更改现有的Flux本身工作流。
-   **`2025/08/02`**: 我们的全参数  [**TextFlux-beta**](https://huggingface.co/yyyyyxie/textflux-beta) 权重和 [**TextFlux-LoRA-beta**](https://huggingface.co/yyyyyxie/textflux-lora-beta) 权重现已发布！单行文本生成准确率分别显著提升了 **10.9% 和 11.2%** 👋！
-   **`2025/08/02`**: [**训练集**](https://huggingface.co/datasets/yyyyyxie/textflux-anyword) 和 [**测试集**](https://huggingface.co/datasets/yyyyyxie/textflux-test-datasets) 现已可获得👋!
-   **`2025/08/01`**: 我们的 [**评估脚本**](https://huggingface.co/yyyyyxie/textflux) 现已可获得 👋!
-   **`2025/05/27`**: 我们的 [**全参数权重**](https://huggingface.co/yyyyyxie/textflux) 和 [**LoRA 权重**](https://huggingface.co/yyyyyxie/textflux-lora) 现已发布 🤗！
-   **`2025/05/25`**: 我们的 [**论文已在 ArXiv 上发布**](https://arxiv.org/abs/2505.17778) 🥳！



## TextFlux-beta 版本

我们发布了 [**TextFlux-beta**](https://huggingface.co/yyyyyxie/textflux-beta) 和 [**TextFlux-LoRA-beta**](https://huggingface.co/yyyyyxie/textflux-lora-beta)， 这是我们专为单行文本编辑任务优化的新版本模型。

### 核心优势

- **显著提升**单行文本的渲染质量。
- 将单行文本的**推理速度提升**约 **1.4 倍**。
- **大幅增强**小尺寸文本的合成准确率。

### 实现原理

我们考虑到单行编辑是许多用户的核心应用场景，并且通常能产生更稳定、更高质量的结果。为此，我们专门发布了针对该场景优化的新权重。

与原版模型将字形渲染在与原图相同大小的掩码上不同，beta版本采用了更高效的**紧凑单行图像条 (compact, single-line image strip)** 作为字形条件。这种方法不仅节省了不必要的计算开销，还能提供一个更稳定、更高质量的监督信号，从而直接带来了在单行文本和小文本渲染上的显著性能提升。类似于[这个样本](resource/demo_singleline.png):


请参考更新后的 demo.py、run_inference.py 和 run_inference_lora.py 文件来使用新模型。尽管beta版本保留了生成多行文本的能力，我们**强烈推荐**在单行任务中使用它，以获得最佳的性能和稳定性。

### 性能表现

下表展示了 TextFlux-beta 模型在单行文本编辑任务上取得了卓越的 **~11 个点的SeqAcc提升**，同时将**推理速度提升了1.4倍以上**。评估在ReCTS编辑测试集上进行。

| 方法               | SeqAcc-Editing (%)↑ | NED (%)↑ | FID ↓ | LPIPS ↓ | 推理速度 (s/img)↓ |
| ------------------ | ------------------- | -------- | ----- | ------- | ----------------- |
| TextFlux-LoRA      | 37.2                | 58.2     | 4.93  | 0.063   | 16.8              |
| TextFlux           | 40.6                | 60.7     | 4.84  | 0.062   | 15.6              |
| TextFlux-LoRA-beta | 48.4                | 70.5     | 4.69  | 0.062   | 12.0              |
| TextFlux-beta      | **51.5**            | **72.9** | 4.59  | 0.061   | **10.9**          |



## 安装

1.  **克隆/下载：** 获取必要的代码和模型权重。
2.  **依赖项：**
    ```bash
    conda create -n textflux python==3.11.4 -y
    conda activate textflux
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install -r requirements.txt
    cd diffusers
    pip install -e .
    # 确保 gradio == 3.50.1
    ```

## Gradio 演示

提供“普通模式”（用于预组合的输入）和“自定义模式”（上传场景图片、自行手动绘制掩码、输入文本以自动生成模板）。

```bash
python demo.py
```




## 模型训练

本指南提供了训练与微调 **TextFlux** 模型的操作说明。

-----

### 多行文本训练 (复现论文结果)

请按照以下步骤复现原始论文中的多行文本生成结果。

1.  **准备数据集**
    下载 [**Multi-line**](https://huggingface.co/datasets/yyyyyxie/textflux-multi-line) 数据集，并按照以下目录结构进行组织：

    ```
    |- ./datasets
       |- multi-lingual
       |  |- processed_mlt2017
       |  |- processed_ReCTS_train_images
       |  |- processed_totaltext
       |  ....
    ```

2.  **运行训练脚本**
    执行相应的训练脚本。`train.sh` 用于标准训练，而 `train_lora.sh` 用于 LoRA 训练。

    ```bash
    # 标准训练
    bash scripts/train.sh
    ```

    或

    ```bash
    # LoRA 训练
    bash scripts/train_lora.sh
    ```

    *注意：请确保您使用的是脚本中为**多行**训练指定的命令和配置。*

-----

### 单行文本训练 (微调)

为了创建针对单行任务优化的 TextFlux Beta 版权重，我们对预训练的多行模型进行了微调。具体来说，我们加载了 [**TextFlux**](https://huggingface.co/yyyyyxie/textflux) 和 [**TextFLux-LoRA**](https://huggingface.co/yyyyyxie/textflux-lora) 模型的权重，并在单行数据集上额外进行了 10,000 步的训练。

如果您希望复现此过程，可以按照以下步骤操作：

1.  **准备数据集**
    首先，下载 [**Single-line**](https://huggingface.co/datasets/yyyyyxie/textflux-anyword) 数据集，并按如下方式组织：

    ```
    |- ./datasets
       |- anyword
       |  |- ReCTS
       |  |- TotalText
       |  |- ArT
       |  ...
       ....
    ```

2.  **运行微调脚本**
    请确保您的脚本已配置为加载多行模型的预训练权重，然后执行微调命令。

    ```bash
    # 标准微调
    bash scripts/train.sh
    ```

    或

    ```bash
    # LoRA 微调
    bash scripts/train_lora.sh
    ```




## 评估

首先，使用 scripts/batch_eval.sh 脚本批量推理测试集中的图片。

```
bash scripts/batch_eval.sh
```

推理完成后，使用 eval/eval_ocr.sh 评估 OCR 准确度，并使用 eval/eval_fid_lpips.sh 评估 FID 和 LPIPS 指标。

```
bash eval/eval_ocr.sh
```

```
bash eval/eval_fid_lpips.sh
```



## TODO

- [x] 发布训练数据集和测试数据集
- [x] 发布训练脚本
- [x] 发布评估脚本
- [x] 支持 ComfyUI

## 致谢

我们的代码基于 [Diffusers](https://github.com/huggingface/diffusers) 修改。我们采用 [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) 作为基础模型。感谢所有贡献者参与的讨论！同样真挚地感谢以下代码仓： [AnyText](https://github.com/tyxsspa/AnyText), [AMO](https://github.com/hxixixh/amo-release)。


## 引用

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
