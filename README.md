# RSHR+: Remote Sensing Hierarchical Reasoning Framework for Visual Question Answering

<p align="center">
  <img src="https://img.shields.io/badge/Status-Under%20Review-yellow" />
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-blue" />
  <img src="https://img.shields.io/badge/Task-RSVQA-green" />
</p>

> 🚧 **Note:** This repository is currently under review. The full code will be released upon acceptance.  
> 🚧 **注意：** 本仓库对应论文正在审稿中，代码将在录用后公开。

---

## 📖 Introduction / 简介

**English:**  
Vision-Language Models pre-trained on natural images still face significant challenges in **Remote Sensing Visual Question Answering (RSVQA)**, due to (1) the vast domain discrepancy between natural and remote sensing imagery, and (2) the long-sequence reasoning bottleneck caused by the fusion of high-resolution visual features and text.

We propose **RSHR+**, a lightweight end-to-end framework that addresses both challenges:
- A **Progressive Question-Conditioned Visual Representation Calibration Mechanism** composed of:
  - **GIBSR** (Gradient-Isolated Bidirectional Synergistic Representation Network): decouples local convolution and global attention into heterogeneous pathways with bidirectional mutual guidance and backward gradient isolation for domain alignment.
  - **QRTS** (Query-Relevance Token Selection): models question semantics via learnable query prototypes and dynamic Top-k sparse routing to generate soft prompts recalibrating visual tokens.
- **R4SNet** (Remote Sensing Semantic State Space Network): leverages State Space Models (SSMs) with near-linear complexity, enhanced by a dynamic Element-wise Affine Transformation FFN (EAT-FFN) for efficient multimodal long-range dependency modeling.

---


> RSHR+ achieves state-of-the-art performance **without** relying on large-scale remote sensing pre-training.  


---

## 🔧 Installation / 安装

> Code will be released upon paper acceptance. The following is a preview of the installation steps.  
> 代码将在论文录用后公开，以下为预期安装方式。

```bash
# Clone the repository
git clone https://github.com/your-username/RSHR-plus.git
cd RSHR-plus

# Create conda environment
conda create -n rshr python=3.10 -y
conda activate rshr

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies / 主要依赖：**
- Python >= 3.10
- PyTorch >= 2.0
- transformers
- timm

---

## 🗂️ Dataset / 数据集

| Dataset | Description |
|---------|-------------|
| [RSVQA-LR](https://rsvqa.sylvainlobry.com/) | Low-resolution remote sensing VQA |
| [RSVQA-HR](https://rsvqa.sylvainlobry.com/) | High-resolution remote sensing VQA |
| [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) | Flood scene understanding VQA |

---

## 📬 Citation / 引用

If you find this work useful, please cite our paper:  
如果本工作对您有帮助，请引用我们的论文：

```bibtex
@article{rshrplus2025,
  title     = {RSHR+: A Lightweight Framework for Remote Sensing Visual Question Answering via Progressive Visual Calibration and State Space Modeling},
  author    = {Your Name and Co-authors},
  journal   = {Under Review},
  year      = {2025}
}
```

> ⚠️ BibTeX will be updated with the official venue info upon acceptance.  
> ⚠️ 录用后将更新正式会议/期刊引用信息。

---

## 📄 License / 许可证

This project will be released under the [MIT License](LICENSE) upon acceptance.  
本项目将在录用后以 MIT 协议开源。
