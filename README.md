# RSHR 🔨

**RSHR: Hierarchical Visual Representation and State-Space Reasoning for Remote Sensing Visual Question Answering**

🚀 This repository provides the official implementation of **RSHR**, a lightweight framework for **Remote Sensing Visual Question Answering (RSVQA)**.  
RSHR addresses two key challenges in RSVQA:  
(1) the visual domain gap between natural images and remote sensing imagery
(2) the inefficiency of Transformer-based reasoning over long multimodal sequences.

---

## Notes 🧩
Upon paper acceptance, the repository will be fully synchronized with the camera-ready version, including detailed instructions, pretrained models, and additional analysis.


## Overview 📊

RSHR consists of two core components:

- **HAVRNet** 🔨  
  A hierarchical visual refinement module that enhances semantic robustness through synergistic spatial and attention modeling.

- **R4SNet** ⚙️  
  A multimodal reasoning module based on **State Space Models (SSMs)**, enabling linear-complexity long-sequence reasoning with reduced computational cost.


---

## Usage 🛠️

```bash
git clone https://github.com/your_username/RSHR.git
cd RSHR
pip install -r requirements.txt


