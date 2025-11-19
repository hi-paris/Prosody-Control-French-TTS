# Improving Synthetic Speech Quality via SSML Prosody Control

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Conda environment](https://img.shields.io/badge/conda-env-green.svg)](https://docs.conda.io/)
[![Paper](https://img.shields.io/badge/📄-Paper-important)](https://aclanthology.org/2025.icnlsp-1.30/)
[![Demo](https://img.shields.io/badge/🎧-Live_Demo-success)](https://hi-paris.github.io/DemoTTS/)
[![HF Models](https://img.shields.io/badge/🤗-HuggingFace_Models-yellow)](https://huggingface.co/hi-paris/ssml-text2breaks-fr-lora)
---

## 📝 Abstract

This repository contains the code and models for the paper:

**"Improving Synthetic Speech Quality via SSML Prosody Control"**

We present a novel, end-to-end pipeline for enhancing the prosody of French synthetic speech using SSML (Speech Synthesis Markup Language) tags. Our approach leverages both supervised and large language model (LLM) methods to automatically annotate text with prosodic cues (pitch, volume, rate, and pauses), significantly improving the naturalness and expressiveness of TTS output.


## 🚀 Quick Links
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K3bcLHRfbSy9syWRZR6D0hyTb5lqivGi)
[![Demo](https://img.shields.io/badge/🎧-Live_Demo-success)](https://hi-paris.github.io/DemoTTS/)

---

## Table of Contents

- [🌟 Overview](#-overview)
- [⚡ Installation](#-installation)
- [🏗️ Project Structure](#️-project-structure)
- [🎮 Usage](#-usage)
- [🤖 Models](#-models)
- [📊 Demo](#-demo)
- [📚 Citation](#-citation)
- [📄 License](#-license)
- [📬 Contact](#-contact)
---

## Overview

Despite advances in TTS, synthetic French voices often lack natural prosody, especially in expressive contexts. This project provides:

- 🎵 **SSML Annotation Pipeline** (`audioPipeline.py`) for French speech
- 📊 **Baseline Models** (BERT, BiLSTM) for prosody and break prediction  
- 🧠 **LLM-based Models** (zero-shot, few-shot, and cascaded Qwen)
- 📁 Example data and configuration for reproducible experiments


---

## ⚡Installation

We recommend using **Ubuntu 22.04.3** or similar for best compatibility.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hi-paris/Prosody-Control-French-TTS
 
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f tts-env.yml
   conda activate tts-env
   ```

3. **Download required tools:**
   - Download the `.rar` archive from [Google Drive](https://drive.google.com/file/d/1UR22BRf_IQhjQ6yPPhM1aeoxJeF1Obe2/view?usp=sharing)
   - Place it in a folder named `Tools` at the root of the repository (`prosodyControl/Tools/`)

4. **Add your Azure TTS API key:**
   -  at the root of the repository
   - Paste your Azure API key into this file

---

## Project Structure

```
prosodyControl/
│
├── Code/
│   ├── audioPipeline.py           # Main SSML pipeline
│   ├── audioPipeline_legacy.py    # Legacy pipeline scripts
│   ├── pipeline_class_legacy.py   # Legacy pipeline class
│   ├── prepare_AB_test.py         # AB test preparation script
│   ├── Aligners/                  # Alignment tools (Whisper, MFA, etc.)
│   ├── Pipeline/                  # Prosody extraction and processing modules
│   ├── Preprocessing/             # Audio and data preprocessing scripts
│   ├── baseline_models/           # Baseline BERT and BiLSTM models
│   └── ssml_models/               # Zero-shot, few-shot, and cascaded LLM models
│
├── Data/
│   └── voice/
│       └── records/
│           └── audio/             # Example segmented audio files
│
├── config.yaml                    # Main configuration file for the pipeline
├── tts-env.yml                    # Conda environment specification
├── Azure_API_key.txt              # Use environment variables instead
├── README.md                      # This file
```

- **`Code/audioPipeline.py`**: The main entry point for the SSML annotation pipeline. All processing steps are managed here.
- **`Code/Aligners/`**, **`Code/Pipeline/`**, **`Code/Preprocessing/`**: Contain scripts for alignment, prosody extraction, and preprocessing, used as part of the pipeline.
- **`Code/baseline_models/`**: Implements the BERT and BiLSTM baselines referenced in the paper.
- **`Code/ssml_models/`**: Contains our zero-shot, few-shot, and cascaded LLM approaches for SSML tag prediction.
- **`Data/voice/records/audio/`**: Example segmented audio files for demonstration and testing.

---

##  🎮 Usage

All pipeline settings are controlled via [`config.yaml`](config.yaml). This includes data paths, voice names, Azure TTS settings, prosody parameters, and which steps to run.

**To run the full SSML annotation pipeline:**

```bash
conda activate tts-env
python Code/audioPipeline.py
```

- Adjust `config.yaml` as needed for your data and experiment.
- The pipeline will process all voices specified in `voice_names` and execute the steps listed in `steps_to_run`.
- Intermediate and final outputs (e.g., SSML, audio, CSVs) will be saved according to your configuration.

---

## 🤖 Models

- **Baselines**: See `Code/baseline_models/` for BERT and BiLSTM models for pause and prosody prediction.
- **LLM Approaches**: See `Code/ssml_models/` for zero-shot, few-shot, and cascaded Qwen-based models for SSML tag generation.

All models and scripts are referenced in the paper and can be used or extended for further research.

---

## 📚 Citation

Paper is available :

[Improving French Synthetic Speech Quality via SSML Prosody Control](https://aclanthology.org/2025.icnlsp-1.30/)

If you use this model, please cite the paper.

```
@inproceedings{ouali-etal-2025-improving,
    title = "Improving {F}rench Synthetic Speech Quality via {SSML} Prosody Control",
    author = "Ouali, Nassima Ould  and
      Sani, Awais Hussain  and
      Bueno, Ruben  and
      Dauvet, Jonah  and
      Horstmann, Tim Luka  and
      Moulines, Eric",
    editor = "Abbas, Mourad  and
      Yousef, Tariq  and
      Galke, Lukas",
    booktitle = "Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP-2025)",
    month = aug,
    year = "2025",
    address = "Southern Denmark University, Odense, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.icnlsp-1.30/",
    pages = "302--314"
}

```
<div align="center">
   
⭐ Don't forget to star this repo if you find it useful!

</div> ```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---


## 📬 Contact

**Nassima Ould-Ouali**  
[![Email](https://img.shields.io/badge/Email-nassima.ould--ouali%40polytechnique.edu-blue?style=flat-square&logo=gmail)](mailto:nassima.ould-ouali@polytechnique.edu)
---
