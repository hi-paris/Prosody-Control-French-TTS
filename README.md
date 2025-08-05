# Improving Synthetic Speech Quality via SSML Prosody Control

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Conda environment](https://img.shields.io/badge/conda-env-green.svg)](https://docs.conda.io/)

---

This repository contains the code and models for the paper:

**"Improving Synthetic Speech Quality via SSML Prosody Control"**

We present a novel, end-to-end pipeline for enhancing the prosody of French synthetic speech using SSML (Speech Synthesis Markup Language) tags. Our approach leverages both supervised and large language model (LLM) methods to automatically annotate text with prosodic cues (pitch, volume, rate, and pauses), significantly improving the naturalness and expressiveness of TTS output.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Despite advances in TTS, synthetic French voices often lack natural prosody, especially in expressive contexts. This project provides:

- An **SSML annotation pipeline** (`audioPipeline.py`) for French speech, aligning audio and text, extracting prosodic features, and generating SSML markup.
- **Baseline models** (BERT, BiLSTM) for prosody and break prediction.
- **LLM-based models** (zero-shot, few-shot, and our novel cascaded Qwen approach) for automatic SSML tag generation.
- Example data and configuration for reproducible experiments.

---

## Installation

We recommend using **Ubuntu 22.04.3** or similar for best compatibility.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NassimaOULDOUALI/Prosody-Control-French-TTS
 
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
│   ├── audioPipeline.py           # Main SSML annotation pipeline
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

## Usage

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

## Models

- **Baselines**: See `Code/baseline_models/` for BERT and BiLSTM models for pause and prosody prediction.
- **LLM Approaches**: See `Code/ssml_models/` for zero-shot, few-shot, and cascaded Qwen-based models for SSML tag generation.

All models and scripts are referenced in the paper and can be used or extended for further research.

---

## Citation

If you use this code or models in your research, please cite:

```bibtex
@inproceedings{ould-ouali2025_improving,
  title     = {Improving Synthetic Speech Quality via SSML Prosody Control},
  author    = {Ould-Ouali, Nassima and Sani, Awais and Bueno, Ruben and Dauvet, Jonah and Horstmann, Tim Luka and Moulines, Eric},
  booktitle = {Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP)}, % TODO: vérifier l'intitulé exact utilisé par la conf
  year      = {2025},
  pages     = {XX--YY},   % TODO
  publisher = {—},        % TODO 
  address   = {—}         % TODO
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact:

- Nassima Ould-Ouali: nassima.ould-ouali@polytechnique.edu
- Awais Sani : awais.sani@ip-paris.fr
- Ruben Bueno : ruben.bueno@polytechnique.edu
- Jonah Dauvet: jonah.dauvet@mail.mcgill.ca
- Tim Luka Horstmann: tim.horstmann@ip-paris.fr
- Eric Moulines : eric.moulines@polytechnique.edu
---
