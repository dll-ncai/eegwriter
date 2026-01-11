# EEGWriter: A Multimodal Deep Learning Framework for Automated EEG Diagnostic Report Generation
This repository contains the source code for "EEGWriter: A Multimodal Deep Learning Framework for Automated EEG Diagnostic Report Generation". This study compares the performance of different open-source LLMs on EEG Report Generation Task.

## Getting Started
Clone the repository:
```bash
git clone https://github.com/dll-ncai/eegwriter.git
cd eegwriter
```
Install the required dependencies:
```bash
python -m venv env
# On Linux/MacOS:
source env/bin/activate
# On Windows:
# .\env\Scripts\activate
pip install -r requirements.txt
```

## Usage
The models directory contains the PyTorch Models used in this project.
See report_gen.ipynb for all steps of report generation.
The usage of this notebook requires the NMT Scalp EEG Dataset and NMT Events Dataset and the trained weights of the models on these datasets as well as the corresponding reports (Which are not made public due to patient privacy and anonymity protocols).
Also see the following repositories:
- https://github.com/dll-ncai/Localization-of-Abnormalities-in-EEG-Waveforms
- https://github.com/dll-ncai/eeg_pre-diagnostic_screening

For Datasets see:

https://dll.seecs.nust.edu.pk/downloads/

## Project Structure
```text
eegwriter/
├── models
│   ├── __init__.py
│   ├── scnet.py
│   └── vgg16.py
├── pdr.py
├── pipeline
│   ├── __init__.py
│   ├── channels.py
│   └── pipeline.py
├── README.md
├── report_gen.ipynb
└── requirements.txt
```
