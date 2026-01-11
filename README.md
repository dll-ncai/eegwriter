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
source env/bin/activate (On Linux)
pip install -r requirements.txt
```

## Usage
The models directory contains the PyTorch Models used in this project.
See report_gen.ipynb for all the steps.
The usage of this notebook requires the NMT Scalp EEG Dataset and NMT Events Dataset and the trained weights of the models on these datasets.

## Project Structure
eegwriter/
├── models
│   ├── \_\_init\_\_.py
│   ├── scnet.py
│   └── vgg16.py
├── pdr.py
├── pipeline
│   ├── \_\_init\_\_.py
│   ├── channels.py
│   └── pipeline.py
├── README.md
├── report_gen.ipynb
└── requirements.txt
