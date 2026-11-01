# Industrial Multimodal Bearing Fault Detection and Root Cause Analysis System

## Project Overview

This project presents an Industrial Multimodal Bearing Fault Detection and Root Cause Analysis (RCA) System designed to identify bearing faults using vibration signals and thermal images.

The system combines deep learning-based fault classification, multimodal analysis, and Retrieval-Augmented Generation (RAG) to support fault diagnosis and generate explainable root cause analysis reports.

## Key Features

- **Vibration-Based Fault Detection:** Uses a 1D Convolutional Neural Network (CNN) to classify bearing conditions.
- **Thermal Image Analysis:** Uses thermal image processing and deep learning for temperature-based fault analysis.
- **Multimodal Fusion:** Combines information from vibration and thermal modalities.
- **Root Cause Analysis:** Uses retrieval-based reasoning to generate diagnostic insights.
- **RAG Pipeline:** Integrates document retrieval and language-model-based report generation.
- **PDF Reporting:** Generates downloadable diagnostic reports.
- **Interactive Dashboard:** Provides a Streamlit-based user interface.

## System Architecture

The system consists of the following major modules:

1. Vibration Signal Preprocessing
2. Vibration-Based Bearing Fault Classification
3. Thermal Image Processing and Classification
4. Multimodal Feature Fusion
5. Knowledge Retrieval and RAG-Based Analysis
6. Root Cause Analysis and Report Generation

## Technologies Used

- Python
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- OpenCV
- Streamlit
- LangChain
- LangGraph
- ChromaDB
- Hugging Face Transformers
- Cohere
- Ollama
- ReportLab

## Project Structure

```text
bearing-rca-project/
├── src/
├── saved_models/
├── sop_docs/
├── cross_dataset_results/
├── cross_validation_results/
├── paderborn_results/
├── dataset/                 # Not included in GitHub
├── chroma_db/               # Not included in GitHub
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd bearing-rca-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The complete datasets are not included in this repository because of their size.

Place the required vibration and thermal datasets in the appropriate folders under `dataset/` before running the complete training and evaluation pipelines.

## Running the Application

Run the Streamlit application using:

```bash
streamlit run src/app.py
```

## Model Files

Small vibration and thermal CNN model files are included in the repository where applicable.

The large DeBERTa SCADA model weights are excluded from GitHub because of their size and must be configured separately before using the corresponding functionality.

## Research Objective

The primary objective is to develop an intelligent and explainable industrial diagnostic framework that integrates multiple data modalities with retrieval-based reasoning for bearing fault detection and root cause analysis.

## Author

Amisha Kumari

M.Tech – Computer Science and Engineering (AI & DS)
