# AI-Powered Deepfake & Image Manipulation Detection System

## Hybrid AI + Digital Forensics 

---

# Overview

This project is a hybrid AI-powered digital image forensics framework designed to detect:

* AI-generated images
* Deepfake face manipulations
* Image splicing and tampering
* Compression anomalies and forensic artifacts

The system combines:

* classical forensic signal processing
* machine learning ensemble classifiers
* automated forensic reporting
* evidence visualization
* Streamlit dashboard integration

The framework is designed for:

* cybersecurity
* media authenticity verification
* forensic investigation
* AI-generated content detection
* digital evidence analysis

---

# Hybrid Forensic Architecture

```text
Input Image
    ↓
Feature Extraction
(HOG · LBP · FFT · ELA · Color · DCT · Noise)
    ↓
Compression & ELA Analysis
    ↓
AI-Generated Detection
(GBM + ExtraTrees + SVM)
    ↓
Splicing / Tampering Detection
(GBM + ExtraTrees + SVM)
    ↓
Deepfake Detection
(Custom Ensemble Classifier)
    ↓
Hybrid Score Fusion
(Signal Processing + ML)
    ↓
Evidence Visualization
    ↓
Forensic Report
(JSON · TXT · PDF)
    ↓
Streamlit Dashboard
```

---

# Current Project Structure

```text
deepfake-forensics-tool/
├── app.py
├── dashboard.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── ai_gen_ensemble.pkl
│   ├── ai_gen_scaler.pkl
│   ├── deepfake_model_v2.pkl
│   ├── splicing_ensemble.pkl
│   └── splicing_scaler.pkl
│
├── src/
│   ├── ai_gen_module.py
│   ├── splicing_module.py
│   ├── deepfake_module.py
│   ├── compression_analysis.py
│   ├── feature_extractor.py
│   ├── metadata_analyzer.py
│   ├── score_fusion.py
│   ├── shap_explainer.py
│   ├── pdf_report.py
│   ├── report_generator.py
│   └── __init__.py
│
├── tests/
│   └── test_pipeline.py
│
├── data/
│   ├── CIFAKE/
│   ├── ciplab/
│   ├── splicing/
│   └── test/
│
└── reports/
```

---

# Core Detection Modules

## 1. AI-Generated Image Detection

File:

```text
src/ai_gen_module.py
```

Features:

* FFT spectrum analysis
* GAN artifact detection
* texture fingerprints
* noise inconsistencies
* color distribution analysis

Models:

* GradientBoostingClassifier
* ExtraTreesClassifier
* SVM (RBF)

Saved models:

```text
models/ai_gen_ensemble.pkl
models/ai_gen_scaler.pkl
```

Training Dataset:

* CIFAKE

Performance:

* Accuracy ≈ 91%
* ROC-AUC ≈ 0.975

---

## 2. Splicing / Tampering Detection

File:

```text
src/splicing_module.py
```

Features:

* Error Level Analysis (ELA)
* DCT irregularities
* edge inconsistencies
* lighting inconsistencies
* copy-move artifacts

Models:

* GradientBoostingClassifier
* ExtraTreesClassifier
* SVM (RBF)

Saved models:

```text
models/splicing_ensemble.pkl
models/splicing_scaler.pkl
```

Training Dataset:

* CASIA

Performance:

* Accuracy ≈ 79%
* ROC-AUC ≈ 0.854

---

## 3. Deepfake Detection

File:

```text
src/deepfake_module.py
```

Features:

* HOG facial geometry
* LBP facial texture
* GAN blending artifacts
* noise residual fingerprints
* facial inconsistencies

Models:

* Custom Ensemble Classifier

Saved model:

```text
models/deepfake_model_v2.pkl
```

Training Dataset:

* ciplab / Real and Fake Face Detection

---

# Feature Extraction

File:

```text
src/feature_extractor.py
```

Extracted forensic domains:

* HOG
* LBP
* FFT
* ELA
* DCT
* Color statistics
* Noise analysis
* Texture descriptors

---

# Optional Explainability Layer

File:

```text
src/shap_explainer.py
```

Optional SHAP explainability support is included for future forensic interpretation and feature attribution enhancements.

---

# Forensic Reporting

Files:

```text
src/report_generator.py
src/pdf_report.py
```

Generated outputs:

* JSON reports
* TXT forensic summaries
* PDF forensic evidence reports

Reports include:

* manipulation probability
* confidence score
* forensic evidence
* module breakdown
* chain of custody
* SHA-256 hashing

---

# Streamlit Dashboard

File:

```text
dashboard.py
```

Features:

* image upload
* real-time forensic analysis
* module score visualization
* evidence gallery
* PDF report download
* JSON export
* hybrid forensic verdict system

Run:

```bash
streamlit run dashboard.py
```

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run Full Forensic Pipeline

```bash
python app.py image.jpg
```

Example:

```bash
python app.py tt.jpg
```

---

# Train Models

## AI-Generated Detector

```python
from src.ai_gen_module import train

train(
    data_dir="data/CIFAKE",
    max_per_class=3000
)
```

---

## Splicing Detector

```python
from src.splicing_module import train

train(
    data_dir="data/splicing",
    max_per_class=3000
)
```

---

## Deepfake Detector

```python
from src.deepfake_module import train

train(
    data_root="data/ciplab"
)
```

---

# Datasets

| Dataset | Module                 |
| ------- | ---------------------- |
| CIFAKE  | AI-generated detection |
| CASIA   | Splicing detection     |
| ciplab  | Deepfake detection     |

---

# Tech Stack

* Python
* OpenCV
* NumPy
* scikit-learn
* scikit-image
* Streamlit
* ReportLab
* PIL

---

# System Capabilities

The platform provides:

* Hybrid AI + signal-processing forensic analysis
* Automated manipulation probability estimation
* Real-time forensic dashboard visualization
* Multi-module ensemble detection
* Evidence-oriented forensic reporting
* Digital media authenticity assessment
* Modular and extensible architecture for future forensic modules

---

