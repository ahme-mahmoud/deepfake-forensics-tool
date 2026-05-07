# AI-Powered Deepfake & Image Manipulation Detection System
## Hybrid AI + Digital Forensics Platform  

---

## Architecture

```
Input Image
    ↓
Feature Extraction          (HOG · LBP · FFT · ELA · Color · DCT · Noise → 1983 dims)
    ↓
Signal-Processing Forensics (ELA · Splicing · AI-Gen · Deepfake heuristics)
    ↓
ML Classifiers              (SVM + Random Forest × 3 modules)
    ↓
Score Fusion                (Signal × 40% + ML × 60% per module)
    ↓
SHAP Explainability         (WHY was it flagged?)
    ↓
Evidence Visualization      (ELA panels · Heatmaps · FFT spectrum)
    ↓
Forensic Report             (JSON + TXT + PDF with chain-of-custody)
    ↓
Streamlit Dashboard
```

---

## Project Structure

```
deepfake-forensics-tool/
├── app.py                          # Master pipeline (7 steps)
├── dashboard.py                    # Streamlit dashboard
├── requirements.txt
├── README.md
│
├── src/
│   ├── feature_extractor.py        # HOG, LBP, FFT, ELA, Color, DCT, Noise
│   ├── ml_classifier.py            # SVM + Random Forest per module
│   ├── train_models.py             # Training pipeline + evaluation
│   ├── shap_explainer.py           # SHAP feature importance
│   ├── score_fusion.py             # Hybrid score combination
│   ├── pdf_report.py               # PDF forensic report (ReportLab)
│   │
│   ├── compression_analysis.py     # Module 1 — ELA
│   ├── splicing_detector.py        # Module 2 — Splicing
│   ├── ai_generated_detector.py    # Module 3 — AI Detection
│   ├── deepfake_detector.py        # Module 4 — Deepfake
│   └── report_generator.py         # Module 5 — Text/JSON Report
│
├── tests/
│   └── test_pipeline.py
│
├── data/
│   ├── real/    ← place real images here
│   ├── fake/    ← place manipulated/fake images here
│   └── test/
│
└── models/      ← trained .pkl files saved here
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train ML models (auto if not present)
python src/train_models.py --samples 500

# Analyse an image
python app.py path/to/image.jpg --case-id CASE-001

# Launch dashboard
streamlit run dashboard.py
```

---

## ML Models

| Module | Algorithm | Features Used |
|--------|-----------|--------------|
| Splicing | SVM + RF | HOG, LBP, ELA, DCT, Color |
| AI-Generated | SVM + RF | FFT, Noise, Texture, Color |
| Deepfake | SVM + RF | HOG, LBP, Color, Noise |

Score Fusion: `hybrid = 0.60 × ML_score + 0.40 × signal_score`

---

## Datasets (for real training)

| Dataset | Module | Link |
|---------|--------|------|
| **CIFAKE** | AI-Generated | kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images |
| **FaceForensics++** | Deepfake | github.com/ondyari/FaceForensics |
| **Celeb-DF** | Deepfake | github.com/yuezunli/celeb-deepfakeforensics |
| **CASIA TIDE** | Splicing | forensics.idealtest.org |

Place images in `data/real/` and `data/fake/` then run `python src/train_models.py`

---

## Feature Vector (1983 dims)

| Feature | Dims | Captures |
|---------|------|---------|
| HOG | 1764 | Edge/gradient structure |
| LBP | 64 | Micro-texture patterns |
| FFT | 34 | Frequency spectrum (GAN artifacts) |
| ELA | 5 | Compression history mismatch |
| Color | 108 | Colour distribution anomalies |
| DCT | 4 | JPEG block irregularities |
| Noise | 4 | Sensor noise profile |

---

## Tech Stack

- **Python** — core language
- **OpenCV + scikit-image** — image processing
- **scikit-learn** — SVM + Random Forest
- **SHAP** — explainability
- **ReportLab** — PDF reports
- **Streamlit** — dashboard

