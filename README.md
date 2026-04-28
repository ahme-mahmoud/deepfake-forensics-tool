# AI-Powered Deepfake & Image Manipulation Detection Tool

> **A complete digital forensics investigation system for detecting deepfakes, image splicing, and AI-generated imagery.**

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Module Architecture](#module-architecture)
5. [Forensic Methodology](#forensic-methodology)
6. [Scoring System](#scoring-system)
7. [Sample Output](#sample-output)
8. [Running Tests](#running-tests)
9. [Extending the Tool](#extending-the-tool)
10. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This tool implements a **multi-module forensic pipeline** that analyses digital images for signs of manipulation using signal-processing, computer vision, and statistical techniques — without requiring pre-trained deep-learning models for its core functionality.

| Module | Technique | What it detects |
|--------|-----------|-----------------|
| Compression Analysis | Error Level Analysis (ELA) | JPEG re-save inconsistencies |
| Splicing Detection | Edge + Lighting + Copy-Move | Pasted regions from other images |
| AI-Generation Detection | FFT + Noise + Texture | GAN/diffusion-model outputs |
| Deepfake Detection | Landmark + Blend + Colour | Face-swap deepfakes |

---

## Project Structure

```
deepfake-forensics-tool/
│
├── app.py                        # Master pipeline & CLI entry point
├── requirements.txt
├── README.md
│
├── src/
│   ├── compression_analysis.py   # Module 1 — ELA
│   ├── splicing_detector.py      # Module 2 — Splicing
│   ├── ai_generated_detector.py  # Module 3 — AI Detection
│   ├── deepfake_detector.py      # Module 4 — Deepfake
│   └── report_generator.py       # Module 5 — Report
│
├── tests/
│   └── test_pipeline.py          # Full automated test suite
│
├── data/
│   ├── real/                     # Real/authentic images
│   ├── fake/                     # Known deepfakes / AI-generated
│   └── test/                     # Test images for demo
│
├── models/                       # Pre-trained model weights (optional)
│
└── reports/
    └── evidence/                 # Auto-generated ELA / annotation images
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-forensics-tool.git
cd deepfake-forensics-tool
pip install -r requirements.txt
```

### 2. Analyse an image

```bash
python app.py data/test/sample.jpg --case-id CASE-001
```

### 3. View the report

Reports are saved in `reports/`:
- `reports/forensic_report_CASE-001.txt`   — human-readable
- `reports/forensic_report_CASE-001.json`  — machine-readable

Evidence images are saved in `reports/evidence/`:
- `_ela.jpg`                — ELA heat-map
- `_panel.jpg`              — Side-by-side forensic panel
- `_edge_heatmap.jpg`       — Edge density map
- `_copymove.jpg`           — Copy-move annotated image
- `_fft_spectrum.jpg`       — FFT frequency spectrum
- `_noise_residual.jpg`     — Noise residual map
- `_deepfake_annotated.jpg` — Face detection overlay

---

## Module Architecture

### Module 1 — Compression Analysis (ELA)

```
Input Image
    │
    ├─► Re-save at JPEG quality 95
    │       │
    │       └─► Pixel-wise absolute difference → Amplify (×15)
    │                   │
    │                   ├─► ELA heat-map image (saved as evidence)
    │                   └─► Score = f(mean, std-dev, high-ratio)
    │
    └─► Suspicious region detection (Otsu + contours)
```

**Forensic principle:** Every time a JPEG is saved, data is lost due to lossy compression. A pasted region has a *different compression history* from the original — ELA makes this visible.

---

### Module 2 — Splicing Detection

Three independent sub-detectors:

1. **Edge Inconsistency** — Block-level Canny edge density; outlier blocks flagged.
2. **Lighting Inconsistency** — HOG-style gradient orientation histograms per block; chi-squared divergence from global histogram.
3. **Copy-Move Detection** — DCT feature hashing of overlapping blocks; lexicographic sort for near-duplicate detection.

---

### Module 3 — AI-Generated Detection

Exploits three GAN/diffusion fingerprints:

1. **Frequency Artifacts** — FFT spectrum analysis: periodic GAN checkerboard peaks + radial decay slope.
2. **Noise Residual** — Gaussian residual analysis: std, autocorrelation, entropy.
3. **Texture Regularity** — GLCM entropy: AI images are statistically "too perfect".

---

### Module 4 — Deepfake Detection

Four facial forensics techniques:

1. **Landmark Geometry** — Face/eye aspect ratios, eye count anomalies (Haar cascade).
2. **Blending Boundary** — Gradient ring analysis at the face perimeter ellipse.
3. **Colour Inconsistency** — YCbCr Cb/Cr spatial std-dev + histogram entropy inside face ROI.
4. **Eye Glint Asymmetry** — Top-5% brightness asymmetry between left/right eyes.

---

## Forensic Methodology

### Chain of Custody
Every analysis begins by computing the **SHA-256 hash** of the input file. This hash is embedded in:
- All evidence filenames (first 12 hex chars)
- The forensic report header
- The JSON output

This ensures the evidence can always be traced back to the original file.

### Analysis Log
All module operations are logged with timestamps using Python's `logging` module. The full log is available at `INFO` level.

### Score Derivation
All sub-detector scores are derived from first-principles signal-processing operations — not black-box ML predictions. Each score has an accompanying `interpretation` string explaining the forensic finding in plain English.

---

## Scoring System

| Score Range | Verdict |
|-------------|---------|
| 0.00 – 0.19 | **AUTHENTIC** — Very low suspicion |
| 0.20 – 0.39 | **LIKELY AUTHENTIC** — Low suspicion |
| 0.40 – 0.59 | **INCONCLUSIVE** — Moderate suspicion |
| 0.60 – 0.79 | **LIKELY MANIPULATED** — High suspicion |
| 0.80 – 1.00 | **MANIPULATED** — Very high suspicion |

**Final manipulation probability** = mean of all four module scores.

**Module weights (internal):**

| Module | Weight within module |
|--------|---------------------|
| ELA | mean (0.25) + std (0.50) + ratio (0.25) |
| Splicing | edge (0.40) + lighting (0.35) + copy-move (0.25) |
| AI-Gen | frequency (0.45) + noise (0.30) + texture (0.25) |
| Deepfake | landmark (0.30) + blend (0.25) + colour (0.20) + glint (0.15) + embedding (0.10) |

---

## Sample Output

```
══════════════════════════════════════════════════════════════════════════
    AI-Powered Deepfake & Image Manipulation Detection Tool  v1.0.0
══════════════════════════════════════════════════════════════════════════

  ► 1. Compression / ELA
    Score  : [████████████░░░░░░░░░░░░░░░░░░] 0.42
    Verdict: INCONCLUSIVE (moderate suspicion)
    Analysis: Significant ELA anomalies detected...

  ► 2. Splicing Detection
    Score  : [████████████████░░░░░░░░░░░░░░] 0.54
    Verdict: INCONCLUSIVE (moderate suspicion)
    
  MANIPULATION PROBABILITY: [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.35
  ASSESSMENT: LIKELY AUTHENTIC (low suspicion)
```

---

## Running Tests

```bash
# Run all tests
python tests/test_pipeline.py

# Or with pytest
pip install pytest
pytest tests/ -v
```

**Test coverage:**
- ✅ SHA-256 determinism
- ✅ ELA on clean vs spliced images
- ✅ Evidence file creation
- ✅ All module result structures
- ✅ Report generation (JSON + TXT)
- ✅ End-to-end pipeline integration

---

## CLI Reference

```
python app.py <image_path> [options]

Options:
  --quality INT      ELA JPEG re-save quality (default: 95)
  --scale   INT      ELA amplification factor (default: 15)
  --out-dir PATH     Report output directory (default: reports/)
  --no-save          Skip saving evidence images
  --json-only        Print JSON to stdout
  --case-id STR      Case reference number
  --investigator STR Investigator ID
```

**Exit codes:**
- `0` — Image appears authentic (probability < 0.55)
- `1` — Image likely manipulated (probability ≥ 0.55)

---

## Extending the Tool

### Add FaceNet embeddings
```python
# In src/deepfake_detector.py, FaceEmbeddingChecker._embed():
from facenet_pytorch import InceptionResnetV1
model = InceptionResnetV1(pretrained='vggface2').eval()
```

### Add a CNN deepfake classifier
Place your `.pt` or `.h5` model in `models/` and call it from `deepfake_detector.analyze()`.

### Add Streamlit dashboard
```bash
pip install streamlit
streamlit run dashboard.py
```

### Add SHAP explainability
```bash
pip install shap
# Apply SHAP to any sklearn-compatible model wrapping the module scores
```

---

## Limitations & Future Work

| Limitation | Future Solution |
|------------|-----------------|
| No CNN model (core is signal-processing only) | Fine-tune EfficientNet on FaceForensics++ |
| Haar cascade face detector is basic | Replace with MTCNN / RetinaFace |
| FaceNet stub returns 0.5 | Integrate facenet-pytorch |
| No video support | Add frame-by-frame analysis |
| No reverse image search | Integrate TinEye / Google Lens API |
| SHAP not yet integrated | Wrap scores in sklearn pipeline |

---

## Academic References

1. Farid, H. (2009). *Image forgery detection*. IEEE Signal Processing Magazine.
2. Rossler et al. (2019). *FaceForensics++*. ICCV 2019.
3. Frank et al. (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML 2020.
4. Matern et al. (2019). *Exploiting visual artifacts to expose deepfakes and face manipulations*. WACV Workshop.

---

## License

MIT License — free to use for academic and research purposes.

---

*Built for the Digital Forensics 
