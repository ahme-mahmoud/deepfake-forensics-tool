"""
dashboard.py  —  Streamlit Hybrid Forensics Dashboard  v4.0
============================================================
Run:  streamlit run dashboard.py

Report structure (from app.py v4.0 / report_generator v3.0):
    report["module_scores"]      – {"ela", "ai_gen", "splicing", "deepfake"}
    report["module_labels"]      – {"ela": "LIKELY AUTHENTIC", ...}  (strings)
    report["module_details"]     – per-module rich detail dicts
    report["ml_availability"]    – {"ela", "ai_gen", "splicing", "deepfake"} booleans
    report["fusion_breakdown"]   – {module: weighted_contribution}
    report["shap_explanations"]  – always {} (SHAP disabled)
    report["final"]              – manipulation_probability, confidence,
                                   label, dominant_module, recommendation
    report["evidence_files"]     – list of evidence image paths
    report["chain_of_custody"]   – sha256, timestamps
    report["pdf_path"]           – path to generated PDF (or None)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).parent))

from app import run_pipeline

st.set_page_config(
    page_title="Forensics Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.banner{background:linear-gradient(135deg,#1a1f2e,#0d47a1);
  border-radius:12px;padding:22px 32px;margin-bottom:20px;border-left:5px solid #2196f3}
.banner h1{color:#90caf9;margin:0 0 4px 0;font-size:1.9rem}
.banner p{color:#90a4ae;margin:0;font-size:.88rem}
.score-card{background:#1a1f2e;border-radius:10px;padding:14px 18px;
  margin:6px 0;border-left:4px solid #2196f3}
.score-card h4{margin:0 0 6px 0;font-size:.8rem;color:#90a4ae;
  text-transform:uppercase;letter-spacing:1px}
.score-val{font-size:1.7rem;font-weight:700}
.badge{display:inline-block;padding:7px 18px;border-radius:20px;
  font-weight:700;font-size:.95rem;text-transform:uppercase;letter-spacing:1px}
.badge-g{background:#1b5e20;color:#a5d6a7;border:1px solid #2e7d32}
.badge-y{background:#f57f17;color:#fff9c4;border:1px solid #f9a825}
.badge-r{background:#b71c1c;color:#ffcdd2;border:1px solid #c62828}
.ml-tag{font-size:.7rem;padding:2px 8px;border-radius:10px;font-weight:600;
  background:#0d47a1;color:#bbdefb}
[data-testid="stSidebar"]{background:#1a1f2e!important}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(score: float) -> str:
    return "#4caf50" if score < .35 else "#ff9800" if score < .60 else "#f44336"

def _badge(score: float, label: str) -> str:
    cls = "badge-g" if score < .35 else "badge-y" if score < .60 else "badge-r"
    return f'<span class="badge {cls}">{label}</span>'

def _card(title: str, score: float, ml_used: bool = False) -> None:
    ml_tag = (
        '<span class="ml-tag">ML+Signal</span>' if ml_used
        else '<span style="font-size:.7rem;color:#546e7a">Signal only</span>'
    )
    st.markdown(f"""
    <div class="score-card">
      <h4>{title} {ml_tag}</h4>
      <div class="score-val" style="color:{_col(score)}">{score:.3f}</div>
    </div>""", unsafe_allow_html=True)
    st.progress(int(score * 100))

def _load_img(path: str):
    if path and os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def _fmt(val, fmt=".3f") -> str:
    """Safely format a numeric value; return 'N/A' if None or non-numeric."""
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "N/A"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    ela_q = st.slider("ELA Quality", 70, 99, 95)
    ela_s = st.slider("ELA Scale",    5, 30, 15)
    st.divider()
    st.markdown("### 🧠 Architecture")
    st.markdown("""
    **Pipeline v4.0**
    1. Feature Extraction  
       HOG · LBP · FFT · ELA · Color · DCT · Noise
    2. Compression / ELA Analysis
    3. AI-Generated Detection  
       Ensemble (GBM + ExtraTrees + SVM)
    4. Splicing Detection  
       Ensemble (GBM + ExtraTrees + SVM)
    5. Deepfake Detection  
       Ensemble classifier v2
    6. Score Fusion  
       Hybrid weighted combination
    7. PDF Evidence Report
    """)

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🔍 Hybrid AI Forensics System</h1>
  <p>Signal Processing + ML Ensembles · Digital Forensics Platform v4.0</p>
</div>""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image to investigate",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)
if not uploaded:
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 1️⃣ Upload\nDrop any image file above.")
    c2.markdown("### 2️⃣ Analyse\nClick **Run Analysis** to start the full pipeline.")
    c3.markdown("### 3️⃣ Review\nScores · Evidence · PDF Report")
    st.stop()

# ── Preview ───────────────────────────────────────────────────────────────────
pil_img = Image.open(uploaded)
ci, cm  = st.columns([1, 1])
with ci:
    st.image(pil_img, caption="Uploaded image", use_container_width=True)
with cm:
    st.markdown("### 📋 Image Info")
    st.markdown(f"- **Filename:** `{uploaded.name}`")
    st.markdown(f"- **Dimensions:** `{pil_img.width} × {pil_img.height} px`")
    st.markdown(f"- **Format:** `{pil_img.format or 'N/A'}`")
    st.markdown(f"- **Size:** `{uploaded.size:,}` bytes")
    run_btn = st.button("🔬 Run Forensic Analysis", type="primary",
                        use_container_width=True)

if not run_btn:
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
suffix   = Path(uploaded.name).suffix or ".jpg"
tmp_dir  = tempfile.mkdtemp()
img_path = os.path.join(tmp_dir, f"upload{suffix}")
with open(img_path, "wb") as fh:
    fh.write(uploaded.getvalue())

with st.spinner("🔍 Running hybrid forensic analysis…"):
    try:
        report = run_pipeline(
            image_path      = img_path,
            ela_quality     = ela_q,
            ela_scale       = ela_s,
            out_dir         = tmp_dir,
            evidence_dir    = os.path.join(tmp_dir, "evidence"),
            save_evidence   = True,
            case_id         = f"DASH-{Path(uploaded.name).stem[:10].upper()}",
            investigator_id = "streamlit-v4.0",
        )
    except Exception as exc:
        st.error(f"❌ Analysis failed: {exc}")
        st.exception(exc)
        st.stop()

st.success(f"✅ Analysis complete in {report.get('elapsed_seconds', '?')}s")
st.divider()

# ── Unpack new report structure ───────────────────────────────────────────────

# Flat module scores  {"ela": 0.3, "splicing": 0.6, ...}
mod_scores = report.get("module_scores", {})
s_ela      = float(mod_scores.get("ela",      0))
s_ai       = float(mod_scores.get("ai_gen",   0))
s_spl      = float(mod_scores.get("splicing", 0))
s_dfk      = float(mod_scores.get("deepfake", 0))

# ML availability  {"ela": False, "splicing": True, ...}
ml_avail = report.get("ml_availability", {})
ml_ai    = bool(ml_avail.get("ai_gen",   False))
ml_spl   = bool(ml_avail.get("splicing", False))
ml_dfk   = bool(ml_avail.get("deepfake", False))

# module_labels is a dict of STRINGS — e.g. {"ela": "LIKELY AUTHENTIC"}
# Per-module rich detail dicts live in module_details
mod_details = report.get("module_details", {})
ela_det     = mod_details.get("compression_ela",        {})
spl_det     = mod_details.get("splicing_detection",     {})
ai_det      = mod_details.get("ai_generated_detection", {})
dfk_det     = mod_details.get("deepfake_detection",     {})

# Final verdict
final   = report.get("final", {})
s_fin   = float(final.get("manipulation_probability", 0))
label   = final.get("label",          "UNKNOWN")
conf    = final.get("confidence",     "N/A")
rec     = final.get("recommendation", "")
dom_mod = final.get("dominant_module", "N/A")

# Fusion breakdown  {"ela": 0.06, "splicing": 0.15, ...}
fusion = report.get("fusion_breakdown", {})

# ── Module Score Cards ────────────────────────────────────────────────────────
st.markdown("## 📊 Module Scores")
c1, c2, c3, c4 = st.columns(4)
with c1: _card("🗜️ Compression/ELA", s_ela, False)
with c2: _card("✂️ Splicing",        s_spl, ml_spl)
with c3: _card("🤖 AI-Generation",   s_ai,  ml_ai)
with c4: _card("👤 Deepfake",        s_dfk, ml_dfk)
st.divider()

# ── Final Verdict ─────────────────────────────────────────────────────────────
st.markdown("## 🏛️ Final Verdict")
vc, rc = st.columns([1, 2])
with vc:
    _card("MANIPULATION PROBABILITY", s_fin, any([ml_ai, ml_spl, ml_dfk]))
    st.markdown(_badge(s_fin, label), unsafe_allow_html=True)
    st.caption(f"Confidence: **{conf.upper() if conf else 'N/A'}**")
with rc:
    if rec:
        st.info(rec)
    if fusion:
        above_50 = sum(1 for v in mod_scores.values() if float(v) >= 0.50)
        st.caption(
            f"Dominant module: **{dom_mod}** "
            f"| Modules above 50%: **{above_50}/4**"
        )
st.divider()

# ── Detailed Tabs ─────────────────────────────────────────────────────────────
st.markdown("## 🔬 Detailed Analysis")
tabs = st.tabs(["🗜️ ELA", "✂️ Splicing", "🤖 AI-Gen", "👤 Deepfake",
                "🧠 Explainability", "📥 Report"])

# ── Tab 0: ELA ────────────────────────────────────────────────────────────────
with tabs[0]:
    st.metric("ELA Score", f"{s_ela:.4f}")
    st.caption(ela_det.get("interpretation", ""))

    sub_cols = st.columns(2)
    sub_cols[0].metric("Suspicious Regions",
                        ela_det.get("suspicious_regions", "N/A"))
    # sha256 lives in chain_of_custody, not module_details
    coc_sha = report.get("chain_of_custody", {}).get("sha256", "N/A")
    sub_cols[1].metric("SHA-256 (truncated)",
                        str(coc_sha)[:16] + "…" if coc_sha != "N/A" else "N/A")

# ── Tab 1: Splicing ───────────────────────────────────────────────────────────
with tabs[1]:
    st.metric("Splicing Score", f"{s_spl:.4f}")

    # Signal sub-scores from splicing_module.predict() — surfaced via module_details
    c1, c2, c3 = st.columns(3)
    c1.metric("Edge Inconsistency",  _fmt(spl_det.get("edge_score")))
    c2.metric("Lighting Mismatch",   _fmt(spl_det.get("lighting_score")))
    c3.metric("Copy-Move",           _fmt(spl_det.get("copy_move_score")))

    # ML vs signal breakdown
    c4, c5 = st.columns(2)
    c4.metric("ML Score",     _fmt(spl_det.get("ml_score")))
    c5.metric("Signal Score", _fmt(spl_det.get("signal_score")))

    if ml_spl:
        st.markdown(
            '<span class="ml-tag">Ensemble ML active</span>',
            unsafe_allow_html=True,
        )
    label_spl = "SPLICED ⚠️" if spl_det.get("is_spliced") else "Intact ✅"
    st.markdown(f"**Detection label:** {label_spl}")
    st.caption(spl_det.get("interpretation", ""))

# ── Tab 2: AI-Gen ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.metric("AI-Gen Score", f"{s_ai:.4f}")

    # Signal sub-scores from ai_gen_module.predict() — surfaced via module_details
    c1, c2, c3 = st.columns(3)
    c1.metric("Frequency Anomaly", _fmt(ai_det.get("frequency_score")))
    c2.metric("Noise Fingerprint",  _fmt(ai_det.get("noise_score")))
    c3.metric("Texture Pattern",    _fmt(ai_det.get("texture_score")))

    # ML vs signal breakdown
    c4, c5 = st.columns(2)
    c4.metric("ML Score",     _fmt(ai_det.get("ml_score")))
    c5.metric("Signal Score", _fmt(ai_det.get("signal_score")))

    if ml_ai:
        st.markdown(
            '<span class="ml-tag">Ensemble ML active</span>',
            unsafe_allow_html=True,
        )
    label_ai = "AI-GENERATED ⚠️" if ai_det.get("is_ai") else "Authentic ✅"
    st.markdown(f"**Detection label:** {label_ai}")
    st.caption(ai_det.get("interpretation", ""))

# ── Tab 3: Deepfake ───────────────────────────────────────────────────────────
with tabs[3]:
    st.metric("Deepfake Score", f"{s_dfk:.4f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faces Detected",  dfk_det.get("faces_detected", "N/A"))
    c2.metric("Landmark Score",  _fmt(dfk_det.get("landmark_score")))
    c3.metric("Blending Score",  _fmt(dfk_det.get("blending_score")))
    c4.metric("Colour Score",    _fmt(dfk_det.get("colour_score")))

    # ML vs signal breakdown
    c5, c6 = st.columns(2)
    c5.metric("ML Score",     _fmt(dfk_det.get("ml_score")))
    c6.metric("Signal Score", _fmt(dfk_det.get("signal_score")))

    if ml_dfk:
        st.markdown(
            '<span class="ml-tag">Ensemble ML active</span>',
            unsafe_allow_html=True,
        )
    dfk_label = dfk_det.get("df_label") or dfk_det.get("label", "UNKNOWN")
    dfk_conf  = dfk_det.get("confidence", "")
    st.markdown(f"**Detection label:** {dfk_label}"
                + (f" — confidence: {dfk_conf}" if dfk_conf else ""))
    st.caption(dfk_det.get("interpretation", ""))

# ── Tab 4: Explainability (SHAP disabled) ─────────────────────────────────────
with tabs[4]:
    shap_data = report.get("shap_explanations", {})
    if not shap_data:
        st.info(
            "**Feature Explainability (SHAP) is currently disabled.**\n\n"
            "The SHAP layer will be re-enabled once the ensemble classifiers "
            "expose a compatible tree-based interface. All forensic scores above "
            "are produced by the signal-processing + ML ensemble pipeline."
        )
    else:
        # Future: render SHAP results when re-enabled
        for mod, exp in shap_data.items():
            if not exp.get("available"):
                st.warning(f"SHAP unavailable for **{mod}**: {exp.get('summary', '')}")
                continue
            st.markdown(f"### {mod.upper()} — {exp.get('dominant_domain', '')}")
            st.success(exp.get("summary", ""))
            top = exp.get("top_features", [])[:8]
            if top:
                import pandas as pd
                df = pd.DataFrame([{
                    "Group"      : feat["feature_group"],
                    "SHAP"       : f"{feat['shap_value']:+.4f}",
                    "Direction"  : feat["direction"],
                    "Description": feat["description"],
                } for feat in top])
                st.dataframe(df, use_container_width=True, hide_index=True)
            st.divider()

# ── Tab 5: Report & Downloads ─────────────────────────────────────────────────
with tabs[5]:
    # Evidence gallery
    ev = report.get("evidence_files", [])
    if ev:
        st.markdown("### 🗂️ Evidence Images")
        valid = [p for p in ev if _load_img(p) is not None]
        if valid:
            cols = st.columns(min(len(valid), 3))
            for i, path in enumerate(valid[:6]):
                img = _load_img(path)
                if img is not None:
                    cols[i % 3].image(
                        img,
                        caption=Path(path).stem.split("_")[-1],
                        use_container_width=True,
                    )
        else:
            st.info("No renderable evidence images found.")
    else:
        st.info("No evidence images were saved for this run.")

    # Chain of custody
    st.divider()
    coc = report.get("chain_of_custody", {})
    st.markdown("### 🔐 Chain of Custody")
    st.code(f"SHA-256: {coc.get('sha256', 'N/A')}")
    st.markdown(
        f"Case ID: `{report.get('case_id', 'N/A')}` | "
        f"Generated: `{report.get('generated_at', 'N/A')}`"
    )

    # Fusion detail
    if fusion:
        st.divider()
        st.markdown("### ⚖️ Score Fusion Breakdown")
        fb_cols = st.columns(4)
        for col, (key, label_) in zip(
            fb_cols,
            [("ela", "ELA"), ("ai_gen", "AI-Gen"),
             ("splicing", "Splicing"), ("deepfake", "Deepfake")],
        ):
            col.metric(label_, _fmt(mod_scores.get(key, 0)))

    # Downloads
    st.divider()
    st.markdown("### 📥 Downloads")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            "⬇️ JSON Report",
            json.dumps(report, indent=2, default=str),
            file_name=f"forensic_{report.get('case_id', 'report')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dc2:
        pdf = report.get("pdf_path")
        if pdf and os.path.isfile(str(pdf)):
            with open(pdf, "rb") as fh:
                st.download_button(
                    "⬇️ PDF Report",
                    fh.read(),
                    file_name=f"forensic_{report.get('case_id', 'report')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.info("PDF generation requires reportlab — install it to enable PDF export.")

    with st.expander("📄 Raw JSON"):
        st.json(report)
