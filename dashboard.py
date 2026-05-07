"""
dashboard.py  —  Streamlit Hybrid Forensics Dashboard  v3.0
============================================================
Run:  streamlit run dashboard.py
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


def _col(score):
    return "#4caf50" if score < .35 else "#ff9800" if score < .60 else "#f44336"

def _badge(score, label):
    cls = "badge-g" if score < .35 else "badge-y" if score < .60 else "badge-r"
    return f'<span class="badge {cls}">{label}</span>'

def _card(title, score, ml_used=False):
    ml_tag = '<span class="ml-tag">ML+Signal</span>' if ml_used else \
             '<span style="font-size:.7rem;color:#546e7a">Signal only</span>'
    st.markdown(f"""
    <div class="score-card">
      <h4>{title} {ml_tag}</h4>
      <div class="score-val" style="color:{_col(score)}">{score:.3f}</div>
    </div>""", unsafe_allow_html=True)
    st.progress(int(score * 100))

def _load_img(path):
    if path and os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    ela_q   = st.slider("ELA Quality",    70, 99, 95)
    ela_s   = st.slider("ELA Scale",       5, 30, 15)
    n_train = st.slider("Auto-train samples", 100, 500, 200, 50,
                        help="Synthetic samples per class if models missing")
    st.divider()
    st.markdown("### 🧠 Architecture")
    st.markdown("""
    **Pipeline v3.0**
    1. Feature Extraction  
       HOG · LBP · FFT · ELA · Color · DCT · Noise
    2. Signal-Processing Forensics  
       ELA · Splicing · AI-Gen · Deepfake
    3. ML Classifiers  
       SVM + Random Forest × 3 modules
    4. Score Fusion  
       Hybrid weighted combination
    5. SHAP Explainability  
       Feature importance per module
    6. PDF Evidence Report
    """)

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🔍 Hybrid AI Forensics System</h1>
  <p>Signal Processing + ML + SHAP · Digital Forensics Platform v3.0</p>
</div>""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image to investigate",
    type=["jpg","jpeg","png","bmp","webp"],
)
if not uploaded:
    c1,c2,c3 = st.columns(3)
    c1.markdown("###  Upload\nDrop any image file above.")
    c2.markdown("###  Analyse\nClick **Run Analysis** to start the full pipeline.")
    c3.markdown("###  Review\nScores · SHAP · Evidence · PDF Report")
    st.stop()

# ── Preview ───────────────────────────────────────────────────────────────────
pil_img = Image.open(uploaded)
ci, cm  = st.columns([1,1])
with ci:
    st.image(pil_img, caption="Uploaded image", use_container_width=True)
with cm:
    st.markdown("###  Image Info")
    st.markdown(f"- **Filename:** `{uploaded.name}`")
    st.markdown(f"- **Dimensions:** `{pil_img.width} × {pil_img.height} px`")
    st.markdown(f"- **Format:** `{pil_img.format or 'N/A'}`")
    st.markdown(f"- **Size:** `{uploaded.size:,}` bytes")
    run_btn = st.button("🔬 Run Forensic Analysis", type="primary",
                        use_container_width=True)

if not run_btn:
    st.stop()

# ── Run ───────────────────────────────────────────────────────────────────────
suffix   = Path(uploaded.name).suffix or ".jpg"
tmp_dir  = tempfile.mkdtemp()
img_path = os.path.join(tmp_dir, f"upload{suffix}")
with open(img_path, "wb") as f:
    f.write(uploaded.getvalue())

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
            investigator_id = "streamlit-v3",
        )
    except Exception as exc:
        st.error(f" Analysis failed: {exc}")
        st.exception(exc)
        st.stop()

st.success(f"Analysis complete in {report.get('elapsed_seconds','?')}s")
st.divider()

# ── Scores ────────────────────────────────────────────────────────────────────
st.markdown("##  Module Scores")
mods   = report.get("modules", {})
ml_s   = report.get("ml_scores", {})

s_ela  = mods.get("compression_ela",       {}).get("score", 0)
s_spl  = mods.get("splicing_detection",    {}).get("score", 0)
s_ai   = mods.get("ai_generated_detection",{}).get("score", 0)
s_dfk  = mods.get("deepfake_detection",    {}).get("score", 0)
s_fin  = report.get("final",{}).get("manipulation_probability", 0)
label  = report.get("final",{}).get("label","UNKNOWN")
conf   = report.get("final",{}).get("confidence","N/A")

c1,c2,c3,c4 = st.columns(4)
with c1: _card("🗜️ Compression/ELA", s_ela, False)
with c2: _card("✂️ Splicing",        s_spl, ml_s.get("splicing",{}).get("trained",False))
with c3: _card("🤖 AI-Generation",   s_ai,  ml_s.get("ai_gen",  {}).get("trained",False))
with c4: _card("👤 Deepfake",        s_dfk, ml_s.get("deepfake",{}).get("trained",False))

st.divider()

# ── Verdict ───────────────────────────────────────────────────────────────────
st.markdown("## 🏛️ Final Verdict")
vc, rc = st.columns([1,2])
with vc:
    _card("MANIPULATION PROBABILITY", s_fin,
          any(v.get("trained") for v in ml_s.values()))
    st.markdown(_badge(s_fin, label), unsafe_allow_html=True)
    st.caption(f"Confidence: **{conf.upper()}**")
with rc:
    rec = report.get("final",{}).get("recommendation","")
    st.info(rec)
    fusion = report.get("fusion_breakdown", {})
    if fusion:
        st.caption(f"Dominant module: **{fusion.get('dominant_module','N/A')}** "
                   f"| Modules above 50%: **{fusion.get('modules_above_50pct',0)}/4**")

st.divider()

# ── Detailed tabs ─────────────────────────────────────────────────────────────
st.markdown("## 🔬 Detailed Analysis")
tabs = st.tabs(["🗜️ ELA", "✂️ Splicing", "🤖 AI-Gen", "👤 Deepfake", "🧠 SHAP", "📥 Report"])

with tabs[0]:
    ela_m = mods.get("compression_ela",{})
    st.metric("ELA Score", f"{ela_m.get('score',0):.4f}")
    st.caption(ela_m.get("interpretation",""))
    st.metric("Suspicious Regions", ela_m.get("suspicious_regions",0))

with tabs[1]:
    spl_m = mods.get("splicing_detection",{})
    st.metric("Splicing Score", f"{spl_m.get('score',0):.4f}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Edge",      f"{spl_m.get('edge_score',0):.3f}")
    c2.metric("Lighting",  f"{spl_m.get('lighting_score',0):.3f}")
    c3.metric("Copy-Move", f"{spl_m.get('copy_move_score',0):.3f}")
    if ml_s.get("splicing",{}).get("trained"):
        c4,c5 = st.columns(2)
        c4.metric("RF Score",  f"{ml_s['splicing'].get('rf_score',0):.3f}")
        c5.metric("SVM Score", f"{ml_s['splicing'].get('svm_score',0):.3f}")
    st.caption(spl_m.get("interpretation",""))

with tabs[2]:
    ai_m = mods.get("ai_generated_detection",{})
    st.metric("AI-Gen Score", f"{ai_m.get('score',0):.4f}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Frequency",   f"{ai_m.get('frequency_score',0):.3f}")
    c2.metric("Noise",       f"{ai_m.get('noise_score',0):.3f}")
    c3.metric("Texture",     f"{ai_m.get('texture_score',0):.3f}")
    if ml_s.get("ai_gen",{}).get("trained"):
        c4,c5 = st.columns(2)
        c4.metric("RF Score",  f"{ml_s['ai_gen'].get('rf_score',0):.3f}")
        c5.metric("SVM Score", f"{ml_s['ai_gen'].get('svm_score',0):.3f}")
    st.caption(ai_m.get("interpretation",""))

with tabs[3]:
    dfk_m = mods.get("deepfake_detection",{})
    st.metric("Deepfake Score", f"{dfk_m.get('score',0):.4f}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Faces",    dfk_m.get("faces_detected",0))
    c2.metric("Landmark", f"{dfk_m.get('landmark_score',0):.3f}")
    c3.metric("Blending", f"{dfk_m.get('blending_score',0):.3f}")
    c4.metric("Colour",   f"{dfk_m.get('colour_score',0):.3f}")
    if ml_s.get("deepfake",{}).get("trained"):
        c5,c6 = st.columns(2)
        c5.metric("RF Score",  f"{ml_s['deepfake'].get('rf_score',0):.3f}")
        c6.metric("SVM Score", f"{ml_s['deepfake'].get('svm_score',0):.3f}")
    st.caption(dfk_m.get("interpretation",""))

with tabs[4]:
    shap_data = report.get("shap_explanations",{})
    if not shap_data:
        st.info("SHAP explanations unavailable (models not trained yet).")
    else:
        for mod, exp in shap_data.items():
            if not exp.get("available"):
                st.warning(f"SHAP unavailable for {mod}: {exp.get('summary','')}")
                continue
            st.markdown(f"### {mod.upper()} — {exp.get('dominant_domain','')}")
            st.success(exp.get("summary",""))
            top = exp.get("top_features",[])[:8]
            if top:
                import pandas as pd
                df = pd.DataFrame([{
                    "Group"      : f["feature_group"],
                    "SHAP"       : f"{f['shap_value']:+.4f}",
                    "Direction"  : f["direction"],
                    "Description": f["description"],
                } for f in top])
                st.dataframe(df, use_container_width=True, hide_index=True)
            st.divider()

with tabs[5]:
    # Evidence gallery
    ev = report.get("evidence_files",[])
    if ev:
        st.markdown("### Evidence Images")
        valid = [p for p in ev if _load_img(p) is not None]
        if valid:
            cols = st.columns(min(len(valid), 3))
            for i, path in enumerate(valid[:6]):
                img = _load_img(path)
                if img is not None:
                    cols[i % 3].image(img, caption=Path(path).stem.split("_")[-1],
                                       use_container_width=True)

    # SHA-256
    st.divider()
    coc = report.get("chain_of_custody",{})
    st.markdown("###  Chain of Custody")
    st.code(f"SHA-256: {coc.get('sha256','N/A')}")
    st.markdown(f"Case ID: `{report.get('case_id','N/A')}` | "
                f"Generated: `{report.get('generated_at','N/A')}`")

    # Downloads
    st.divider()
    st.markdown("### Downloads")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            " JSON Report",
            json.dumps(report, indent=2, default=str),
            file_name=f"forensic_{report.get('case_id','report')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dc2:
        pdf = report.get("pdf_path")
        if pdf and os.path.isfile(pdf):
            with open(pdf,"rb") as f:
                st.download_button(
                    " PDF Report",
                    f.read(),
                    file_name=f"forensic_{report.get('case_id','report')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.info("PDF generation requires reportlab")

    with st.expander(" Raw JSON"):
        st.json(report)
