"""
VisionGuard AI - Industrial Quality Inspection Operator Dashboard
"""

import base64
import io
import os
import random
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("VISIONGUARD_API_URL", "http://127.0.0.1:8000")
RAW_DATA_DIR = Path("data/raw")

st.set_page_config(
    page_title="VisionGuard AI | Quality Inspection Console",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ VisionGuard AI - Industrial Surface Inspection Console")
st.caption("PatchCore Multi-Scale Anomaly Detection • Zero-Leakage 3-State Calibration (PASS / REVIEW / FAIL)")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    cat_info = requests.get(f"{API_URL}/categories", timeout=3).json()["categories"]
    cat_map = {c["category"]: c for c in cat_info}
except Exception as exc:
    st.error(
        f"Cannot connect to VisionGuard FastAPI backend at `{API_URL}`.\n\n"
        f"Start the API server in a terminal first:\n"
        f"`uvicorn api.main:app --host 127.0.0.1 --port 8000`\n\nDetails: {exc}"
    )
    st.stop()

with st.sidebar:
    st.header("Inspection Line Setup")
    selected_category = st.selectbox(
        "Active Product Line",
        options=list(cat_map.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )
    meta = cat_map[selected_category]
    st.markdown("---")
    st.subheader("Calibrated Thresholds")
    st.metric("Normal Baseline (Mean ± Std)", f"{meta['val_mean']:.2f} ± {meta['val_std']:.2f}")
    col_a, col_b = st.columns(2)
    col_a.metric("PASS Cutoff", f"<= {meta['pass_threshold']:.2f}")
    col_b.metric("FAIL Cutoff", f"> {meta['fail_threshold']:.2f}")

    st.markdown("---")
    st.subheader("Quick Test Sample Loader")
    sample_type = st.radio("Sample Type", ["Defective Part", "Nominal (Good) Part"], horizontal=True)
    if st.button("Load Random MVTec Sample", use_container_width=True):
        test_root = RAW_DATA_DIR / selected_category / "test"
        if sample_type == "Nominal (Good) Part":
            candidates = list((test_root / "good").glob("*.png"))
        else:
            candidates = [
                p for p in test_root.rglob("*.png") if p.parent.name != "good"
            ]
        if candidates:
            chosen = random.choice(candidates)
            st.session_state["sample_bytes"] = chosen.read_bytes()
            st.session_state["sample_name"] = f"{chosen.parent.name}/{chosen.name}"
        else:
            st.warning("No local MVTec test images found under data/raw.")

stats = requests.get(f"{API_URL}/stats", timeout=3).json()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Inspected", stats["total_inspected"])
k2.metric("PASS Rate", f"{stats['pass_rate_pct']}%")
k3.metric("PASS Count", stats["pass_count"])
k4.metric("REVIEW Queue", stats["review_count"])
k5.metric("Avg Latency", f"{stats['avg_latency_ms']} ms")

st.markdown("---")

left_col, right_col = st.columns([1, 1.35])

with left_col:
    st.subheader("1. Part Image Input")
    uploaded = st.file_uploader(
        "Upload part image (PNG / JPG)", type=["png", "jpg", "jpeg"]
    )

    image_bytes = None
    filename = None
    if uploaded is not None:
        image_bytes = uploaded.read()
        filename = uploaded.name
    elif "sample_bytes" in st.session_state:
        image_bytes = st.session_state["sample_bytes"]
        filename = st.session_state["sample_name"]

    if image_bytes is not None:
        st.image(Image.open(io.BytesIO(image_bytes)), caption=f"Input: {filename}", use_container_width=True)
        run_btn = st.button("🔍 Run Quality Inspection", type="primary", use_container_width=True)
    else:
        st.info("Upload an image or click **Load Random MVTec Sample** in the sidebar.")
        run_btn = False

with right_col:
    st.subheader("2. Inspection Verdict & Defect Localization")
    if run_btn and image_bytes is not None:
        with st.spinner("Extracting multi-scale patches & scoring memory bank..."):
            try:
                resp = requests.post(
                    f"{API_URL}/inspect",
                    data={"category": selected_category},
                    files={"file": (filename, image_bytes, "image/png")},
                    timeout=120,
                )
            except requests.Timeout:
                st.error("The inspection is taking longer than 120 seconds. Please retry the same image.")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"Could not contact the inspection API: {exc}")
                st.stop()
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except requests.JSONDecodeError:
                detail = resp.text
            if resp.status_code == 422:
                st.error(f"### CATEGORY / OBJECT MISMATCH\n\n{detail}")
            else:
                st.error(f"Inspection failed: {detail}")
        else:
            res = resp.json()
            st.session_state["last_result"] = res
            st.session_state["last_input_bytes"] = image_bytes
            st.session_state["last_filename"] = filename

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        decision = res["decision"]

        if decision == "PASS":
            st.success(f"### VERDICT: PASS  (Inspection #{res['inspection_id']})")
        elif decision == "REVIEW":
            st.warning(f"### VERDICT: REVIEW REQUIRED  (Inspection #{res['inspection_id']})")
        else:
            st.error(f"### VERDICT: FAIL - DEFECT DETECTED  (Inspection #{res['inspection_id']})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Anomaly Score", f"{res['raw_score']:.2f}")
        m2.metric("Z-Score", f"{res['z_score']:.2f} σ")
        m3.metric("Defect Area", f"{res['defect_area_pct']:.2f}%")
        m4.metric("Latency", f"{res['latency_ms']:.1f} ms")

        st.progress(min(max(res["normalized_score"], 0.0), 1.0), text=f"Normalized Risk Index: {res['normalized_score']:.2f} / 1.00")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(
                Image.open(io.BytesIO(st.session_state["last_input_bytes"])),
                caption="Original Part",
                use_container_width=True,
            )
        with img_col2:
            overlay_bytes = base64.b64decode(res["overlay_base64"])
            st.image(
                Image.open(io.BytesIO(overlay_bytes)),
                caption="PatchCore Anomaly Heatmap Overlay",
                use_container_width=True,
            )

st.markdown("---")
st.subheader("3. Recent Inspection Audit Log")
history = requests.get(f"{API_URL}/history?limit=15", timeout=3).json()["records"]
if history:
    df_hist = pd.DataFrame(history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
else:
    st.caption("No inspections logged yet. Run an inspection above to populate the audit database.")
