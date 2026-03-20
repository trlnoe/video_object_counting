import os

import json
import requests
import streamlit as st

API_URL = "http://localhost:8153"

st.set_page_config(page_title="CountVid", page_icon="🎬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d0d0d; color: #f0ede6; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
.block-container { max-width: 720px; padding-top: 3rem; }

.title-block { margin-bottom: 2.5rem; }
.title-block h1 { font-size: 3.2rem; line-height: 1.05; color: #f0ede6; margin-bottom: 0.3rem; }
.title-block p { color: #888; font-size: 1rem; font-family: 'DM Mono', monospace; margin-top: 0; }
.accent { color: #e8ff47; }

.result-card {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-left: 4px solid #e8ff47; border-radius: 4px;
    padding: 1.5rem 2rem; margin: 1.5rem 0;
}
.result-count { font-size: 4rem; font-weight: 800; color: #e8ff47; font-family: 'Syne', sans-serif; line-height: 1; }
.result-label { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
.result-question { font-size: 1.1rem; color: #f0ede6; margin-top: 0.8rem; }

.log-box {
    background: #0a0a0a; border: 1px solid #1e1e1e; border-radius: 4px;
    padding: 1rem 1.2rem; font-family: 'DM Mono', monospace;
    font-size: 0.78rem; color: #aaa; max-height: 260px;
    overflow-y: auto; margin: 1rem 0; line-height: 1.6;
    white-space: pre-wrap; word-break: break-all;
}
.log-line-highlight { color: #e8ff47; }
.log-line-stage { color: #88ccff; }

.video-label { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }

.stTextInput > div > div > input {
    background-color: #1a1a1a !important; border: 1px solid #2a2a2a !important;
    border-radius: 4px !important; color: #f0ede6 !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.95rem !important; padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus { border-color: #e8ff47 !important; box-shadow: 0 0 0 1px #e8ff47 !important; }
.stTextInput > label { color: #888 !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }

.stButton > button {
    background-color: #e8ff47 !important; color: #0d0d0d !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; border: none !important; border-radius: 4px !important;
    padding: 0.75rem 2rem !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.error-box {
    background: #1a0a0a; border: 1px solid #3a1a1a; border-left: 4px solid #ff4747;
    border-radius: 4px; padding: 1rem 1.5rem; font-family: 'DM Mono', monospace;
    font-size: 0.82rem; color: #ff8888; margin-top: 1rem; white-space: pre-wrap;
}
.divider { border: none; border-top: 1px solid #1e1e1e; margin: 2rem 0; }

/* Progress bar */
.progress-label { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #888; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>Count<span class="accent">Vid</span></h1>
    <p>image processing & computer vision · Professor: Mai Tien Dung  </p>
    <p>object counting in video · powered by CountGD + SAM 2 </p>
    <p>tranbinh · 250101007 </p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────────────────────
video_id = st.text_input("YouTube Video ID", placeholder="e.g.  k4pbOtw3omo")
question = st.text_input("What to count?", placeholder="e.g.  penguin, fish, person ...")

with st.expander("⚙️  Advanced options"):
    col0, col1, col2, col3 = st.columns(4)
    with col0: cuda_device = st.selectbox("GPU card", [0,1,2,3,4,5,6,7], index=4)
    with col1: obj_batch = st.number_input("obj_batch_size", value=30, min_value=1)
    with col2: img_batch = st.number_input("img_batch_size", value=10, min_value=1)
    with col3: downsample = st.number_input("downsample_factor", value=1, min_value=1)

run = st.button("▶  Run CountVid")

# ── Helper: classify log line ───────────────────────────────────────────────
def log_color(line: str) -> str:
    l = line.lower()
    if any(k in l for k in ["total number", "time stage", "stage 1", "stage 2", "stage 3"]):
        return f'<span class="log-line-highlight">{line}</span>'
    if any(k in l for k in ["batch idx", "propagate", "frame loading", "number of object"]):
        return f'<span class="log-line-stage">{line}</span>'
    return line

# ── Run ────────────────────────────────────────────────────────────────────
if run:
    if not video_id.strip():
        st.markdown('<div class="error-box">⚠ Please enter a YouTube Video ID.</div>', unsafe_allow_html=True)
    elif not question.strip():
        st.markdown('<div class="error-box">⚠ Please enter what to count.</div>', unsafe_allow_html=True)
    else:
        # ── Stage indicator ────────────────────────────────────────────────
        stage_placeholder = st.empty()
        stage_placeholder.markdown('<div class="progress-label">⏳ Downloading video & extracting frames...</div>', unsafe_allow_html=True)

        # ── Log box ────────────────────────────────────────────────────────
        log_placeholder = st.empty()
        log_lines = []

        def render_logs():
            html_lines = [log_color(l) for l in log_lines[-80:]]  # giữ 80 dòng cuối
            log_placeholder.markdown(
                f'<div class="log-box">{"<br>".join(html_lines)}</div>',
                unsafe_allow_html=True
            )

        result_data = None
        error_msg   = None

        try:
            with requests.post(
                f"{API_URL}/count/stream",
                json={
                    "video_id": video_id.strip(),
                    "question": question.strip(),
                    "obj_batch_size": obj_batch,
                    "img_batch_size": img_batch,
                    "downsample_factor": downsample,
                    "cuda_device": cuda_device,
                },
                stream=True,
                timeout=900,
            ) as resp:
                if resp.status_code != 200:
                    error_msg = f"HTTP {resp.status_code}: {resp.text}"
                else:
                    stage_placeholder.markdown('<div class="progress-label">🔍 Running CountVid...</div>', unsafe_allow_html=True)
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        if not line.startswith("data:"):
                            continue
                        payload = json.loads(line[5:].strip())

                        if payload["type"] == "log":
                            log_lines.append(payload["line"])
                            # Update stage label based on content
                            l = payload["line"].lower()
                            if "time stage 1" in l:
                                stage_placeholder.markdown('<div class="progress-label">🧠 Stage 1 done · Running Stage 3 (tracking)...</div>', unsafe_allow_html=True)
                            elif "time stage 3" in l:
                                stage_placeholder.markdown('<div class="progress-label">🎬 Generating output videos...</div>', unsafe_allow_html=True)
                            render_logs()

                        elif payload["type"] == "done":
                            result_data = payload
                            stage_placeholder.markdown('<div class="progress-label" style="color:#e8ff47">✅ Done!</div>', unsafe_allow_html=True)

                        elif payload["type"] == "error":
                            error_msg = payload["message"]

        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to API at {API_URL}.\nMake sure `python api.py` is running."
        except requests.exceptions.Timeout:
            error_msg = "Request timed out. Try increasing downsample_factor."
        except Exception as e:
            error_msg = str(e)

        # ── Show result ────────────────────────────────────────────────────
        if error_msg:
            st.markdown(f'<div class="error-box">❌ {error_msg}</div>', unsafe_allow_html=True)

        # Auto scroll xuống kết quả
        st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)

        if result_data:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card">
                <div class="result-count">{result_data['count']}</div>
                <div class="result-label">objects detected</div>
                <div class="result-question">"{question.strip()}" in video <code style="color:#e8ff47;background:none">{video_id.strip()}</code></div>
            </div>
            """, unsafe_allow_html=True)

            if result_data.get("final_video"):
                st.markdown('<div class="video-label">📹 Final tracking video</div>', unsafe_allow_html=True)
                path = result_data["final_video"]
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.video(f.read())
                else:
                    st.warning(f"File not found: {path}")

            if result_data.get("countgd_video"):
                st.markdown('<div class="video-label" style="margin-top:1.5rem">📦 CountGD box detection video</div>', unsafe_allow_html=True)
                path = result_data["countgd_video"]
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.video(f.read())
                else:
                    st.warning(f"File not found: {path}")