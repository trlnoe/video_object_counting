import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.workflow.pipeline import run_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Object Counter",
    page_icon="🎬",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 Video Object Counting")
st.caption("Đếm đối tượng trong video YouTube theo câu hỏi tự nhiên.")

st.divider()

# ── Example questions ─────────────────────────────────────────────────────────
with st.expander("📋 Ví dụ câu hỏi & video ID"):
    st.markdown("""
| Video ID | Question | Ground Truth |
|---|---|---|
| N623MG6xnak | how many times we see a lap top? | twice |
| Gw_73ZoYFr8 | how many books are in the video? | one |
| KanVJWDHduE | How many fish are in the video? | two |
| Gq-fFH6p7cQ | how many steps the man takes before stepping into the water? | two |
""")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("query_form"):
    col1, col2 = st.columns([1, 2])

    with col1:
        video_id = st.text_input(
            "YouTube Video ID",
            placeholder="e.g. Gw_73ZoYFr8",
        )

    with col2:
        question = st.text_input(
            "Question",
            placeholder="e.g. How many books are in the video?",
        )

    submitted = st.form_submit_button("🔍 Run", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if submitted:
    if not video_id.strip() or not question.strip():
        st.warning("⚠️ Vui lòng nhập cả Video ID và câu hỏi.")
    else:
        with st.spinner("Đang xử lý video... (có thể mất vài phút)"):
            try:
                result = run_pipeline(video_id.strip(), question.strip())

                st.success("✅ Hoàn thành!")

                # ── Results ──
                st.divider()
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("📌 Task", result["task"])

                with col_b:
                    st.metric("🎯 Answer", result["answer"])

                with col_c:
                    st.metric("🎞️ Video", result["video_id"])

                # ── YouTube embed ──
                st.divider()
                st.subheader("📺 Video")
                yt_url = f"https://www.youtube.com/watch?v={video_id.strip()}"
                st.video(yt_url)

                # ── Details ──
                with st.expander("🔍 Chi tiết"):
                    st.json({
                        "video_id": result["video_id"],
                        "question": result["question"],
                        "task_type": result["task"],
                        "answer": result["answer"],
                        "video_path": result["video_path"],
                        "audio_path": result["audio_path"],
                    })

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Môn học: Image Processing & Computer Vision | GV: Mai Tiến Dũng | HV: Trần Lê Hải Bình")