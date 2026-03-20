"""
CountVid API
Nhận video_id (YouTube) + question, trả về object count + path 2 video output.

Usage:
    pip install fastapi uvicorn yt-dlp opencv-python
    python api.py

Endpoints:
    POST /count
    GET  /video?path=...
    GET  /health
"""

import os
import glob
import re
import shutil
import subprocess
import tempfile
from typing import Optional

import cv2
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="CountVid API")

# ── Paths ──────────────────────────────────────────────────────────────────
COUNTVID_DIR = os.path.join(os.getcwd(), "CountVid")
COUNT_SCRIPT = os.path.join(COUNTVID_DIR, "count_in_videos.py")
SAM2_CKPT    = os.path.join(COUNTVID_DIR, "checkpoints", "sam2.1_hiera_large.pt")
SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
COUNTGD_CKPT = os.path.join(COUNTVID_DIR, "checkpoints", "countgd_box.pth")

# Lưu kết quả vào thư mục cố định trong CountVid/results/
RESULTS_DIR  = os.path.join(COUNTVID_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Schemas ────────────────────────────────────────────────────────────────
class CountRequest(BaseModel):
    video_id: str
    question: str
    obj_batch_size: int = 30
    img_batch_size: int = 10
    downsample_factor: int = 1
    gpu_id: int = 0   # GPU card index (0-7)
    cuda_device: int = 0   # GPU card index (0-7)


class CountResponse(BaseModel):
    video_id: str
    question: str
    count: int
    output_dir: str
    final_video: Optional[str] = None      # final-video.mp4
    countgd_video: Optional[str] = None    # countgd-video.avi


# ── Download video ─────────────────────────────────────────────────────────
def download_video(video_id: str, output_dir: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "outtmpl": os.path.join(output_dir, f"{video_id}.%(ext)s"),
        "format": "best[ext=mp4]/best",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    files = glob.glob(os.path.join(output_dir, f"{video_id}.*"))
    if not files:
        raise FileNotFoundError(f"Download failed for video_id={video_id}")
    return files[0]


# ── Extract frames ─────────────────────────────────────────────────────────
def extract_frames(video_path: str, frames_dir: str, fps: int = 3) -> int:
    """
    Extract frames tại fps chỉ định vào frames_dir.
    CountVid nhận --video_dir là thư mục chứa các .jpg frames.
    """
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(round(video_fps / fps)))

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            cv2.imwrite(os.path.join(frames_dir, f"{saved + 1:05d}.jpg"), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


# ── Parse count ────────────────────────────────────────────────────────────
def parse_count(output: str) -> int:
    """
    count_in_videos.py in ra:
        "Total Number of Objects: 7"
    """
    for line in reversed(output.splitlines()):
        if "total number of objects" in line.lower():
            numbers = re.findall(r"\d+", line)
            if numbers:
                return int(numbers[-1])
    raise ValueError(f"Cannot parse count from output:\n{output}")


# ── Convert video to H.264 mp4 (browser-compatible) ──────────────────────
def convert_to_mp4(input_path: str, output_path: str) -> bool:
    """Re-encode video sang H.264 mp4 để browser play được."""
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vcodec", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",   # quan trọng: browser cần yuv420p
            "-acodec", "aac",
            output_path
        ], capture_output=True, text=True)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False

# ── GPU memory cleanup ────────────────────────────────────────────────────
def _free_gpu_memory():
    """Giải phóng GPU memory sau mỗi lần inference."""
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

# ── Run CountVid ───────────────────────────────────────────────────────────
def build_cmd(video_dir, question, temp_dir, output_dir, obj_batch_size, img_batch_size, downsample_factor):
    return [
        "python", COUNT_SCRIPT,
        "--video_dir",           video_dir,
        "--input_text",          question,
        "--sam_checkpoint",      SAM2_CKPT,
        "--sam_model_cfg",       SAM2_CFG,
        "--obj_batch_size",      str(obj_batch_size),
        "--img_batch_size",      str(img_batch_size),
        "--downsample_factor",   str(downsample_factor),
        "--pretrain_model_path", COUNTGD_CKPT,
        "--temp_dir",            temp_dir,
        "--output_dir",          output_dir,
        "--save_final_video",
        "--save_countgd_video",
    ]

def build_env(cuda_device: int = 0):
    env = os.environ.copy()
    env["PYTHONPATH"] = COUNTVID_DIR + ":" + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    return env

def run_countvid(
    video_dir: str,
    question: str,
    temp_dir: str,
    output_dir: str,
    obj_batch_size: int = 30,
    img_batch_size: int = 10,
    downsample_factor: int = 1,
    cuda_device: int = 0,
) -> str:
    cmd = build_cmd(video_dir, question, temp_dir, output_dir, obj_batch_size, img_batch_size, downsample_factor)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=COUNTVID_DIR, env=build_env(cuda_device),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"count_in_videos.py failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    # Giải phóng GPU memory sau mỗi lần chạy
    _free_gpu_memory()
    return result.stdout

def stream_countvid(
    video_dir: str,
    question: str,
    temp_dir: str,
    output_dir: str,
    obj_batch_size: int = 30,
    img_batch_size: int = 10,
    downsample_factor: int = 1,
    cuda_device: int = 0,
):
    """Generator: yield log lines realtime as SSE (text/event-stream)."""
    import json as _json
    cmd = build_cmd(video_dir, question, temp_dir, output_dir, obj_batch_size, img_batch_size, downsample_factor)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=COUNTVID_DIR, env=build_env(cuda_device),
        bufsize=1,
    )
    full_output = []
    for line in proc.stdout:
        line = line.rstrip()
        full_output.append(line)
        yield f"data: {_json.dumps({'type': 'log', 'line': line})}\n\n"
    proc.wait()
    # Giải phóng GPU memory sau mỗi lần chạy
    _free_gpu_memory()
    stdout_all = "\n".join(full_output)
    if proc.returncode != 0:
        yield f"data: {_json.dumps({'type': 'error', 'message': stdout_all})}\n\n"
    else:
        try:
            count = parse_count(stdout_all)
        except ValueError as e:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
        final_raw   = os.path.join(output_dir, "final-video.mp4")
        final_video = os.path.join(output_dir, "final-video-h264.mp4")
        countgd_avi = os.path.join(output_dir, "countgd-video.avi")
        countgd_mp4 = os.path.join(output_dir, "countgd-video.mp4")
        # Re-encode cả 2 sang H.264 cho browser
        if os.path.exists(final_raw):
            convert_to_mp4(final_raw, final_video)
        if os.path.exists(countgd_avi):
            convert_to_mp4(countgd_avi, countgd_mp4)
        countgd_video = countgd_mp4 if os.path.exists(countgd_mp4) else None
        yield f"data: {_json.dumps({'type': 'done', 'count': count, 'output_dir': output_dir, 'final_video': final_video if os.path.exists(final_video) else None, 'countgd_video': countgd_video})}\n\n"


# ── POST /count ────────────────────────────────────────────────────────────
@app.post("/count", response_model=CountResponse)
def count_objects(req: CountRequest):
    import time as _time
    video_id_str = req.video_id.strip().replace("/", "_")
    job_id     = f"{video_id_str}_{int(_time.time())}"
    work_dir   = tempfile.mkdtemp(prefix="countvid_")
    temp_dir   = os.path.join(work_dir, "temp")
    output_dir = os.path.join(RESULTS_DIR, job_id)   # lưu cố định
    os.makedirs(temp_dir, exist_ok=True)
    # KHÔNG tạo output_dir — count_in_videos.py tự gọi os.mkdir()

    try:
        # 1. Download
        video_path = download_video(req.video_id, work_dir)

        # 2. Extract frames → video_dir (đây là --video_dir của CountVid)
        video_dir = os.path.join(work_dir, "frames")
        n_frames = extract_frames(video_path, video_dir, fps=3)
        if n_frames == 0:
            raise HTTPException(status_code=400, detail="No frames extracted from video.")

        # 3. Run CountVid
        stdout = run_countvid(
            video_dir=video_dir,
            question=req.question,
            temp_dir=temp_dir,
            output_dir=output_dir,
            obj_batch_size=req.obj_batch_size,
            img_batch_size=req.img_batch_size,
            downsample_factor=req.downsample_factor,
        )

        # 4. Parse "Total Number of Objects: N"
        count = parse_count(stdout)

        # 5. Paths video output
        final_raw    = os.path.join(output_dir, "final-video.mp4")
        final_video  = os.path.join(output_dir, "final-video-h264.mp4")
        countgd_avi  = os.path.join(output_dir, "countgd-video.avi")
        countgd_mp4  = os.path.join(output_dir, "countgd-video.mp4")
        # Re-encode cả 2 sang H.264 cho browser
        if os.path.exists(final_raw):
            convert_to_mp4(final_raw, final_video)
        if os.path.exists(countgd_avi):
            convert_to_mp4(countgd_avi, countgd_mp4)
        countgd_video = countgd_mp4 if os.path.exists(countgd_mp4) else None

        return CountResponse(
            video_id=req.video_id,
            question=req.question,
            count=count,
            output_dir=output_dir,
            final_video=final_video if os.path.exists(final_video) else None,
            countgd_video=countgd_video,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Xóa frames + temp, giữ lại output
        shutil.rmtree(os.path.join(work_dir, "frames"), ignore_errors=True)
        shutil.rmtree(os.path.join(work_dir, "temp"),   ignore_errors=True)


# ── POST /count/stream ────────────────────────────────────────────────────
@app.post("/count/stream")
def count_stream(req: CountRequest):
    """
    Streaming endpoint: trả về SSE logs realtime.
    Client lắng nghe từng dòng log, cuối cùng nhận event 'done' với kết quả.
    """
    import time as _time
    video_id_str = req.video_id.strip().replace("/", "_")
    job_id     = f"{video_id_str}_{int(_time.time())}"
    work_dir   = tempfile.mkdtemp(prefix="countvid_")
    temp_dir   = os.path.join(work_dir, "temp")
    output_dir = os.path.join(RESULTS_DIR, job_id)   # lưu cố định
    os.makedirs(temp_dir, exist_ok=True)

    try:
        video_path = download_video(req.video_id, work_dir)
        video_dir  = os.path.join(work_dir, "frames")
        n_frames   = extract_frames(video_path, video_dir, fps=3)
        if n_frames == 0:
            raise HTTPException(status_code=400, detail="No frames extracted.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return StreamingResponse(
        stream_countvid(
            video_dir=video_dir,
            question=req.question,
            temp_dir=temp_dir,
            output_dir=output_dir,
            obj_batch_size=req.obj_batch_size,
            img_batch_size=req.img_batch_size,
            downsample_factor=req.downsample_factor,
            cuda_device=req.cuda_device,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── GET /video ─────────────────────────────────────────────────────────────
@app.get("/video")
def get_video(path: str):
    """
    Stream video theo path trả về từ /count.
    VD: GET /video?path=/tmp/countvid_xxx/output/final-video.mp4
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found.")
    ext = os.path.splitext(path)[-1].lower()
    media_type = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))


# ── GET /health ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8153, reload=False)