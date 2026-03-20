# CountVid WebApp

A web interface for counting objects in YouTube videos, powered by **CountGD + SAM 2**.

## Project Structure

```
video_object_counting/
├── CountVid/        ← original repo (do not modify)
├── sam2/            ← original repo (do not modify)
└── webapp/
    ├── api.py       ← FastAPI backend
    ├── app.py       ← Streamlit frontend
    └── README.md
```

---

## Part 1 — Setting Up CountVid (One-Time Setup)

### 1. Clone Repositories

```bash
git clone https://github.com/niki-amini-naieni/CountVid.git
cd ..
git clone https://github.com/facebookresearch/sam2.git
```

### 2. Create Conda Environment

```bash
conda create -n countvid python=3.10
conda activate countvid
conda install -c conda-forge gxx_linux-64 compilers libstdcxx-ng
```

### 3. Install GCC 11 (if no sudo)

```bash
# Install via conda instead of apt
conda install -y -c conda-forge gcc=11 gxx=11
conda install -y -c nvidia/label/cuda-12.1.0 cuda-toolkit

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 4. Install SAM 2

```bash
cd sam2
pip install -e .
cd ..
```

### 5. Install CountVid Dependencies

```bash
cd CountVid
pip install -r requirements.txt
```

### 6. Build GroundingDINO Ops

```bash
cd models/GroundingDINO/ops
python setup.py build install
python test.py   # should print 6 lines of: * True
cd ../../../
```

### 7. Install Detectron2

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### 8. Download Checkpoints

```bash
mkdir -p checkpoints

# Download BERT (required by CountGD)
python download_bert.py

# Download CountGD-Box model weights (~1.2 GB)
pip install gdown
gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/countgd_box.pth

# Download SAM 2.1 weights
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

After downloading, your `checkpoints/` directory should contain:

```
CountVid/checkpoints/
├── bert-base-uncased/      ← downloaded by download_bert.py
├── countgd_box.pth         ← CountGD-Box model (~1.2 GB)
└── sam2.1_hiera_large.pt   ← SAM 2.1 weights
```

### 9. Verify Setup

```bash
# Run the demo to confirm everything works
python count_in_videos.py \
  --video_dir demo \
  --input_text "penguin" \
  --sam_checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --pretrain_model_path checkpoints/countgd_box.pth \
  --temp_dir ./demo_temp \
  --output_dir ./demo_output \
  --save_final_video \
  --save_countgd_video
```

---

## Part 2 — Running the WebApp

### Install WebApp Dependencies

```bash
conda activate countvid
pip install fastapi uvicorn yt-dlp streamlit requests
conda install -y -c conda-forge ffmpeg
```

### Copy WebApp Files

```bash
mkdir -p /path/to/video_object_counting/webapp
cp api.py app.py README.md /path/to/video_object_counting/webapp/
```

### Start the App

Open **2 terminals**, both with conda activated:

```bash
conda activate countvid
cd /path/to/video_object_counting/webapp
```

**Terminal 1 — Backend:**
```bash
python api.py
# → http://0.0.0.0:8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run app.py
# → http://localhost:8501
```

---

## Usage

1. Open `http://<server-ip>:8501` in your browser
2. Enter a **YouTube Video ID** (the part after `?v=` in the URL)
3. Enter the **object to count** (e.g. `penguin`, `fish`, `car`)
4. Select the **GPU** to use
5. Click **Run CountVid** and watch the realtime logs
6. View the final count and output videos

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/count` | Run CountVid, return result when complete |
| `POST` | `/count/stream` | Run CountVid with realtime SSE log streaming |
| `GET`  | `/video?path=...` | Stream a video output file |
| `GET`  | `/health` | Health check |

### Example

```bash
curl -X POST http://localhost:8000/count \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "k4pbOtw3omo",
    "question": "penguin",
    "gpu_id": 0
  }'
```

**Response:**
```json
{
  "video_id": "k4pbOtw3omo",
  "question": "penguin",
  "count": 7,
  "output_dir": "/.../CountVid/results/k4pbOtw3omo_1234567890",
  "final_video": "/.../final-video-h264.mp4",
  "countgd_video": "/.../countgd-video.mp4"
}
```

### Request Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_id` | string | required | YouTube video ID |
| `question` | string | required | Object to count (e.g. `cat`, `person`) |
| `gpu_id` | int | `0` | GPU card index |
| `obj_batch_size` | int | `30` | Max objects per SAM 2 batch |
| `img_batch_size` | int | `10` | Batch size for CountGD inference |
| `downsample_factor` | int | `1` | Reduce total frames by this factor |

---

## Output Files

Results are saved permanently to `CountVid/results/<video_id>_<timestamp>/`:

```
├── final-video.mp4           ← original output (MP4V codec)
├── final-video-h264.mp4      ← H.264 re-encoded (browser-compatible) ✅
├── countgd-video.avi         ← CountGD bounding boxes (raw)
└── countgd-video.mp4         ← H.264 re-encoded (browser-compatible) ✅
```

---

## Notes

- **Runtime:** ~2–5 minutes depending on video length and object count
- **GPU memory:** Automatically freed after each run (isolated subprocess)
- **Disk cleanup:** Frames and temp files deleted after each run; only output videos kept
- **`downsample_factor`:** Increase for long videos (e.g. `3` = keep every 3rd frame)
- **`question`:** Use simple nouns (`cat`, `car`) rather than full questions for best accuracy
- **Results persist** across reboots under `CountVid/results/`

---

## Citation

```bibtex
@article{AminiNaieni25,
  title   = {Open-World Object Counting in Videos},
  author  = {Amini-Naieni, N. and Zisserman, A.},
  journal = {arXiv preprint arXiv:2506.15368},
  year    = {2025}
}

@InProceedings{AminiNaieni24,
  title     = {CountGD: Multi-Modal Open-World Counting},
  author    = {Amini-Naieni, N. and Han, T. and Zisserman, A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```