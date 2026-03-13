import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks

# YOLOv8 Pose — dùng CUDA trực tiếp
model = YOLO("yolov8n-pose.pt")

# ── Landmark indices (YOLOv8 Pose = COCO format) ─────────────────────────────
# https://docs.ultralytics.com/tasks/pose
LANDMARKS = {
    "nose":            0,
    "left_eye":        1,  "right_eye":       2,
    "left_ear":        3,  "right_ear":       4,
    "left_shoulder":   5,  "right_shoulder":  6,
    "left_elbow":      7,  "right_elbow":     8,
    "left_wrist":      9,  "right_wrist":     10,
    "left_hip":        11, "right_hip":       12,
    "left_knee":       13, "right_knee":      14,
    "left_ankle":      15, "right_ankle":     16,
}

# ── Action configs ────────────────────────────────────────────────────────────
ACTION_CONFIGS = {
    "steps": {
        "keywords": ["step", "steps", "walk", "walks", "walking", "stride"],
        "signal_fn": lambda kp: abs(kp[LANDMARKS["left_ankle"]][1] - kp[LANDMARKS["right_ankle"]][1]),
        "peaks_kw":  {"distance": 8, "prominence": 5},
    },
    "jumps": {
        "keywords": ["jump", "jumps", "jumping", "leap", "leaps", "hop", "hops"],
        "signal_fn": lambda kp: (kp[LANDMARKS["left_hip"]][1] + kp[LANDMARKS["right_hip"]][1]) / 2,
        "peaks_kw":  {"distance": 10, "prominence": 10},
    },
    "flips": {
        "keywords": ["flip", "flips", "flipping", "somersault", "tumble", "rolls"],
        "signal_fn": lambda kp: kp[LANDMARKS["nose"]][1],
        "peaks_kw":  {"distance": 15, "prominence": 15},
    },
    "passes": {
        "keywords": ["pass", "passes", "passing"],
        "signal_fn": lambda kp: kp[LANDMARKS["right_wrist"]][1],
        "peaks_kw":  {"distance": 10, "prominence": 8},
    },
    "shoots": {
        "keywords": ["shoot", "shoots", "shot", "shots", "kick", "kicks"],
        "signal_fn": lambda kp: kp[LANDMARKS["right_knee"]][1],
        "peaks_kw":  {"distance": 10, "prominence": 8},
    },
    "squats": {
        "keywords": ["squat", "squats", "squatting"],
        "signal_fn": lambda kp: (kp[LANDMARKS["left_hip"]][1] + kp[LANDMARKS["right_hip"]][1]) / 2,
        "peaks_kw":  {"distance": 10, "prominence": 10},
    },
    "waves": {
        "keywords": ["wave", "waves", "waving"],
        "signal_fn": lambda kp: kp[LANDMARKS["left_wrist"]][1],
        "peaks_kw":  {"distance": 8, "prominence": 8},
    },
    "pushups": {
        "keywords": ["push", "pushup", "push-up", "pushups"],
        "signal_fn": lambda kp: (kp[LANDMARKS["left_shoulder"]][1] + kp[LANDMARKS["right_shoulder"]][1]) / 2,
        "peaks_kw":  {"distance": 10, "prominence": 8},
    },
    "situps": {
        "keywords": ["sit", "situp", "sit-up", "situps", "crunch", "crunches"],
        "signal_fn": lambda kp: abs(
            (kp[LANDMARKS["left_shoulder"]][1] + kp[LANDMARKS["right_shoulder"]][1]) / 2 -
            (kp[LANDMARKS["left_hip"]][1]      + kp[LANDMARKS["right_hip"]][1])      / 2
        ),
        "peaks_kw":  {"distance": 10, "prominence": 8},
    },
    "raises": {
        "keywords": ["raise", "raises", "raising", "lift", "lifts"],
        "signal_fn": lambda kp: (kp[LANDMARKS["left_wrist"]][1] + kp[LANDMARKS["right_wrist"]][1]) / 2,
        "peaks_kw":  {"distance": 8, "prominence": 8},
    },
}

DEFAULT_CONFIG = {
    "signal_fn": lambda kp: (kp[LANDMARKS["left_wrist"]][1] + kp[LANDMARKS["right_wrist"]][1]) / 2,
    "peaks_kw":  {"distance": 10, "prominence": 8},
}


def detect_action_type(question: str) -> tuple[str, dict]:
    """Từ câu hỏi → tìm action config phù hợp nhất."""
    q = question.lower()
    for action_name, config in ACTION_CONFIGS.items():
        for keyword in config["keywords"]:
            if keyword in q:
                print(f"  Detected action: '{action_name}' (keyword='{keyword}')")
                return action_name, config

    print("  No action detected → using default")
    return "unknown", DEFAULT_CONFIG


def _smooth(signal: list, window: int = 5) -> np.ndarray:
    """Moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="valid")


def count_actions(video_path: str, action_name: str, smoothing: int = 5, device: str = "cuda:0") -> int:
    """Đếm số lần lặp action dùng YOLOv8 Pose + CUDA.

    Args:
        video_path:  Đường dẫn video.
        action_name: Tên action (từ detect_action_type).
        smoothing:   Window size moving average.
        device:      CUDA device, vd: 'cuda:0', 'cuda:2'.

    Returns:
        Số lần action lặp lại.
    """

    config    = ACTION_CONFIGS.get(action_name, DEFAULT_CONFIG)
    signal_fn = config["signal_fn"]
    peaks_kw  = config["peaks_kw"]

    cap = cv2.VideoCapture(video_path)
    signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy YOLOv8 Pose trên GPU — trả về keypoints pixel coords (x, y, conf)
        results = model(frame, device=device, verbose=False)

        for r in results:
            if r.keypoints is None or len(r.keypoints.xy) == 0:
                continue

            # Lấy người đầu tiên detect được
            kp = r.keypoints.xy[0].cpu().numpy()  # shape: (17, 2) → [x, y] pixels

            # Tính signal value từ keypoints
            value = signal_fn(kp)
            signal.append(value)
            break  # chỉ lấy 1 người chính

    cap.release()

    if len(signal) < smoothing * 2:
        print("  Too few frames with pose detected.")
        return 0

    # Smooth → tìm peaks
    smoothed = _smooth(signal, smoothing)
    peaks, _ = find_peaks(-smoothed, **peaks_kw)

    print(f"  Signal length : {len(signal)} frames")
    print(f"  Peaks found   : {len(peaks)}")

    return len(peaks)


# ---------------------
# test phase
# ---------------------
if __name__ == "__main__":
    video    = "data/1UgJI6O8T2U.mp4"
    question = "How many steps the man takes before dancing?"

    action_name, _ = detect_action_type(question)
    result = count_actions(video, action_name, device="cuda:2")  # GPU rảnh nhất

    print(f"Action type   : {action_name}")
    print(f"Action counted: {result}")