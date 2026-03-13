from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

def extract_target_from_question(question: str) -> str:
    """Try to extract the object type from the question.
    Example: 'how many books are in the video?' -> 'book'
    """
    import re

    q = question.lower()
    # Pattern: "how many <object(s)>"
    match = re.search(r"how many ([a-z\s]+?)(?:\s+are|\s+is|\s+do|\s+in|\?|$)", q)
    if match:
        obj = match.group(1).strip()
        # Remove trailing 's' for singular matching
        if obj.endswith("s"):
            obj = obj[:-1]
        return obj
    return None

def compute_iou(box1, box2) -> float:
    """Tính IoU giữa 2 bounding box định dạng [x1, y1, x2, y2]."""

    # Tọa độ vùng giao nhau
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Diện tích giao nhau (nếu không giao thì = 0)
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    if inter == 0:
        return 0.0

    # Diện tích từng box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU = giao / hợp
    return inter / (area1 + area2 - inter)


def count_objects(video_path: str, target_class: str = None) -> int:
    """Đếm object bằng max per frame + ByteTrack để ổn định hơn."""

    results = model.track(
        source=video_path,
        tracker="bytetrack_custom.yaml",  # dùng bytetrack
        persist=True,              # giữ track state giữa các frame
        verbose=False,
        stream=True,
    )

    class_map = model.names
    max_count = 0

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        frame_count = 0

        for i in range(len(r.boxes)):
            if target_class is not None:
                cls_idx = int(r.boxes.cls[i])
                cls_name = class_map[cls_idx].lower()
                if target_class.lower() not in cls_name:
                    continue
            frame_count += 1

        if frame_count > max_count:
            max_count = frame_count

    return max_count

# ---------------------
# test phase
# ---------------------
if __name__ == "__main__":
    video = "data/1UgJI6O8T2U.mp4"
    target_class = "cat"

    count = count_objects(video, target_class)
    print("Objects counted:", count)