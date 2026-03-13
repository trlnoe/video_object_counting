import cv2


def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames


# ---------------------
# test phase
# ---------------------

if __name__ == "__main__":

    video = "data/72YsRrXi3B8.mp4"

    frames = extract_frames(video)

    print("Total frames:", len(frames))