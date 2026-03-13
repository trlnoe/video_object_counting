import moviepy.editor as mp
import os


def extract_audio(video_path):

    os.makedirs("data", exist_ok=True)

    video = mp.VideoFileClip(video_path)

    audio_path = video_path.replace(".mp4", ".wav")

    video.audio.write_audiofile(audio_path)

    return audio_path


# ---------------------
# test phase
# ---------------------

if __name__ == "__main__":

    audio = extract_audio("data/k4pbOtw3omo.mp4")

    print("Audio saved:", audio)