import yt_dlp
import os


def download_video(video_id):

    url = f"https://www.youtube.com/watch?v={video_id}"

    # os.makedirs("data", exist_ok=True)

    output_path = f"data/{video_id}.%(ext)s"

    ydl_opts = {
        "outtmpl": output_path,
        "format": "best[ext=mp4]/best",
        "quiet": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return f"data/{video_id}.mp4"


# test phase
if __name__ == "__main__":
    video_path = download_video("1UgJI6O8T2U")
    print("Video saved at:", video_path)