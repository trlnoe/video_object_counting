# pipeline.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.download_video import download_video
from preprocessing.extract_audio import extract_audio
from question.parser import parse_question
from workflow.router import route


def run_pipeline(video_id: str, question: str) -> dict:
    """Full end-to-end pipeline: download → extract → classify → answer.

    Args:
        video_id: YouTube video ID (e.g. 'k4pbOtw3omo').
        question: Natural language question about the video.

    Returns:
        Dict with keys: video_id, question, task, answer, video_path, audio_path.
    """

    print(f"\n{'='*55}")
    print(f"  Video ID : {video_id}")
    print(f"  Question : {question}")
    print(f"{'='*55}")

    # Step 1: download video from YouTube
    print("\n[1/4] Downloading video...")
    video_path = download_video(video_id)
    print(f"      Saved: {video_path}")

    # Step 2: extract audio track from video
    print("\n[2/4] Extracting audio...")
    try:
        audio_path = extract_audio(video_path)
        print(f"      Saved: {audio_path}")
    except ValueError as e:
        # Video may have no audio track (e.g. silent video)
        print(f"      Warning: {e} — speech_numeric will not be available.")
        audio_path = None

    # Step 3: classify question → task type
    print("\n[3/4] Parsing question...")
    task = parse_question(question)
    print(f"      Task: {task}")

    # Step 4: run the appropriate task module
    print("\n[4/4] Running task...")
    answer = route(task, video_path, audio_path, question)
    print(f"      Answer: {answer}")

    print(f"\n{'='*55}")
    print(f"  Final Answer: {answer}")
    print(f"{'='*55}\n")

    return {
        "video_id":   video_id,
        "question":   question,
        "task":       task,
        "answer":     answer,
        "video_path": video_path,
        "audio_path": audio_path,
    }


# ---------------------
# test phase
# ---------------------
if __name__ == "__main__":

    # Test cases from the assignment
    test_cases = [
        ("1UgJI6O8T2U", "how many cats are in the video?"),           # object_count
        ("72YsRrXi3B8", "how many steps the man takes?"),              # action_count
        ("k4pbOtw3omo", "How much did the guitar cost?"),              # speech_numeric
    ]

    for video_id, question in test_cases:
        result = run_pipeline(video_id, question)
        print(f"Video  : {result['video_id']}")
        print(f"Task   : {result['task']}")
        print(f"Answer : {result['answer']}\n")

