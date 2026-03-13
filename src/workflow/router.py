# router.py

from tasks.object_count import count_objects, extract_target_from_question
from tasks.action_count import count_actions, detect_action_type
from tasks.speech_numeric import speech_numeric


def route(task: str, video_path: str, audio_path: str, question: str = "") -> str:
    """Route to the appropriate counting module based on task type.

    Args:
        task:        One of 'object_count', 'action_count', 'speech_numeric'.
        video_path:  Path to the video file.
        audio_path:  Path to the extracted audio file.
        question:    Original question (used to extract target object/action).

    Returns:
        Answer as a string.
    """

    if task == "object_count":
        # Extract target object from question e.g. "how many books" → "book"
        target = extract_target_from_question(question) if question else None
        count  = count_objects(video_path, target_class=target)
        return str(count)

    if task == "action_count":
        # Detect action type from question e.g. "how many steps" → "steps"
        action_name, _ = detect_action_type(question)
        count = count_actions(video_path, action_name)
        return str(count)

    if task == "speech_numeric":
        # Extract number from speech in audio
        result = speech_numeric(audio_path)
        return result if result is not None else "unknown"

    raise ValueError(f"Unknown task type: {task}")