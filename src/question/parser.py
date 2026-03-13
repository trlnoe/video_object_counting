import re


# Keywords that indicate action/repeated movement counting
ACTION_KEYWORDS = [
    "times", "flip", "flips", "jump", "jumps", "pass", "passes",
    "step", "steps", "shoot", "shots", "bounce", "bounces",
    "kick", "kicks", "hit", "hits", "swing", "swings", "turn", "turns",
    "spin", "spins", "roll", "rolls", "dance", "dances", "wave", "waves",
]

# Keywords that indicate speech/audio numeric extraction
SPEECH_KEYWORDS = [
    "how much", "how long", "how far", "cost", "price", "distance",
    "height", "weight", "speed", "time", "duration", "learn",
]


def parse_question(question: str) -> str:
    """Classify question into one of three task types:
        - object_count:   counting visible objects
        - action_count:   counting repeated actions/movements
        - speech_numeric: extracting numbers from speech/audio

    Args:
        question: Natural language question about the video.

    Returns:
        Task type string: 'object_count' | 'action_count' | 'speech_numeric'
    """

    q = question.lower().strip()

    # Check speech/numeric first (cost, duration, distance, etc.)
    for kw in SPEECH_KEYWORDS:
        if kw in q:
            return "speech_numeric"

    # Check action count
    if "how many" in q:
        for kw in ACTION_KEYWORDS:
            if kw in q:
                return "action_count"
        return "object_count"

    # Fallback
    return "object_count"


# ---------------------
# test phase
# ---------------------
if __name__ == "__main__":
    tests = [
        ("How many books are in the video?", "object_count"),
        ("How many flips does this gymnast do?", "action_count"),
        ("How many times does the man shoot the ball?", "action_count"),
        ("How much did the guitar cost?", "speech_numeric"),
        ("How long did it take to learn the flute?", "speech_numeric"),
        ("How many passes before the goal?", "action_count"),
    ]

    print(f"{'Question':<55} {'Predicted':<15} {'Expected':<15} {'OK'}")
    print("-" * 100)
    for q, expected in tests:
        pred = parse_question(q)
        ok = "✅" if pred == expected else "❌"
        print(f"{q:<55} {pred:<15} {expected:<15} {ok}")