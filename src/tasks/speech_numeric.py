import re
import whisper

# Load Whisper model once at module level — "base" is fast, "small"/"medium" is more accurate
model = whisper.load_model("base")

# ── Word → number mapping ─────────────────────────────────────────────────────
WORD_TO_NUM = {
    # Cardinal numbers
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    # Special words that imply a quantity
    "once": 1, "twice": 2, "thrice": 3,
    "a": 1, "an": 1, "couple": 2, "few": 3, "several": 7,
    "half": 0.5, "quarter": 0.25,
}


def _extract_number(text: str) -> str | None:
    """Extract the first number from a transcript string.

    Priority order:
        1. Currency symbols: "$200", "€50", "£100"
        2. Plain digit numbers: "42", "3.5"
        3. Compound word numbers: "two hundred and fifty" → 250
        4. Single word numbers: "twice" → 2, "three" → 3
    """

    text_lower = text.lower()

    # Priority 1: currency amounts like "$200", "€ 50", "£100"
    currency = re.findall(r"[$€£]\s*(\d+(?:[.,]\d+)?)", text_lower)
    if currency:
        return currency[0]

    # Priority 2: plain digit numbers including decimals like "3.5", "1,000"
    digits = re.findall(r"\b\d+(?:[.,]\d+)?\b", text_lower)
    if digits:
        return digits[0]

    # Priority 3: compound word numbers like "two hundred and fifty" → 250
    compound = _parse_compound_number(text_lower)
    if compound is not None:
        return str(compound)

    # Priority 4: single word numbers, sorted by length (longest first)
    # to avoid matching "one" inside "nineteen"
    sorted_words = sorted(WORD_TO_NUM.keys(), key=len, reverse=True)
    for word in sorted_words:
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            return str(WORD_TO_NUM[word])

    return None


def _parse_compound_number(text: str) -> int | None:
    """Parse compound English number phrases into integers.

    Examples:
        "two hundred"             → 200
        "two hundred and fifty"   → 250
        "one thousand five hundred" → 1500
    """

    # Extract only known number words from the text
    tokens = re.findall(r"\b[a-z]+\b", text)

    result  = 0   # accumulated total
    current = 0   # current segment being built
    found_any = False

    for token in tokens:
        if token not in WORD_TO_NUM:
            # Skip non-number words like "and", "dollars", etc.
            continue

        val = WORD_TO_NUM[token]
        found_any = True

        if val == 100:
            # "two hundred" → multiply current by 100 (e.g. 2 * 100 = 200)
            current = (current if current != 0 else 1) * 100

        elif val == 1000:
            # "two thousand" → finalize current segment and scale by 1000
            current = (current if current != 0 else 1) * 1000
            result += current
            current = 0  # reset for the next segment

        else:
            # Regular number word → add to current segment
            current += val

    # Add remaining current segment to result
    result += current

    return result if found_any and result > 0 else None


def speech_numeric(audio_path: str) -> str | None:
    """Transcribe audio file and extract a numeric answer from the speech.

    Args:
        audio_path: Path to the .wav audio file.

    Returns:
        The first number found in the transcript as a string,
        or None if no number is detected.
    """

    # Step 1: transcribe audio using Whisper
    result = model.transcribe(audio_path)
    text   = result["text"].strip()
    print(f"  Transcript : {text}")

    # Step 2: extract number from transcript
    number = _extract_number(text)
    print(f"  Extracted  : {number}")

    return number


# ---------------------
# test phase
# ---------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from preprocessing.extract_audio import extract_audio

    video_path = "data/k4pbOtw3omo.mp4"

    audio_path = extract_audio(video_path)

    question = "How much did the guitar cost?"
    result = speech_numeric(audio_path)

    print(f"Question : {question}")
    print(f"Answer   : {result}")