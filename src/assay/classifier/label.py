"""Label every sentence of a posting keep or cut by asking Claude (headless `claude -p`, Haiku).

The labels are what the classifier learns from. The prompt leans toward keeping: missing a requirement
is much worse than keeping a useless sentence.
"""
import json
import re
import subprocess
import tempfile

MODEL = "claude-haiku-5-5"
CHUNK = 40  # sentences per call; a flat list of 60+ labels failed once
CODES = "HRFWX"

SYSTEM = """You label sentences from a job posting so a classifier can learn which sentences to KEEP when shortening the posting. Missing a requirement is much worse than keeping a useless sentence, so when unsure, KEEP.
For each numbered sentence output one code:
H = hard requirement or qualification: years of experience, a required or preferred language/stack/technology/domain, a degree line, seniority, anything a candidate could be screened out on
R = responsibility: what the person will do
F = key fact: compensation, location, remote/hybrid/onsite, level/title, employment type, visa or work authorization
W = one sentence that says what the company does (use for at most the one or two best such sentences per posting)
X = cut: benefits and perks, legal or EEO boilerplate, how to apply, mission fluff, repeated or filler text, headings with no content
Always use H for any "N+ years", any required language/stack, any degree line. Always use F for comp, level, location.
Reply with JSON only, one entry per sentence number you were given: {"labels": {"1": "H", "2": "X", ...}}"""


class LabelError(Exception):
    """Claude could not label the sentences."""


def label_sentences(sentences):
    """One code (H, R, F, W or X) per sentence. Raises LabelError if Claude cannot label them all."""
    labels = []
    for start in range(0, len(sentences), CHUNK):
        labels += _label_chunk(sentences[start:start + CHUNK], start)
    return labels


def _label_chunk(chunk, offset):
    prompt = "\n".join(f"{i}. {sentence[:600]}" for i, sentence in enumerate(chunk, 1))
    for _attempt in range(2):
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--tools", "",
                 "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"],
                input=prompt, capture_output=True, text=True, encoding="utf-8",
                cwd=tempfile.gettempdir(), timeout=300)
        except FileNotFoundError:
            raise LabelError("The `claude` command was not found.")
        except subprocess.TimeoutExpired:
            continue
        labels = _parse(result.stdout, len(chunk))
        if labels:
            return labels
    raise LabelError(f"Claude did not return valid labels for sentences {offset + 1} to {offset + len(chunk)}.")


def _parse(stdout, count):
    """The list of codes from Claude's JSON reply, or None unless every sentence has a valid one."""
    try:
        reply = json.loads(re.search(r"\{.*\}", json.loads(stdout)["result"], re.S).group(0))["labels"]
        codes = [str(reply[str(i)]).strip()[:1].upper() for i in range(1, count + 1)]
    except (ValueError, KeyError, AttributeError, TypeError):
        return None
    return codes if all(code in CODES for code in codes) else None
