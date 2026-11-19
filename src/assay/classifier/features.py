"""Sentence splitting and the features the classifier learns from.

Shared by the extractive minifier, which applies the model, and by training, which builds it, so the two
cannot drift apart. Changing anything here changes what a saved model means: rebuild after any edit.
"""
import re

# Words are tokenised with this pattern when the model is trained.
TOKEN_PATTERN = r"[\w+#]+"

# Phrases that tell the classifier something about a sentence beyond its words.
CUES = {
    "cue_years": r"\d+\s*\+?\s*(?:years|yrs)",
    "cue_money": r"\$\s*\d",
    "cue_such": r"\bsuch as\b|\bor similar\b|\be\.g\.",
    "cue_preferred": r"\bpreferred\b|\bnice to have\b|\bbonus\b|\bplus\b",
    "cue_youhave": r"\byou have\b|\byou've\b|\byou are\b|\byou'll\b",
    "cue_experience": r"\bexperience (?:with|in|using)\b|\bproficien",
}


def split_sentences(line):
    # split into sentences, but not after abbreviations such as "U.S.", "e.g." or "Inc."
    return re.split(r"(?<!U\.S\.)(?<!e\.g\.)(?<!i\.e\.)(?<!Inc\.)(?<!Ltd\.)(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<=[.!?])\s+", line.strip())


def split_posting(raw):
    """The posting's sentences, in order. Labels are stored against this exact split."""
    return [u.strip() for line in raw.splitlines() if line.strip() for u in split_sentences(line) if u.strip()]


def is_heading(text):
    words = text.split()
    return len(words) <= 5 and not text.endswith((".", "!", "?"))


def featurize(sentences):
    """One text per sentence: its words + cue tokens + section tokens from the latest heading."""
    out, section = [], ""
    for s in sentences:
        if is_heading(s):
            section = " ".join("sec_" + w for w in re.findall(r"[a-z]+", s.lower()))
        cues = " ".join(name for name, pat in CUES.items() if re.search(pat, s, re.I))
        out.append(f"{s} {cues} {section}")
    return out
