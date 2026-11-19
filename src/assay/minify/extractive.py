"""Minify a posting by keeping or cutting each original sentence (no model call).

A classifier (model/classifier.pkl) scores each sentence.
A sentence is also kept by a fixed safety-net rule (N+ years, $ amount, remote/hybrid/onsite, named language/tech,
framing lines). Needs scikit-learn 1.9.1, the version that saved the pickle.
"""
import pickle
import re

from assay import paths
from assay.classifier.features import featurize, split_posting
from assay.minify import MinifyError

THRESHOLD = 0.5

RULES = re.compile(
    r"\d+\s*\+?\s*(?:years|yrs)"
    r"|\$\s*\d"
    r"|\b(?:remote|hybrid|on-?site)\b"
    r"|\b(?:python|ruby|rails|golang|java|javascript|typescript|c\+\+|c#|rust|kotlin|swift|scala|php|sql|react|aws|kubernetes|docker|terraform|node\.?js)\b"
    r"|\bGo\b",
    re.I,
)
GO_ONLY_CASE = re.compile(r"\bGo\b")
# Lines that frame the list below them ("Required:", "We'll be super impressed if you have...:", "Considered a Plus").
FRAMING = re.compile(r":\s*$|\b(?:plus|preferred|nice to have|impressed|bonus)\b", re.I)


def rule_keep(text):
    if FRAMING.search(text):
        return True
    m = RULES.search(text)
    if not m:
        return False
    if m.group(0).lower() == "go":  # "Go" the language only when capitalised
        return bool(GO_ONLY_CASE.search(text))
    return True


def load_model():
    path = paths.model_path()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise MinifyError(f"Classifier not found at {path}.")
    except (ImportError, pickle.UnpicklingError, EOFError) as e:
        raise MinifyError(f"Could not load the classifier at {path} (needs scikit-learn 1.9.1): {e}")


def classify(raw):
    """Return (kept sentences in original order, total sentence count)."""
    sentences = split_posting(raw)
    bundle = load_model()
    p = bundle["model"].predict_proba(bundle["vectorizer"].transform(featurize(sentences)))[:, 1]
    return [s for s, pi in zip(sentences, p) if pi >= THRESHOLD or rule_keep(s)], len(sentences)


def minify(raw):
    kept, _total = classify(raw)
    return "\n".join(kept)
