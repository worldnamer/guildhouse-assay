"""Train the sentence classifier from labeled postings. Needs scikit-learn; only `rebuild` uses this module."""
import os
import pickle
import shutil
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from assay.classifier.features import TOKEN_PATTERN, featurize

KEEP = set("HRFW")  # X is cut
# A word becomes a clue only if it appears in at least this many different postings. Counting postings, not
# sentences, keeps words that belong to a single company (names, products) out of the model.
MIN_POSTINGS = 3


class TrainError(Exception):
    """The labeled postings cannot make a classifier."""


def train(postings):
    """postings: one list of (sentence, label) pairs per posting. Returns (model bundle, summary dict)."""
    analyze = CountVectorizer(token_pattern=TOKEN_PATTERN).build_analyzer()
    texts, y, tokens_per_posting = [], [], []
    for posting in postings:
        featured = featurize([sentence for sentence, _ in posting])
        texts += featured
        y += [1 if label in KEEP else 0 for _, label in posting]
        tokens_per_posting.append({token for text in featured for token in analyze(text)})

    if len(postings) < MIN_POSTINGS:
        raise TrainError(f"Need at least {MIN_POSTINGS} labeled postings; there are {len(postings)}.")
    if len(set(y)) < 2:
        raise TrainError("The labels are all keep or all cut, so there is nothing to learn from.")

    seen = Counter(token for tokens in tokens_per_posting for token in tokens)
    vocabulary = sorted(token for token, count in seen.items() if count >= MIN_POSTINGS)
    if not vocabulary:
        raise TrainError("No word appears in enough postings to learn from.")

    vectorizer = CountVectorizer(binary=True, token_pattern=TOKEN_PATTERN, vocabulary=vocabulary)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(vectorizer.fit_transform(texts), y)
    summary = {"postings": len(postings), "sentences": len(y), "keep": sum(y), "cut": len(y) - sum(y),
               "clues": len(vocabulary)}
    return {"vectorizer": vectorizer, "model": model}, summary


def save_model(bundle, path):
    """Write the model, keeping the previous one beside it as <name>.prev."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = path.with_name(path.name + ".new")
    with open(new, "wb") as f:
        pickle.dump(bundle, f)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".prev"))
    os.replace(new, path)
