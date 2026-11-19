"""assay rebuild: label any posting that has no labels, then train the classifier on all of them."""
from assay import paths, store
from assay.classifier import label
from assay.classifier.features import split_posting


class RebuildError(Exception):
    """Labeling or training failed; the message says why."""


def rebuild(db):
    """Returns a summary of what it did. Labels are saved posting by posting, so a failed run can be repeated
    and carries on where it stopped."""
    todo = store.unlabeled_postings(db)
    for n, (posting_id, raw) in enumerate(todo, 1):
        sentences = split_posting(raw)
        print(f"Labeling posting {posting_id} ({n} of {len(todo)}, {len(sentences)} sentences)...", flush=True)
        try:
            labels = label.label_sentences(sentences)
        except label.LabelError as e:
            raise RebuildError(f"Posting {posting_id}: {e}")
        store.save_labels(db, posting_id, sentences, labels)

    from assay.classifier import train  # not at the top: scikit-learn is slow to load and only rebuild needs it
    try:
        bundle, summary = train.train(store.labeled_postings(db))
    except train.TrainError as e:
        raise RebuildError(str(e))
    train.save_model(bundle, paths.model_path())
    summary["newly_labeled"] = len(todo)
    return summary
