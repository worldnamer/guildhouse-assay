import pickle

from assay import paths, store
from assay.classifier import label


def add_postings(db, n=3):
    for i in range(n):
        raw = "You need 5 years of python experience.\nWe offer free snacks and perks.\nRemote role here.\nOur snacks and perks."
        store.save_posting(db, "s", raw + f" {i}", "claude", "min", "applied")


def test_rebuild_labels_trains_and_writes_model(run, db):
    add_postings(db)
    code, out, _ = run("rebuild")
    assert code is None and "Rebuilt" in out and "3 postings newly labeled" in out
    assert paths.model_path().exists()
    assert store.unlabeled_postings(store.connect()) == []


def test_rebuild_only_labels_postings_without_labels(run, db, labeler):
    add_postings(db)
    run("rebuild")
    labeler.clear()
    store.save_posting(db, "s", "You need 2 years of python experience. snacks and perks. Remote.", "claude", "m", "passed")
    run("rebuild")
    assert len(labeler) == 1


def test_rebuild_keeps_previous_model_as_prev(run, db):
    add_postings(db)
    run("rebuild")
    first = paths.model_path().read_bytes()
    run("rebuild")
    prev = paths.model_path().with_name("classifier.pkl.prev")
    assert prev.read_bytes() == first


def test_rebuilt_model_loads_with_vectorizer_and_model(run, db):
    add_postings(db)
    run("rebuild")
    bundle = pickle.loads(paths.model_path().read_bytes())
    assert set(bundle) == {"vectorizer", "model"}


def test_rebuild_with_too_few_postings_exits(run, db):
    add_postings(db, 2)
    code, _, _ = run("rebuild")
    assert "at least 3" in code
    assert not paths.model_path().exists()


def test_label_failure_exits_and_keeps_earlier_labels(run, db, monkeypatch):
    add_postings(db)
    calls = []

    def flaky(sentences):
        calls.append(1)
        if len(calls) == 2:
            raise label.LabelError("claude refused")
        return ["H"] * len(sentences)
    monkeypatch.setattr(label, "label_sentences", flaky)
    code, _, _ = run("rebuild")
    assert "Posting 2" in code and "claude refused" in code
    assert len(store.unlabeled_postings(store.connect())) == 2


def test_rebuild_without_database_says_run_init(run):
    code, _, _ = run("rebuild")
    assert "assay init" in code
