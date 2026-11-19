import pickle

import pytest

from assay.classifier import train
from conftest import tiny_postings


def test_too_few_postings_is_an_error():
    with pytest.raises(train.TrainError, match="at least 3"):
        train.train(tiny_postings()[:2])


def test_only_one_class_is_an_error():
    postings = [[("Keep this one.", "H"), ("And this.", "R")] for _ in range(3)]
    with pytest.raises(train.TrainError, match="all keep or all cut"):
        train.train(postings)


def test_no_shared_words_is_an_error():
    postings = [[(f"alpha{i} keepword{i}.", "H"), (f"beta{i} cutword{i}.", "X")] for i in range(3)]
    with pytest.raises(train.TrainError, match="No word"):
        train.train(postings)


def test_summary_counts_postings_and_classes():
    _, summary = train.train(tiny_postings())
    assert (summary["postings"], summary["sentences"]) == (3, 12)
    assert (summary["keep"], summary["cut"]) == (6, 6)


def test_words_in_fewer_than_three_postings_are_not_clues():
    postings = tiny_postings()
    postings[0].append(("Zorblax is unique.", "X"))
    bundle, _ = train.train(postings)
    assert "zorblax" not in bundle["vectorizer"].vocabulary_
    assert "snacks" in bundle["vectorizer"].vocabulary_


def test_model_predicts_a_probability_per_sentence():
    bundle, _ = train.train(tiny_postings())
    X = bundle["vectorizer"].transform(["free snacks and perks", "5 years of python experience"])
    snacks, python = bundle["model"].predict_proba(X)[:, 1]
    assert snacks < 0.5 < python


def test_save_model_creates_directory_and_keeps_previous(tmp_path):
    path = tmp_path / "model" / "classifier.pkl"
    train.save_model({"v": 1}, path)
    assert not path.with_name("classifier.pkl.prev").exists()
    train.save_model({"v": 2}, path)
    assert pickle.loads(path.read_bytes()) == {"v": 2}
    assert pickle.loads(path.with_name("classifier.pkl.prev").read_bytes()) == {"v": 1}
    assert not path.with_name("classifier.pkl.new").exists()
