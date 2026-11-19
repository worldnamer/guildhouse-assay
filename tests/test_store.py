import os
import sqlite3

import pytest

from assay import paths, store


def test_init_creates_directory_and_database_with_mode_600():
    existed, created = store.init()
    assert existed is False
    assert set(created) == {"postings", "sentences", "resume"}
    assert os.stat(paths.db_path()).st_mode & 0o777 == 0o600


def test_init_again_changes_nothing():
    store.init()
    assert store.init() == (True, [])


def test_init_adds_missing_table_and_keeps_data(db):
    store.set_resume(db, "my resume", "r.txt")
    db.execute("DROP TABLE sentences")
    db.commit()
    existed, created = store.init()
    assert (existed, created) == (True, ["sentences"])
    assert store.get_resume(store.connect())["text"] == "my resume"


def test_connect_without_database_says_run_init():
    with pytest.raises(store.StoreError, match="assay init"):
        store.connect()


def test_connect_with_missing_table_says_out_of_date(db):
    db.execute("DROP TABLE resume")
    db.commit()
    with pytest.raises(store.StoreError, match="out of date.*assay init"):
        store.connect()


def test_connect_rejects_a_file_that_is_not_a_database():
    paths.db_path().parent.mkdir(parents=True)
    paths.db_path().write_text("not sqlite at all")
    with pytest.raises(store.StoreError):
        store.connect()


def test_removing_a_posting_removes_its_labels(db):
    pid = store.save_posting(db, "s", "raw", "claude", "min", "applied")
    store.save_labels(db, pid, ["one.", "two."], ["H", "X"])
    store.delete_posting(db, pid)
    assert db.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 0


def test_relabeling_a_posting_is_refused(db):
    pid = store.save_posting(db, "s", "raw", "claude", "min", "applied")
    store.save_labels(db, pid, ["one."], ["H"])
    with pytest.raises(store.StoreError):
        store.save_labels(db, pid, ["one."], ["X"])


def test_labels_are_all_or_nothing(db):
    pid = store.save_posting(db, "s", "raw", "claude", "min", "applied")
    with pytest.raises(store.StoreError):
        store.save_labels(db, pid, ["one.", "two."], ["H", "Z"])
    assert store.unlabeled_postings(db) == [(pid, "raw")]


def test_find_posting_matches_exact_raw_text(db):
    pid = store.save_posting(db, "s", "raw text", "claude", "min", "passed")
    assert store.find_posting(db, "raw text")["id"] == pid
    assert store.find_posting(db, "raw text ") is None


def test_find_posting_returns_the_latest_copy(db):
    store.save_posting(db, "s", "same", "claude", "min", "passed")
    latest = store.save_posting(db, "s", "same", "claude", "min", "applied")
    assert store.find_posting(db, "same")["id"] == latest


def test_set_decision_changes_only_the_decision(db):
    pid = store.save_posting(db, "s", "raw", "claude", "min", "applied")
    store.set_decision(db, pid, "passed")
    posting = store.get_posting(db, pid)
    assert (posting["decision"], posting["minified_text"]) == ("passed", "min")


def test_set_decision_can_replace_minified_text(db):
    pid = store.save_posting(db, "s", "raw", "extractive", "old", "applied")
    store.set_decision(db, pid, "passed", "claude", "new")
    assert store.get_posting(db, pid)["minified_text"] == "new"


def test_save_posting_rejects_unknown_decision(db):
    with pytest.raises(store.StoreError):
        store.save_posting(db, "s", "raw", "claude", "min", "maybe")


def test_examples_are_filtered_by_decision_oldest_first(db):
    store.save_posting(db, "s", "r1", "claude", "first", "applied")
    store.save_posting(db, "s", "r2", "claude", "other", "passed")
    store.save_posting(db, "s", "r3", "claude", "second", "applied")
    assert store.examples(db, "applied") == ["first", "second"]


def test_unlabeled_postings_excludes_labeled_ones(db):
    a = store.save_posting(db, "s", "r1", "claude", "m", "applied")
    b = store.save_posting(db, "s", "r2", "claude", "m", "applied")
    store.save_labels(db, a, ["x."], ["H"])
    assert store.unlabeled_postings(db) == [(b, "r2")]


def test_labeled_postings_groups_sentences_in_order(db):
    a = store.save_posting(db, "s", "r1", "claude", "m", "applied")
    store.save_labels(db, a, ["one.", "two."], ["H", "X"])
    assert store.labeled_postings(db) == [[("one.", "H"), ("two.", "X")]]


def test_resume_is_none_until_saved(db):
    assert store.get_resume(db) is None


def test_set_resume_replaces_and_returns_the_old_text(db):
    assert store.set_resume(db, "first", "a.txt") is None
    assert store.set_resume(db, "second", "b.txt") == "first"
    saved = store.get_resume(db)
    assert (saved["text"], saved["source"]) == ("second", "b.txt")
    assert db.execute("SELECT COUNT(*) FROM resume").fetchone()[0] == 1
