from assay import store


def add(db, decision="applied", minified="Title line\nsecond line", raw="raw"):
    return store.save_posting(db, "f.txt", raw, "claude", minified, decision)


def test_list_with_nothing_saved(run, db):
    _, out, _ = run("list")
    assert "No postings saved" in out


def test_list_shows_one_line_per_posting_and_totals(run, db):
    add(db, "applied", "Alpha title", "r1")
    add(db, "passed", "Beta title", "r2")
    _, out, _ = run("list")
    assert "Alpha title" in out and "Beta title" in out
    assert "2 postings: 1 applied, 1 passed" in out


def test_list_truncates_long_titles(run, db):
    add(db, minified="W" * 100)
    _, out, _ = run("list")
    assert "W" * 100 not in out and "…" in out


def test_list_id_shows_minified_text(run, db):
    pid = add(db, minified="Full minified text.")
    _, out, _ = run("list", str(pid))
    assert f"Posting {pid}" in out and "Full minified text." in out


def test_list_unknown_id_exits(run, db):
    code, _, _ = run("list", "99")
    assert "No posting with id 99" in code


def test_list_rejects_non_numeric_id(run, db):
    code, _, _ = run("list", "abc")
    assert code == 2


def test_update_changes_decision(run, db):
    pid = add(db, "applied")
    _, out, _ = run("update", str(pid), "pass")
    assert "changed from applied to passed" in out
    assert store.get_posting(db, pid)["decision"] == "passed"


def test_update_to_same_decision_says_already(run, db):
    pid = add(db, "applied")
    _, out, _ = run("update", str(pid), "apply")
    assert "already applied" in out


def test_update_unknown_id_exits(run, db):
    code, _, _ = run("update", "5", "apply")
    assert "No posting with id 5" in code


def test_update_rejects_bad_decision(run, db):
    code, _, _ = run("update", "1", "maybe")
    assert code == 2


def test_remove_deletes_posting_and_labels(run, db):
    pid = add(db)
    store.save_labels(db, pid, ["a."], ["H"])
    _, out, _ = run("remove", str(pid))
    assert "Removed posting" in out
    assert store.get_posting(db, pid) is None
    assert db.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 0


def test_remove_unknown_id_exits(run, db):
    code, _, _ = run("remove", "7")
    assert "No posting with id 7" in code


def test_missing_table_gives_run_init_message(run, db):
    db.execute("DROP TABLE resume")
    db.commit()
    code, _, _ = run("list")
    assert "assay init" in code


def test_list_without_database_says_run_init(run):
    code, _, _ = run("list")
    assert "assay init" in code
