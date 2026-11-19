from assay import paths, store


def test_init_creates_database(run):
    code, out, _ = run("init")
    assert code is None
    assert "Created" in out and paths.db_path().exists()


def test_init_hints_when_no_resume(run):
    _, out, _ = run("init")
    assert "assay resume" in out


def test_init_twice_says_up_to_date(run):
    run("init")
    _, out, _ = run("init")
    assert "up to date" in out


def test_init_adds_missing_table_and_says_updated(run, db):
    db.execute("DROP TABLE sentences")
    db.commit()
    _, out, _ = run("init")
    assert "Updated" in out and "sentences" in out


def test_init_does_not_hint_when_resume_exists(run, db):
    store.set_resume(db, "text", "r.txt")
    _, out, _ = run("init")
    assert "No résumé" not in out


def test_resume_saves_from_file(run, db, posting_file):
    code, out, _ = run("resume", posting_file("My synthetic resume."))
    assert code is None and "saved" in out
    assert store.get_resume(store.connect())["text"] == "My synthetic resume."


def test_resume_replaces_the_old_one(run, db, posting_file):
    run("resume", posting_file("first"))
    _, out, _ = run("resume", posting_file("second version", "two.txt"))
    assert "replaced" in out and "not kept" in out
    assert store.get_resume(store.connect())["text"] == "second version"


def test_resume_without_file_shows_the_saved_one(run, db):
    store.set_resume(db, "Shown text.", "r.txt")
    _, out, _ = run("resume")
    assert "Shown text." in out


def test_resume_without_file_and_none_saved_exits(run, db):
    code, _, _ = run("resume")
    assert "No résumé saved" in code


def test_resume_empty_file_exits_and_saves_nothing(run, db, posting_file):
    code, _, _ = run("resume", posting_file("  "))
    assert "is empty" in code
    assert store.get_resume(store.connect()) is None


def test_resume_missing_file_exits(run, db, tmp_path):
    code, _, _ = run("resume", str(tmp_path / "nope.txt"))
    assert "Could not read" in code


def test_resume_without_database_says_run_init(run):
    code, _, _ = run("resume")
    assert "assay init" in code
