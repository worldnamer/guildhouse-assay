from assay import store


def rows(db):
    return db.execute("SELECT decision, minifier, minified_text, source FROM postings ORDER BY id").fetchall()


def test_apply_records_posting_minified_by_claude(run, db, posting_file):
    code, out, _ = run("apply", posting_file("Title line.\nMore."))
    assert code is None and "Recorded as applied" in out
    assert rows(db) == [("applied", "claude", "CLAUDE: Title line.", rows(db)[0][3])]


def test_pass_records_decision_passed(run, db, posting_file):
    run("pass", posting_file())
    assert rows(db)[0][0] == "passed"


def test_apply_reads_clipboard_without_files(run, db, clipboard):
    clipboard("Clipboard posting.")
    run("apply")
    assert rows(db)[0][3] == "clipboard"


def test_same_posting_updates_the_existing_row(run, db, posting_file):
    path = posting_file()
    run("apply", path)
    _, out, _ = run("pass", path)
    assert len(rows(db)) == 1 and rows(db)[0][0] == "passed"
    assert "Changed from applied to passed" in out


def test_same_posting_same_decision_says_already_recorded(run, db, posting_file):
    path = posting_file()
    run("apply", path)
    _, out, _ = run("apply", path)
    assert "Already recorded as applied" in out


def test_claude_text_is_reused_not_regenerated(run, db, posting_file, claude_minify):
    path = posting_file()
    run("apply", path)
    run("pass", path)
    assert len(claude_minify) == 1


def test_posting_minified_another_way_is_minified_by_claude_again(run, db, posting_file, claude_minify):
    raw = "Some raw posting."
    store.save_posting(db, "x", raw, "extractive", "old text", "applied")
    run("pass", posting_file(raw))
    assert len(claude_minify) == 1
    assert rows(db) == [("passed", "claude", "CLAUDE: Some raw posting.", rows(db)[0][3])]


def test_several_files_all_recorded(run, db, posting_file):
    code, out, _ = run("apply", posting_file("One.", "1.txt"), posting_file("Two.", "2.txt"))
    assert code is None and len(rows(db)) == 2
    assert "2 of 2 recorded as applied" in out


def test_failure_among_several_files_is_reported_and_rest_still_run(run, db, posting_file, tmp_path):
    code, out, err = run("apply", posting_file("One.", "1.txt"), str(tmp_path / "missing.txt"),
                         posting_file("Two.", "2.txt"))
    assert code == 1
    assert "FAILED" in err and "missing.txt" in err
    assert len(rows(db)) == 2
    assert "2 of 3 recorded" in out


def test_minify_failure_exits_1(run, db, posting_file, monkeypatch):
    from assay.minify import MinifyError, claude

    def boom(raw):
        raise MinifyError("claude broke")
    monkeypatch.setattr(claude, "minify", boom)
    code, _, err = run("apply", posting_file(), posting_file("Other.", "o.txt"))
    assert code == 1 and "claude broke" in err
    assert rows(db) == []


def test_single_file_failure_exits_1(run, db, tmp_path):
    code, _, err = run("apply", str(tmp_path / "missing.txt"))
    assert code == 1 and "FAILED" in err


def test_empty_minified_text_is_a_failure(run, db, posting_file, monkeypatch):
    from assay.minify import claude
    monkeypatch.setattr(claude, "minify", lambda raw: "  ")
    code, _, err = run("apply", posting_file())
    assert code == 1 and "produced nothing" in err
    assert rows(db) == []


def test_apply_without_database_says_run_init(run, posting_file):
    code, _, _ = run("apply", posting_file())
    assert "assay init" in code
