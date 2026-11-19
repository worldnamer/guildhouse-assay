import json
import os

from assay import store


def lines(text):
    return [json.loads(line) for line in text.splitlines()]


def test_export_to_file_has_resume_first_then_postings(run, seeded, tmp_path):
    target = tmp_path / "out.jsonl"
    code, _, err = run("export", str(target))
    assert code is None and "Exported 2 postings and 1 résumé" in err
    records = lines(target.read_text(encoding="utf-8"))
    assert [r["table"] for r in records] == ["resume", "postings", "postings"]
    assert records[0]["text"] == "Synthetic resume text."
    assert records[1]["raw_text"] == "raw applied"


def test_export_file_has_mode_600(run, seeded, tmp_path):
    target = tmp_path / "out.jsonl"
    run("export", str(target))
    assert os.stat(target).st_mode & 0o777 == 0o600


def test_export_refuses_to_overwrite(run, seeded, tmp_path):
    target = tmp_path / "out.jsonl"
    target.write_text("keep me")
    code, _, _ = run("export", str(target))
    assert "already exists" in code
    assert target.read_text() == "keep me"


def test_export_to_stdout(run, seeded):
    _, out, _ = run("export")
    assert [r["table"] for r in lines(out)] == ["resume", "postings", "postings"]


def test_export_without_resume_has_only_postings(run, db):
    store.save_posting(db, "s", "raw", "claude", "min", "applied")
    _, out, _ = run("export")
    assert [r["table"] for r in lines(out)] == ["postings"]


def test_export_to_unwritable_path_exits(run, seeded, tmp_path):
    code, _, _ = run("export", str(tmp_path / "no_dir" / "out.jsonl"))
    assert "Could not write" in code


def test_export_without_database_says_run_init(run, tmp_path):
    code, _, _ = run("export", str(tmp_path / "out.jsonl"))
    assert "assay init" in code
