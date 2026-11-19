import builtins

from assay import paths


def answer(monkeypatch, text):
    monkeypatch.setattr(builtins, "input", lambda prompt="": text)


def test_purge_with_nothing_saved_says_so(run):
    code, out, _ = run("purge")
    assert code is None and "Nothing to delete" in out


def test_purge_yes_deletes_database_and_model(run, seeded, model_file, monkeypatch):
    seeded.close()
    answer(monkeypatch, "yes")
    code, out, _ = run("purge")
    assert code is None and "pipx uninstall" in out
    assert not paths.data_dir().exists()


def test_purge_anything_but_yes_keeps_everything(run, seeded, monkeypatch):
    answer(monkeypatch, "y")
    code, _, err = run("purge")
    assert code == "Not deleted."
    assert paths.db_path().exists()


def test_purge_without_a_terminal_keeps_everything(run, seeded, monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr(builtins, "input", eof)
    code, _, _ = run("purge")
    assert code == "Not deleted."
    assert paths.db_path().exists()


def test_purge_leaves_files_it_did_not_create(run, seeded, monkeypatch):
    seeded.close()
    other = paths.data_dir() / "notes.txt"
    other.write_text("mine")
    answer(monkeypatch, "yes")
    run("purge")
    assert other.exists() and not paths.db_path().exists()


def test_purge_takes_no_arguments(run):
    code, _, _ = run("purge", "now")
    assert code == 2
