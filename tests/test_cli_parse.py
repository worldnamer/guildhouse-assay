import sys

import pytest

from assay import cli


def parse(*args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["assay", *args])
    return cli.parse_args()


@pytest.mark.parametrize("word", ["init", "resume", "list", "apply", "pass", "update", "remove", "export", "rebuild", "purge"])
def test_reserved_word_is_a_command(word, monkeypatch):
    args = {"update": ["update", "1", "apply"], "remove": ["remove", "1"]}.get(word, [word])
    assert parse(*args, monkeypatch=monkeypatch).command == word


def test_dot_slash_init_is_a_file(monkeypatch):
    args = parse("./init", monkeypatch=monkeypatch)
    assert args.command is None and args.path == "./init"


def test_reserved_word_in_a_longer_name_is_a_file(monkeypatch):
    assert parse("init.txt", monkeypatch=monkeypatch).path == "init.txt"


def test_no_arguments_means_clipboard_run(monkeypatch):
    args = parse(monkeypatch=monkeypatch)
    assert (args.command, args.path, args.claude) == (None, None, False)


def test_claude_flag(monkeypatch):
    assert parse("--claude", "f.txt", monkeypatch=monkeypatch).claude is True


def test_init_takes_no_arguments(monkeypatch):
    with pytest.raises(SystemExit):
        parse("init", "extra", monkeypatch=monkeypatch)


def test_command_word_only_counts_as_first_argument(monkeypatch):
    args = parse("--claude", "list", monkeypatch=monkeypatch)
    assert args.command is None and args.path == "list"


def test_update_rejects_unknown_decision(monkeypatch):
    with pytest.raises(SystemExit):
        parse("update", "1", "maybe", monkeypatch=monkeypatch)


def test_apply_collects_several_files(monkeypatch):
    assert parse("apply", "a", "b", monkeypatch=monkeypatch).paths == ["a", "b"]
