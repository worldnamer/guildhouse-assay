"""Shared fixtures. Nothing here may reach the real Claude, Jev, the clipboard or the user's data."""
import json
import sys

import pytest
import pyperclip
import requests

from assay import cli, store
from assay.classifier import label, train
from assay.minify import claude

POSTING_A = "Senior widget engineer.\nYou need 5 years of python experience.\nWe offer free snacks and perks."


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text or json.dumps(body)

    def json(self):
        return self._body


def jev_answer(apply_prob, tokens=1000):
    return {"answers": {"decision": {"probabilities": {"apply": apply_prob, "pass": 1 - apply_prob},
                                     "confidence": 0.9}},
            "usage": {"input_tokens": tokens}}


class JevStub:
    """Stands in for requests.post; records every call."""
    def __init__(self):
        self.response = FakeResponse(200, jev_answer(0.5))
        self.calls = []

    def set_probability(self, p, tokens=1000):
        self.response = FakeResponse(200, jev_answer(p, tokens))

    def __call__(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own data directory and fake key, and a stubbed outside world."""
    monkeypatch.setenv("ASSAY_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    monkeypatch.setattr(pyperclip, "paste", lambda: "")


@pytest.fixture(autouse=True)
def jev_post(monkeypatch):
    stub = JevStub()
    monkeypatch.setattr(requests, "post", stub)
    return stub


@pytest.fixture(autouse=True)
def claude_minify(monkeypatch):
    calls = []

    def fake(raw):
        calls.append(raw)
        return "CLAUDE: " + raw.strip().splitlines()[0]

    monkeypatch.setattr(claude, "minify", fake)
    return calls


@pytest.fixture(autouse=True)
def labeler(monkeypatch):
    calls = []

    def fake(sentences):
        calls.append(list(sentences))
        return ["H" if i % 2 == 0 else "X" for i in range(len(sentences))]

    monkeypatch.setattr(label, "label_sentences", fake)
    return calls


@pytest.fixture
def clipboard(monkeypatch):
    def put(text):
        monkeypatch.setattr(pyperclip, "paste", lambda: text)
    return put


@pytest.fixture
def run(monkeypatch, capsys):
    """run("apply", "file.txt") calls main() and returns (exit code or None, stdout, stderr)."""
    def go(*args):
        monkeypatch.setattr(sys, "argv", ["assay", *args])
        code = None
        try:
            cli.main()
        except SystemExit as e:
            code = e.code
        out = capsys.readouterr()
        return code, out.out, out.err
    return go


@pytest.fixture
def db():
    store.init()
    return store.connect()


@pytest.fixture
def seeded(db):
    """A database with a résumé, one applied and one passed posting."""
    store.set_resume(db, "Synthetic resume text.", "resume.txt")
    store.save_posting(db, "a.txt", "raw applied", "claude", "applied posting", "applied")
    store.save_posting(db, "p.txt", "raw passed", "claude", "passed posting", "passed")
    return db


def tiny_postings():
    one = [("You need 5 years of python experience.", "H"), ("We offer free snacks and perks.", "X"),
           ("Remote role on a small team.", "F"), ("Our snacks and perks are great.", "X")]
    return [list(one) for _ in range(3)]


@pytest.fixture
def model_file():
    """Train a tiny classifier on synthetic postings and save it where the extractive minifier looks."""
    from assay import paths
    bundle, _ = train.train(tiny_postings())
    train.save_model(bundle, paths.model_path())
    return paths.model_path()


@pytest.fixture
def posting_file(tmp_path):
    def make(text=POSTING_A, name="posting.txt"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)
    return make
