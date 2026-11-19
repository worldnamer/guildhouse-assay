"""The plain `assay [file] [--claude]` run."""
import pytest

from assay import cli, store
from assay.minify import claude, extractive


@pytest.fixture
def ready(seeded, model_file):
    return seeded


def test_missing_api_key_exits(run, ready, posting_file, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    code, _, _ = run(posting_file())
    assert "TYPESAFE_API_KEY" in code


def test_missing_database_says_run_init(run, posting_file):
    code, _, _ = run(posting_file())
    assert "assay init" in code


def test_missing_resume_exits(run, db, posting_file):
    code, _, _ = run(posting_file())
    assert "No résumé saved" in code


def test_needs_an_applied_and_a_passed_example(run, db, posting_file):
    store.set_resume(db, "resume", "r.txt")
    store.save_posting(db, "s", "raw", "claude", "m", "applied")
    code, _, _ = run(posting_file())
    assert "at least one applied and one passed" in code


@pytest.mark.parametrize("prob, word", [
    (0.75, "APPLY"), (0.99, "APPLY"),
    (0.40, "PASS"), (0.0, "PASS"),
    (0.7499, "DIG IN"), (0.4001, "DIG IN"), (0.5, "DIG IN"),
])
def test_recommendation_bands(prob, word, run, ready, posting_file, jev_post):
    jev_post.set_probability(prob)
    code, out, _ = run(posting_file())
    assert code is None
    assert f"Recommendation: {word}" in out


def test_recommendation_function_edges():
    assert cli.recommendation(cli.APPLY_AT) == "APPLY"
    assert cli.recommendation(cli.PASS_AT) == "PASS"
    assert cli.recommendation(0.6).startswith("DIG IN")


def test_prints_probability_and_tokens(run, ready, posting_file, jev_post):
    jev_post.set_probability(0.8, tokens=1234)
    _, out, _ = run(posting_file())
    assert "Apply probability: 80%" in out and "Tokens used: 1234" in out


def test_plain_run_uses_extractive_minifier_by_default(run, ready, posting_file, jev_post, claude_minify):
    _, out, _ = run(posting_file())
    assert "Minifying (extractive)" in out
    assert claude_minify == []
    state = jev_post.calls[0]["json"]["state"]
    assert "You need 5 years of python experience." in state
    assert "free snacks and perks" not in state.split("JOB TO EVALUATE:")[1]


def test_claude_flag_uses_claude_minifier(run, ready, posting_file, jev_post, claude_minify):
    _, out, _ = run("--claude", posting_file())
    assert "Minifying (claude)" in out
    assert len(claude_minify) == 1
    assert "CLAUDE: Senior widget engineer." in jev_post.calls[0]["json"]["state"]


def test_plain_run_saves_nothing(run, ready, posting_file):
    before = ready.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
    run(posting_file())
    assert store.connect().execute("SELECT COUNT(*) FROM postings").fetchone()[0] == before


def test_plain_run_reads_clipboard_without_file(run, ready, clipboard, jev_post):
    clipboard("You need 3 years of python experience.")
    code, _, _ = run()
    assert code is None and len(jev_post.calls) == 1


def test_empty_clipboard_exits_without_calling_jev(run, ready, jev_post):
    code, _, _ = run()
    assert "Nothing to read" in code and jev_post.calls == []


def test_sends_resume_and_examples_to_jev(run, ready, posting_file, jev_post):
    run(posting_file())
    state = jev_post.calls[0]["json"]["state"]
    assert "Synthetic resume text." in state and "applied posting" in state and "passed posting" in state


def test_jev_http_error_exits_1(run, ready, posting_file, jev_post):
    from conftest import FakeResponse
    jev_post.response = FakeResponse(500, text="boom")
    code, _, _ = run(posting_file())
    assert "Jev HTTP 500" in code


def test_missing_model_exits_with_message(run, seeded, posting_file):
    code, _, _ = run(posting_file())
    assert "Classifier not found" in code


def test_sampling_notice_when_examples_exceed_budget(run, ready, posting_file, jev_post):
    for i in range(80):
        store.save_posting(ready, "s", f"r{i}", "claude", f"example {i} " + "x" * 2000, "applied" if i % 2 else "passed")
    _, out, _ = run(posting_file())
    assert "random sample" in out
    assert len(jev_post.calls) == 1


def test_no_sampling_notice_when_examples_fit(run, ready, posting_file):
    _, out, _ = run(posting_file())
    assert "random sample" not in out


def test_warns_when_jev_read_close_to_limit(run, ready, posting_file, jev_post):
    jev_post.set_probability(0.5, tokens=31_000)
    _, _, err = run(posting_file())
    assert "Warning" in err
