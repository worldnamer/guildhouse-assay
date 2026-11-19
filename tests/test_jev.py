import pytest

from assay import jev


def big(n_tokens, tag):
    """Synthetic example text of about n_tokens estimated tokens, starting with tag."""
    return tag + "x" * int(n_tokens * jev.CHARS_PER_TOKEN)


def test_all_examples_are_used_when_they_fit():
    applied, passed = ["a1", "a2"], ["p1"]
    assert jev.select_examples("posting", "resume", applied, passed) == (applied, passed)


def make_pools():
    applied = [big(500, f"A{i}-") for i in range(60)]
    passed = [big(500, f"P{i}-") for i in range(60)]
    return applied, passed


def tokens(texts):
    return sum(jev.estimate_tokens(t) for t in texts)


def test_sample_stays_within_budget_when_examples_do_not_fit():
    applied, passed = make_pools()
    sa, sp = jev.select_examples("posting", "resume", applied, passed)
    room = jev.BUDGET - jev.PROMPT_TOKENS - jev.estimate_tokens("resume") - jev.estimate_tokens("posting")
    assert tokens(sa + sp) <= room
    assert len(sa + sp) < 120


def test_sample_is_balanced_between_applied_and_passed():
    applied, passed = make_pools()
    sa, sp = jev.select_examples("posting", "resume", applied, passed)
    assert abs(len(sa) - len(sp)) <= 1


def test_sample_only_contains_given_examples():
    applied, passed = make_pools()
    sa, sp = jev.select_examples("posting", "resume", applied, passed)
    assert set(sa) <= set(applied) and set(sp) <= set(passed)


def test_same_posting_gets_the_same_sample():
    applied, passed = make_pools()
    first = jev.select_examples("posting", "resume", applied, passed)
    assert jev.select_examples("posting", "resume", applied, passed) == first


def test_different_posting_gets_a_different_sample():
    applied, passed = make_pools()
    assert jev.select_examples("posting one", "resume", applied, passed) != \
        jev.select_examples("posting two", "resume", applied, passed)


def test_oversized_example_is_skipped_but_others_still_used():
    huge = big(jev.BUDGET * 2, "HUGE-")
    applied = [huge] + [big(500, f"A{i}-") for i in range(60)]
    passed = [big(500, f"P{i}-") for i in range(60)]
    sa, sp = jev.select_examples("posting", "resume", applied, passed)
    assert huge not in sa
    assert len(sa) > 0 and len(sp) > 0


def test_evaluate_sends_key_and_all_text(jev_post):
    jev.evaluate("the job", "secret-key", "my resume", ["applied one"], ["passed one"])
    call = jev_post.calls[0]
    assert call["headers"]["Authorization"] == "Bearer secret-key"
    state = call["json"]["state"]
    for text in ("my resume", "applied one", "passed one", "the job"):
        assert text in state


def test_evaluate_returns_the_json_answer(jev_post):
    jev_post.set_probability(0.9)
    assert jev.evaluate("j", "k", "r", ["a"], ["p"])["answers"]["decision"]["probabilities"]["apply"] == 0.9


def test_http_error_raises_jev_error(jev_post):
    from conftest import FakeResponse
    jev_post.response = FakeResponse(500, text="server broke")
    with pytest.raises(jev.JevError, match="500"):
        jev.evaluate("j", "k", "r", ["a"], ["p"])


def test_max_tokens_error_has_its_own_message(jev_post):
    from conftest import FakeResponse
    jev_post.response = FakeResponse(400, text='{"error": "max_tokens_exceeded"}')
    with pytest.raises(jev.JevError, match="too large"):
        jev.evaluate("j", "k", "r", ["a"], ["p"])
