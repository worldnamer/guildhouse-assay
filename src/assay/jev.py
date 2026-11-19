"""Ask Jev whether to apply to a minified posting, given the résumé and past apply/pass examples."""
import hashlib
import random

import requests

URL = "https://api.typesafe.ai/v1/systemone"

# Jev 1.14 takes 64k tokens per request and 32k per question, and our one question reads all of the text,
# so we plan for 32k and aim well under it, because the token counts below are estimates.
CONTEXT_LIMIT = 32_000
BUDGET = 24_000
CHARS_PER_TOKEN = 4.5  # Jev's own count came to about 4.9 on our postings, so this overestimates a little
PROMPT_TOKENS = 1_500  # the instructions and questions wrapped around the text


class JevError(Exception):
    """Jev answered with an HTTP error."""


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN) + 1


def select_examples(minified, resume, applied, passed):
    """Pick the examples to send. Returns (applied, passed) lists.

    If they all fit in the budget, all are used. Otherwise it takes a random sample, alternating between
    applied and passed so neither crowds out the other. The sample is seeded by the posting's text, so the
    same posting always gets the same examples and its score does not drift between runs."""
    room = BUDGET - PROMPT_TOKENS - estimate_tokens(resume) - estimate_tokens(minified)
    if sum(estimate_tokens(t) for t in applied + passed) <= room:
        return list(applied), list(passed)

    rng = random.Random(int(hashlib.sha256(minified.encode("utf-8")).hexdigest(), 16))
    pools = [list(applied), list(passed)]
    for pool in pools:
        rng.shuffle(pool)
    chosen, used, turn = ([], []), 0, 0
    while pools[0] or pools[1]:
        side = turn % 2
        turn += 1
        if not pools[side]:
            continue
        text = pools[side].pop()
        if used + estimate_tokens(text) <= room:  # one too big to fit is skipped; smaller ones may still go in
            chosen[side].append(text)
            used += estimate_tokens(text)
    return chosen[0], chosen[1]


def evaluate(minified, key, resume, applied, passed):
    """Return Jev's raw JSON answer. Raises JevError on an HTTP error.

    applied and passed are lists of minified postings: the examples Jev compares the new posting against."""
    applied = "\n\n---\n\n".join(applied)
    passed = "\n\n---\n\n".join(passed)

    state = f"""RESUME:
{resume}

JOBS I APPLIED TO (strong matches):
{applied}

JOBS I PASSED ON (poor matches):
{passed}

JOB TO EVALUATE:
{minified}
"""
    payload = {
        "model": "jev-latest",
        "state": state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Based on the resume and the candidate's past apply/pass decisions, should they apply to this job?",
                "criteria": {
                    "apply": "Strong enough match — candidate should apply",
                    "pass": "Poor match — candidate should skip",
                },
            },
        },
    }
    response = requests.post(
        URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload)
    if response.status_code != 200:
        if "max_tokens_exceeded" in response.text:
            raise JevError("Jev refused the request as too large (max_tokens_exceeded). Too many examples for its limit.")
        raise JevError(f"Jev HTTP {response.status_code}: {response.text[:200]}")
    return response.json()
