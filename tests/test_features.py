from assay.classifier.features import featurize, split_posting


def test_split_posting_gives_one_sentence_per_item():
    assert split_posting("First one. Second one!\nThird line") == ["First one.", "Second one!", "Third line"]


def test_split_posting_skips_blank_lines():
    assert split_posting("\n\nA line.\n   \nAnother.\n") == ["A line.", "Another."]


def test_split_posting_does_not_split_after_abbreviations():
    assert split_posting("Based in the U.S. with e.g. Python.") == ["Based in the U.S. with e.g. Python."]


def test_split_posting_of_empty_text_is_empty():
    assert split_posting("  \n ") == []


def test_featurize_returns_one_text_per_sentence():
    out = featurize(["Requirements", "You need 5 years of Rust."])
    assert len(out) == 2
    assert out[1].startswith("You need 5 years of Rust.")


def test_featurize_adds_cue_tokens():
    out = featurize(["Plenty of reading here.", "You need 5+ years of Rust."])
    assert "cue_years" in out[1]
    assert "cue_years" not in out[0]


def test_featurize_adds_section_tokens_from_latest_heading():
    out = featurize(["About us", "We make things.", "Requirements", "Know things well."])
    assert "sec_about sec_us" in out[1]
    assert "sec_requirements" in out[3]
    assert "sec_about" not in out[3]
