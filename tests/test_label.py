import json

from assay.classifier.label import _parse


def reply(labels):
    return json.dumps({"result": "Here you go: " + json.dumps({"labels": labels})})


def test_valid_reply_gives_codes_in_order():
    assert _parse(reply({"1": "H", "2": "X", "3": "r"}), 3) == ["H", "X", "R"]


def test_reply_with_too_few_labels_is_rejected():
    assert _parse(reply({"1": "H"}), 2) is None


def test_extra_labels_are_ignored():
    assert _parse(reply({"1": "H", "2": "X"}), 1) == ["H"]


def test_unknown_code_is_rejected():
    assert _parse(reply({"1": "H", "2": "Z"}), 2) is None


def test_junk_output_is_rejected():
    assert _parse("not json at all", 1) is None


def test_result_without_json_object_is_rejected():
    assert _parse(json.dumps({"result": "no braces here"}), 1) is None


def test_missing_labels_key_is_rejected():
    assert _parse(json.dumps({"result": '{"other": 1}'}), 1) is None
