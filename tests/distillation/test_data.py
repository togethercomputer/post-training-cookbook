"""Teacher outputs to SFT rows: drop reasons, filters, reasoning and mixing."""

from __future__ import annotations

import json

from post_training_cookbook.distillation.data import (
    in_script,
    mix,
    not_an_echo,
    script_fraction,
    sft_rows,
    write_jsonl,
)
from post_training_cookbook.distillation.teacher import TeacherOutput, TeacherRequest

LAO = "ສະບາຍດີ"
THAI = "สวัสดี"


def _request(key: str, source: str = "hello") -> TeacherRequest:
    return TeacherRequest(key=key, messages=[{"role": "user", "content": source}], meta={"source": source})


def test_script_fraction_counts_letters_only() -> None:
    assert script_fraction(LAO, "LAO") == 1.0
    assert script_fraction(THAI, "LAO") == 0.0
    assert script_fraction("123 !!", "LAO") == 0.0
    assert 0.0 < script_fraction(f"{LAO} Obama", "LAO") < 1.0


def test_in_script_rejects_the_neighbouring_script() -> None:
    keep_lao = in_script("LAO")
    assert keep_lao.__name__ == "in_script_lao"
    assert keep_lao(_request("k"), TeacherOutput(key="k", ok=True, content=LAO))
    assert not keep_lao(_request("k"), TeacherOutput(key="k", ok=True, content=THAI))


def test_not_an_echo() -> None:
    assert not not_an_echo(_request("k", "Hello there"), TeacherOutput(key="k", ok=True, content=" hello there "))
    assert not_an_echo(_request("k", "Hello there"), TeacherOutput(key="k", ok=True, content=LAO))
    assert not_an_echo(TeacherRequest(key="k", messages=[]), TeacherOutput(key="k", ok=True, content="x"))


def test_sft_rows_counts_each_drop_reason() -> None:
    requests = [_request(k) for k in ("ok", "missing", "failed", "truncated", "empty", "thai", "echo")]
    outputs = {
        "ok": TeacherOutput(key="ok", ok=True, content=f" {LAO} "),
        "failed": TeacherOutput(key="failed", ok=False, error="503"),
        "truncated": TeacherOutput(key="truncated", ok=True, content=LAO, finish_reason="length"),
        "empty": TeacherOutput(key="empty", ok=True, content="  "),
        "thai": TeacherOutput(key="thai", ok=True, content=THAI),
        "echo": TeacherOutput(key="echo", ok=True, content="hello"),
    }

    rows, dropped = sft_rows(requests, outputs, filters=[not_an_echo, in_script("LAO")])

    assert [r["id"] for r in rows] == ["ok"]
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": LAO}
    assert rows[0]["messages"][:-1] == requests[0].messages
    assert rows[0]["meta"] == {"source": "hello"}
    assert dict(dropped) == {
        "missing": 1,
        "failed": 1,
        "truncated": 1,
        "empty": 1,
        "in_script_lao": 1,
        "not_an_echo": 1,
    }


def test_reasoning_is_kept_only_when_asked() -> None:
    requests = [_request("k")]
    outputs = {"k": TeacherOutput(key="k", ok=True, content="answer", reasoning_content=" think ")}

    with_reasoning, _ = sft_rows(requests, outputs)
    without, _ = sft_rows(requests, outputs, keep_reasoning=False)

    assert with_reasoning[0]["messages"][-1]["reasoning_content"] == "think"
    assert "reasoning_content" not in without[0]["messages"][-1]


def test_mix_is_a_deterministic_shuffle_of_every_row() -> None:
    a = [{"id": f"a{i}"} for i in range(20)]
    b = [{"id": f"b{i}"} for i in range(5)]

    first = mix(a, b, seed=3)

    assert first == mix(a, b, seed=3)
    assert first != mix(a, b, seed=4)
    assert sorted(r["id"] for r in first) == sorted(r["id"] for r in a + b)


def test_jsonl_round_trip_keeps_non_ascii(tmp_path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    rows = [{"id": "1", "text": LAO}, {"id": "2", "text": THAI}]

    write_jsonl(path, rows)

    assert LAO in path.read_text(encoding="utf-8")
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == rows
