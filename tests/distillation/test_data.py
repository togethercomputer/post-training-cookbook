"""Teacher outputs to SFT rows: drop reasons, filter hooks, reasoning and mixing."""

from __future__ import annotations

import json

from post_training_cookbook.distillation.data import mix, sft_rows, write_jsonl
from post_training_cookbook.distillation.teacher import TeacherOutput, TeacherRequest

LAO = "ສະບາຍດີ"
THAI = "สวัสดี"


def _request(key: str, source: str = "hello") -> TeacherRequest:
    return TeacherRequest(key=key, messages=[{"role": "user", "content": source}], meta={"source": source})


def not_thai(request: TeacherRequest, output: TeacherOutput) -> bool:
    return output.content.strip() != THAI


def no_echo(request: TeacherRequest, output: TeacherOutput) -> bool:
    return output.content.strip() != request.meta["source"]


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

    rows, dropped = sft_rows(requests, outputs, filters=[no_echo, not_thai])

    assert [r["id"] for r in rows] == ["ok"]
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": LAO}
    assert rows[0]["messages"][:-1] == requests[0].messages
    assert rows[0]["meta"] == {"source": "hello"}
    assert dict(dropped) == {
        "missing": 1,
        "failed": 1,
        "truncated": 1,
        "empty": 1,
        "not_thai": 1,
        "no_echo": 1,
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
