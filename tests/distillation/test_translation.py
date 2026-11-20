"""FLORES-200 recipe helpers with the Hugging Face download stubbed: prompts, filters, items, sources and metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from post_training_cookbook.distillation.teacher import Spend, TeacherConfig, TeacherOutput, TeacherRequest

# The translation helpers are recipe code (recipes/distillation/ is not a package), so load the file by path.
_PATH = Path(__file__).resolve().parents[2] / "recipes" / "distillation" / "translation.py"
_spec = importlib.util.spec_from_file_location("translation", _PATH)
translation = importlib.util.module_from_spec(_spec)
sys.modules["translation"] = translation
_spec.loader.exec_module(translation)

LAO = ["ສະບາຍດີ", "ຂອບໃຈ", "ລາກ່ອນ"]
THAI = ["สวัสดี", "ขอบคุณ", "ลาก่อน"]
FLORES = {
    ("eng_Latn", "devtest"): ("Hello.", "Thank you.", "Goodbye."),
    ("lao_Laoo", "devtest"): tuple(LAO),
    ("eng_Latn", "dev"): ("Good morning.", "Hello.", "See you."),
}


@pytest.fixture(autouse=True)
def _flores(monkeypatch):
    monkeypatch.setattr(translation, "flores_lines", lambda lang, split: FLORES[(lang, split)])


def _request(key: str, source: str = "hello") -> TeacherRequest:
    return TeacherRequest(key=key, messages=[{"role": "user", "content": source}], meta={"source": source})


def test_script_fraction_counts_letters_only() -> None:
    assert translation.script_fraction(LAO[0], "LAO") == 1.0
    assert translation.script_fraction(THAI[0], "LAO") == 0.0
    assert translation.script_fraction("123 !!", "LAO") == 0.0
    assert 0.0 < translation.script_fraction(f"{LAO[0]} Obama", "LAO") < 1.0


def test_in_script_rejects_the_neighbouring_script() -> None:
    keep_lao = translation.in_script("LAO")
    assert keep_lao.__name__ == "in_script_lao"
    assert keep_lao(_request("k"), TeacherOutput(key="k", ok=True, content=LAO[0]))
    assert not keep_lao(_request("k"), TeacherOutput(key="k", ok=True, content=THAI[0]))


def test_not_an_echo() -> None:
    output = TeacherOutput(key="k", ok=True, content=" hello there ")
    assert not translation.not_an_echo(_request("k", "Hello there"), output)
    assert translation.not_an_echo(_request("k", "Hello there"), TeacherOutput(key="k", ok=True, content=LAO[0]))
    assert translation.not_an_echo(TeacherRequest(key="k", messages=[]), TeacherOutput(key="k", ok=True, content="x"))


def test_prompt_wording_is_stable() -> None:
    # Cache keys hash only the source, so a reworded prompt would silently reuse old translations.
    assert translation.translation_prompt("lao_Laoo", "Hi.") == (
        "Translate the following English text into Lao. Output only the Lao translation, with no explanation.\n\nHi."
    )


def test_teacher_requests_are_keyed_by_language_and_source() -> None:
    reqs = translation.teacher_requests("lao_Laoo", ["Hi.", "Bye.", "Hi."])

    assert len(reqs) == 2
    assert all(r.key.startswith("lao_Laoo:") for r in reqs)
    assert reqs[0].key == translation.teacher_requests("lao_Laoo", ["Hi."])[0].key
    assert reqs[0].key != translation.teacher_requests("tha_Thai", ["Hi."])[0].key
    assert reqs[0].meta == {"lang": "lao_Laoo", "source": "Hi."}


def test_eval_items_pair_source_and_reference_by_line() -> None:
    items = translation.eval_items("lao_Laoo", n=2, offset=1)

    assert [it["id"] for it in items] == ["lao_Laoo:devtest:1", "lao_Laoo:devtest:2"]
    assert items[0]["source"] == "Thank you." and items[0]["reference"] == LAO[1]
    assert items[0]["messages"] == translation.translation_messages("lao_Laoo", "Thank you.")
    assert len(translation.eval_items("lao_Laoo")) == 3


def test_eval_items_reject_misaligned_files(monkeypatch) -> None:
    monkeypatch.setitem(FLORES, ("lao_Laoo", "devtest"), tuple(LAO[:2]))
    with pytest.raises(ValueError, match="2 lines"):
        translation.eval_items("lao_Laoo")


def test_english_sources_never_include_a_devtest_sentence() -> None:
    sources = translation.english_sources(extra=["Goodbye.", "New one.", "Good morning."])
    assert sources == ["Good morning.", "See you.", "New one."]
    assert translation.english_sources(flores_dev=False, extra=["New one."]) == ["New one."]


def test_metrics_reward_the_reference_and_flag_the_wrong_script() -> None:
    items = translation.eval_items("lao_Laoo")

    perfect = translation.translation_metrics(items, [f'"{t}"' for t in LAO])  # quotes are cleaned
    wrong_language = translation.translation_metrics(items, THAI)

    assert perfect["chrf"] == pytest.approx(100.0)
    assert perfect["script_rate"] == 1.0
    assert wrong_language["chrf"] < 10.0
    assert wrong_language["script_rate"] == 0.0


def test_latin_script_languages_have_no_script_rate() -> None:
    assert translation.script_rate(["Sawubona"], "zul_Latn") is None
    assert translation.translation_metrics([], []) == {}


def test_translation_rows_filter_the_wrong_script_and_echoes(monkeypatch, tmp_path) -> None:
    def fake_generate(requests, config, cache_path):
        assert cache_path == tmp_path / "lao_Laoo.jsonl"
        contents = {"One.": f'"{LAO[0]}"', "Two.": THAI[0], "Three.": "Three."}
        return {
            r.key: TeacherOutput(key=r.key, ok=True, content=contents[r.meta["source"]]) for r in requests
        }, Spend()

    monkeypatch.setattr(translation, "generate", fake_generate)
    teacher = TeacherConfig(model="org/Teacher")

    rows = translation.translation_rows("lao_Laoo", ["One.", "Two.", "Three."], teacher, tmp_path)

    assert [r["meta"]["source"] for r in rows] == ["One."]
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": LAO[0]}


def test_translation_eval_set_names() -> None:
    assert translation.translation_eval_set("lao_Laoo", 2, name="lao").name == "lao"
    assert len(translation.translation_eval_set("lao_Laoo", None).items) == 3
    assert translation.translation_eval_set("lao_Laoo", 1).name == "lao_Laoo"


def test_cached_sentences_are_saved_and_reused(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_sentences(n, **kwargs):
        calls.append(n)
        return [f"Sentence {i}." for i in range(n)]

    monkeypatch.setattr(translation, "sentences_from_dataset", fake_sentences)
    path = tmp_path / "wiki.json"

    assert translation.cached_sentences(path, 4) == [f"Sentence {i}." for i in range(4)]
    assert translation.cached_sentences(path, 2) == ["Sentence 0.", "Sentence 1."]
    assert calls == [4]
    translation.cached_sentences(path, 6)
    assert calls == [4, 6]
