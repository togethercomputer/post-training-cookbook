"""FLORES-200 translation helpers for distillation: prompts, eval items, chrF++ and script rate.

FLORES-200 has two splits per language, ``dev`` (997 sentences) and ``devtest`` (1,012), translated
line-for-line from the same English. Teacher sources come from ``dev`` (or any other English text)
and evaluation from ``devtest``; :func:`english_sources` drops any devtest sentence so a test
sentence can never reach training. The files are read from the ``DGME/FLORES-200`` Hugging Face
mirror, which needs no token.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import sacrebleu

from post_training_cookbook.distillation.data import in_script, not_an_echo, script_fraction, sft_rows
from post_training_cookbook.distillation.evaluate import EvalSet
from post_training_cookbook.distillation.teacher import TeacherConfig, TeacherRequest, generate
from post_training_cookbook.types import Message

FLORES_REPO = "DGME/FLORES-200"

LANGUAGES = {
    "eng_Latn": "English",
    "lao_Laoo": "Lao",
    "tha_Thai": "Thai",
    "yor_Latn": "Yoruba",
    "hau_Latn": "Hausa",
    "amh_Ethi": "Amharic",
    "tir_Ethi": "Tigrinya",
    "khm_Khmr": "Khmer",
    "mya_Mymr": "Burmese",
    "zul_Latn": "Zulu",
    "som_Latn": "Somali",
    "quy_Latn": "Ayacucho Quechua",
    "nso_Latn": "Northern Sotho",
}

# Unicode script prefix for languages with their own script (see ``data.script_fraction``).
SCRIPTS = {
    "lao_Laoo": "LAO",
    "tha_Thai": "THAI",
    "amh_Ethi": "ETHIOPIC",
    "tir_Ethi": "ETHIOPIC",
    "khm_Khmr": "KHMER",
    "mya_Mymr": "MYANMAR",
}


@lru_cache(maxsize=64)
def flores_lines(lang: str, split: str) -> tuple[str, ...]:
    """The sentences of one FLORES-200 language and split, in line order."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FLORES_REPO, f"{split}/{lang}.{split}", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return tuple(f.read().splitlines())


def translation_prompt(lang: str, text: str) -> str:
    name = LANGUAGES[lang]
    return (
        f"Translate the following English text into {name}. "
        f"Output only the {name} translation, with no explanation.\n\n{text}"
    )


def translation_messages(lang: str, text: str) -> list[Message]:
    return [{"role": "user", "content": translation_prompt(lang, text)}]


def teacher_requests(lang: str, sources: Iterable[str]) -> list[TeacherRequest]:
    """One teacher request per English source sentence, keyed by language and a hash of the source."""
    return [
        TeacherRequest(
            key=f"{lang}:{hashlib.sha1(src.encode('utf-8')).hexdigest()[:16]}",
            messages=translation_messages(lang, src),
            meta={"lang": lang, "source": src},
        )
        for src in dict.fromkeys(sources)
    ]


def eval_items(lang: str, n: int | None = None, *, split: str = "devtest", offset: int = 0) -> list[dict[str, Any]]:
    """English-to-``lang`` eval items from FLORES-200, each with its human reference."""
    src = flores_lines("eng_Latn", split)
    ref = flores_lines(lang, split)
    if len(src) != len(ref):
        raise ValueError(f"FLORES {lang}.{split} has {len(ref)} lines, English has {len(src)}")
    end = len(src) if n is None else min(offset + n, len(src))
    return [
        {
            "id": f"{lang}:{split}:{i}",
            "lang": lang,
            "source": src[i],
            "reference": ref[i],
            "messages": translation_messages(lang, src[i]),
        }
        for i in range(offset, end)
    ]


def clean(text: str) -> str:
    return (text or "").strip().strip('"').strip()


def chrf(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus chrF++ (character n-grams plus word bigrams), 0 to 100."""
    return sacrebleu.corpus_chrf([clean(h) for h in hypotheses], [list(references)], word_order=2).score


def script_rate(texts: Sequence[str], lang: str) -> float | None:
    """Share of outputs whose letters are mostly in ``lang``'s script; ``None`` for Latin-script languages."""
    script = SCRIPTS.get(lang)
    if script is None or not texts:
        return None
    return sum(script_fraction(clean(t), script) >= 0.5 for t in texts) / len(texts)


def translation_metrics(items: Sequence[dict[str, Any]], outputs: Sequence[str]) -> dict[str, float]:
    """Corpus chrF++ against the human references, plus the script rate where the language has its own script."""
    if not items:
        return {}
    metrics = {"chrf": chrf(outputs, [it["reference"] for it in items])}
    rate = script_rate(outputs, items[0]["lang"])
    if rate is not None:
        metrics["script_rate"] = rate
    return metrics


def english_sources(*, flores_dev: bool = True, extra: Iterable[str] = ()) -> list[str]:
    """English training sources: FLORES-200 ``dev`` plus ``extra``, deduplicated, never a ``devtest`` sentence."""
    devtest = set(flores_lines("eng_Latn", "devtest"))
    candidates = [*(flores_lines("eng_Latn", "dev") if flores_dev else ()), *extra]
    return [s for s in dict.fromkeys(candidates) if s and s not in devtest]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")
_MARKUP = re.compile(r"[\[\]{}|=<>]|https?:")


def sentences_from_dataset(
    name: str = "wikimedia/wikipedia",
    config: str | None = "20231101.en",
    *,
    split: str = "train",
    column: str = "text",
    n: int = 3000,
    min_words: int = 10,
    max_words: int = 40,
    max_rows: int = 3000,
    seed: int = 0,
) -> list[str]:
    """Clean English sentences streamed from a text dataset, for extra teacher sources.

    Skips short paragraphs (headings, captions, lists) and sentences with markup or URLs. Streaming
    reads whole parquet shards, so the first call downloads a few hundred megabytes; keep
    ``max_rows`` small.
    """
    import random

    from datasets import load_dataset

    stream = load_dataset(name, config, split=split, streaming=True)
    seen: dict[str, None] = {}
    for i, row in enumerate(stream):
        if i >= max_rows or len(seen) >= 4 * n:
            break
        for para in str(row[column]).split("\n"):
            if len(para) < 200:
                continue
            for sent in _SENTENCE_SPLIT.split(para):
                sent = sent.strip()
                words = len(sent.split())
                if min_words <= words <= max_words and sent[-1:] in '.!?"”' and not _MARKUP.search(sent):
                    seen.setdefault(sent)
    sentences = list(seen)
    random.Random(seed).shuffle(sentences)
    return sentences[:n]


def cached_sentences(path: Path, n: int, **kwargs: Any) -> list[str]:
    """:func:`sentences_from_dataset`, saved to ``path`` on first use so every run sees the same sentences.

    A saved list with at least ``n`` sentences is reused (its first ``n``); a shorter one is rebuilt.
    """
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if len(saved) >= n:
            return saved[:n]
    sentences = sentences_from_dataset(n=n, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    return sentences


def translation_rows(
    lang: str,
    sources: Iterable[str],
    teacher: TeacherConfig,
    cache_dir: Path,
    *,
    min_script_fraction: float = 0.7,
) -> list[dict[str, Any]]:
    """Teacher translations of ``sources`` into ``lang`` as chat SFT rows.

    Drops failed, truncated and empty outputs, echoes of the English, and, for languages with their
    own script, outputs that aren't mostly in that script. The teacher cache is
    ``cache_dir / f"{lang}.jsonl"``, so reruns only call the teacher for new sources.
    """
    requests = teacher_requests(lang, sources)
    outputs, spend = generate(requests, teacher, cache_dir / f"{lang}.jsonl")
    for output in outputs.values():
        output.content = clean(output.content)
    filters = [not_an_echo]
    if lang in SCRIPTS:
        filters.append(in_script(SCRIPTS[lang], min_script_fraction))
    rows, dropped = sft_rows(requests, outputs, filters=filters, keep_reasoning=False)
    print(
        f"{LANGUAGES[lang]}: {len(rows)} rows from {len(requests)} sources; dropped {dict(dropped)}; "
        f"this run's teacher spend ${spend.usd(teacher.price_per_million):.2f}",
        flush=True,
    )
    return rows


def translation_eval_set(
    lang: str, n: int | None = 300, *, name: str | None = None, every_eval: bool = True
) -> EvalSet:
    """English-to-``lang`` FLORES-200 devtest sentences as an :class:`EvalSet`, scored with chrF++."""
    return EvalSet(name=name or lang, items=eval_items(lang, n), metrics=translation_metrics, every_eval=every_eval)
