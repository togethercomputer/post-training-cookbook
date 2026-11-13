"""Distill Kimi-K3.1-Pro into one Qwen3.8-27B adapter for four low-resource languages, plus Thai.

The same recipe as ``lao_from_kimi_k31_pro.py``, applied to Lao, Northern Sotho and Zulu (the
three largest teacher leads in ``screen_languages.py``) plus Yoruba, the lowest-scoring Latin-script
language for both models. Each gets the 997 FLORES-200 dev sentences plus 1,000 Wikipedia sentences
(about 2K translations per language), and 1,000 Thai translations keep the student's Thai from
eroding. One LoRA adapter trains on all of it for 1.5 epochs (about 830 steps).

Our run, chrF++ on FLORES-200 devtest (300 sentences, all 1,012 for Lao), base -> distilled
(teacher): Lao 5.5 -> 41.8 (47.5), Northern Sotho 32.7 -> 48.0 (50.7), Yoruba 20.7 -> 25.7 (27.7),
Zulu 37.7 -> 39.2 (50.3), Thai 46.7 -> 48.0 (49.9). It took about 66 minutes of GPU time ($79) and
about $10 of Kimi-K3.1-Pro calls. Zulu closed only 12% of its gap, and 1-2% of its outputs fell into
repetition loops during training, so more Zulu data or a repetition penalty is the first thing to try.

Requires TOGETHER_API_KEY. Wikipedia sentences are streamed from Hugging Face on the first run.

Run: uv run python recipes/distillation/multilingual_from_kimi_k31_pro.py
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from post_training_cookbook.distillation.config import DistillConfig
from post_training_cookbook.distillation.data import mix, write_jsonl
from post_training_cookbook.distillation.evaluate import EvalSet
from post_training_cookbook.distillation.teacher import TeacherConfig
from post_training_cookbook.distillation.train import train
from post_training_cookbook.distillation.translation import (
    cached_sentences,
    english_sources,
    translation_eval_set,
    translation_rows,
)

CACHE_DIR = Path("_dev/distillation")
TARGETS = {"lao_Laoo": "lao", "nso_Latn": "nso", "zul_Latn": "zul", "yor_Latn": "yor"}
# Shared with lao_from_kimi_k31_pro.py, so the Lao and Thai translations come from its cache.
NUM_WIKI = 4000
NUM_WIKI_PER_LANGUAGE = 1000
THAI_WIKI = slice(3000, 4000)

teacher = TeacherConfig(
    model="moonshotai/Kimi-K3.1-Pro",
    chat_template_kwargs={"thinking": False},
    max_tokens=1024,
    concurrency=8,
    api_key_envs=("TOGETHER_API_KEY",),
    price_per_million=(2.0, 10.0),
    budget_usd=40.0,
)
TEACHER_CACHE = CACHE_DIR / "teacher" / teacher.model.split("/")[-1]

config = DistillConfig(
    model="Qwen/Qwen3.8-27B",
    enable_thinking=False,
    batch_size=16,
    num_epochs=1.5,
    learning_rate=1e-4,
    lora_rank=64,
    lora_alpha=128,
    max_length=2048,
    eval_interval=100,
    eval_max_tokens=1024,
    wandb_project=None,
)


def training_rows() -> list[dict[str, Any]]:
    """Translations into the four target languages and Thai as chat SFT rows, shuffled together."""
    wiki = cached_sentences(CACHE_DIR / "wiki_en_sentences.json", NUM_WIKI)
    # The target languages share their English sources; Thai gets its own.
    target_sources = english_sources(extra=wiki[:NUM_WIKI_PER_LANGUAGE])
    thai_sources = english_sources(flores_dev=False, extra=wiki[THAI_WIKI])
    groups = [translation_rows(lang, target_sources, teacher, TEACHER_CACHE) for lang in TARGETS]
    groups.append(translation_rows("tha_Thai", thai_sources, teacher, TEACHER_CACHE))
    return mix(*groups, seed=config.seed)


def eval_sets() -> list[EvalSet]:
    sets = [translation_eval_set(lang, 300, name=name) for lang, name in TARGETS.items()]
    sets.append(translation_eval_set("tha_Thai", 300, name="thai"))
    sets.append(translation_eval_set("lao_Laoo", None, name="lao_full", every_eval=False))
    return sets


if __name__ == "__main__":
    load_dotenv(override=True)
    rows = training_rows()
    write_jsonl(CACHE_DIR / "sft_multilingual.jsonl", rows)
    result = train(config, rows, eval_sets())
    for name, metrics in result.eval_metrics.items():
        print(f"{name:<10} chrF++ {metrics['chrf']:.1f}")
    print(f"Run artifacts: {result.run_dir}; inference checkpoint: {result.inference_checkpoint}")
