"""Distill Kimi-K3.1-Pro's English-to-Lao translation into Qwen3.8-27B without losing its Thai.

Asked to translate into Lao, Qwen3.8-27B writes Thai, Lao's closest relative: 5.5 chrF++ on
FLORES-200 devtest, with 11% of outputs in Lao script. Kimi-K3.1-Pro scores 47.5. This recipe has
Kimi-K3.1-Pro translate about 4K English sentences into Lao and 1K into Thai, then fine-tunes a
LoRA adapter on those translations (off-policy, sequence-level distillation: the teacher's text is
the SFT target).

1. Sources: the 997 FLORES-200 dev sentences plus 3,000 English Wikipedia sentences for Lao, and
   1,000 different Wikipedia sentences for Thai. No devtest sentence is used for training.
2. Teacher: Kimi-K3.1-Pro on serverless with thinking off, cached under ``_dev/distillation/teacher``.
3. Filters: drop failed, truncated and empty outputs, echoes of the English, and outputs whose
   letters are under 70% in the target script.
4. Mix in the Thai rows. Trained on Lao alone, the student starts answering some Thai prompts in
   Lao; Thai chrF++ slipped from 46.7 to 44.3 over three epochs.
5. Train: LoRA rank 64, 2 epochs (about 600 steps at batch 16). Lao and Thai (300 sentences each)
   are scored on the generator every 50 steps, and the full 1,012-sentence Lao devtest at the
   start and end.

Our run reached 44.3 chrF++ on Lao (93% of the teacher), 100% Lao script, and 47.8 on Thai. It
took about 46 minutes of GPU time ($55 at $72/h) and $9 of Kimi-K3.1-Pro calls.

Requires TOGETHER_API_KEY. Wikipedia sentences are streamed from Hugging Face on the first run.

Run: uv run python recipes/distillation/lao_from_kimi_k31_pro.py
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from post_training_cookbook.distillation.config import DistillConfig
from post_training_cookbook.distillation.data import mix, write_jsonl
from post_training_cookbook.distillation.evaluate import EvalSet, score_answered, teacher_outputs
from post_training_cookbook.distillation.teacher import TeacherConfig
from post_training_cookbook.distillation.train import train
from post_training_cookbook.distillation.translation import (
    cached_sentences,
    english_sources,
    translation_eval_set,
    translation_rows,
)

CACHE_DIR = Path("_dev/distillation")
# Shared with multilingual_from_kimi_k31_pro.py: Lao uses the first 3,000, Thai the last 1,000.
NUM_WIKI = 4000
NUM_WIKI_LAO = 3000

teacher = TeacherConfig(
    model="moonshotai/Kimi-K3.1-Pro",
    # Translation needs no reasoning. With thinking on, Kimi-K3.1-Pro scored lower on Lao and
    # used about 8x the tokens.
    chat_template_kwargs={"thinking": False},
    max_tokens=1024,
    concurrency=8,
    # List keys from several projects to add their rate limits together.
    api_key_envs=("TOGETHER_API_KEY",),
    price_per_million=(2.0, 10.0),
    budget_usd=25.0,
)
# Cache keys don't include the model, so each teacher gets its own cache directory.
TEACHER_CACHE = CACHE_DIR / "teacher" / teacher.model.split("/")[-1]

config = DistillConfig(
    model="Qwen/Qwen3.8-27B",
    # The teacher rows carry no reasoning, so render and train without a thinking block.
    enable_thinking=False,
    batch_size=16,
    num_epochs=2.0,
    learning_rate=1e-4,
    lora_rank=64,
    lora_alpha=128,
    max_length=2048,
    eval_interval=50,
    eval_max_tokens=1024,
    wandb_project=None,
)


def training_rows() -> list[dict[str, Any]]:
    """Kimi-K3.1-Pro's Lao and Thai translations as chat SFT rows, shuffled together."""
    wiki = cached_sentences(CACHE_DIR / "wiki_en_sentences.json", NUM_WIKI)
    lao_sources = english_sources(extra=wiki[:NUM_WIKI_LAO])
    thai_sources = english_sources(flores_dev=False, extra=wiki[NUM_WIKI_LAO:])
    lao = translation_rows("lao_Laoo", lao_sources, teacher, TEACHER_CACHE)
    thai = translation_rows("tha_Thai", thai_sources, teacher, TEACHER_CACHE)
    return mix(lao, thai, seed=config.seed)


def eval_sets() -> list[EvalSet]:
    return [
        translation_eval_set("lao_Laoo", 300, name="lao"),
        translation_eval_set("tha_Thai", 300, name="thai"),
        translation_eval_set("lao_Laoo", None, name="lao_full", every_eval=False),
    ]


if __name__ == "__main__":
    load_dotenv(override=True)
    rows = training_rows()
    write_jsonl(CACHE_DIR / "sft_lao_thai.jsonl", rows)
    sets = eval_sets()
    # The teacher's score on the same sentences is the target the student is chasing.
    teacher_scores = {
        s.name: score_answered(s, teacher_outputs(s.items, teacher, TEACHER_CACHE / "eval.jsonl")) for s in sets
    }

    result = train(config, rows, sets)

    print(f"\n{'eval set':<10} {'student':>8} {'teacher':>8}")
    for name, metrics in result.eval_metrics.items():
        print(f"{name:<10} {metrics['chrf']:8.1f} {teacher_scores[name].get('chrf', float('nan')):8.1f}")
    print(f"Run artifacts: {result.run_dir}; inference checkpoint: {result.inference_checkpoint}")
