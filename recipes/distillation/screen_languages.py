"""Screen for a distillation gap: score teacher and student on the same FLORES-200 sentences.

Distillation only pays off where the teacher is clearly better, so measure that first. Both
models translate the same 50 English devtest sentences into each candidate language, and the
script prints the chrF++ gap per language, largest first. Kimi-K3.1-Pro against Qwen3.8-27B gave
Lao +41.8 (the student writes Thai instead), Northern Sotho +15.7 and Zulu +13.3; the other eight
were under +10.

The teacher runs on serverless (about $0.50 for all languages) and is cached, so a rerun is free.
The student is scored on one post-training resource's generator, which takes a couple of minutes
once the resource is ready; the resource is stopped when the script exits.

Requires TOGETHER_API_KEY.

Run: uv run python recipes/distillation/screen_languages.py
"""

from pathlib import Path

from dotenv import load_dotenv

from post_training_cookbook.distillation.config import DistillConfig
from post_training_cookbook.distillation.evaluate import score_answered, teacher_outputs
from post_training_cookbook.distillation.teacher import TeacherConfig
from post_training_cookbook.distillation.train import evaluate_student
from post_training_cookbook.distillation.translation import LANGUAGES, translation_eval_set

CANDIDATES = [lang for lang in LANGUAGES if lang not in ("eng_Latn", "tha_Thai")]
NUM_SENTENCES = 50

teacher = TeacherConfig(
    model="moonshotai/Kimi-K3.1-Pro",
    # Translation needs no reasoning; with thinking off a call costs about $0.001.
    chat_template_kwargs={"thinking": False},
    max_tokens=1024,
    price_per_million=(2.0, 10.0),
    budget_usd=5.0,
)
# Shared with the other distillation recipes; one cache per teacher model.
TEACHER_CACHE = Path("_dev/distillation/teacher") / teacher.model.split("/")[-1]

student = DistillConfig(
    model="Qwen/Qwen3.8-27B",
    enable_thinking=False,
    eval_max_tokens=1024,
)

if __name__ == "__main__":
    load_dotenv(override=True)
    eval_sets = [translation_eval_set(lang, NUM_SENTENCES) for lang in CANDIDATES]

    teacher_scores = {
        s.name: score_answered(s, teacher_outputs(s.items, teacher, TEACHER_CACHE / "eval.jsonl")) for s in eval_sets
    }
    student_scores = evaluate_student(student, eval_sets)

    rows = sorted(
        ((lang, teacher_scores[lang].get("chrf", float("nan")), student_scores[lang]["chrf"]) for lang in CANDIDATES),
        key=lambda r: r[1] - r[2],
        reverse=True,
    )
    print(f"\n{'language':<20} {'teacher':>8} {'student':>8} {'gap':>7}")
    for lang, t, s in rows:
        print(f"{LANGUAGES[lang]:<20} {t:8.1f} {s:8.1f} {t - s:+7.1f}")
