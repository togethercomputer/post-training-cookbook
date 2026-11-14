"""Multi-teacher on-policy distillation (MOPD) of three GRPO experts into one student, on one model resource.

A single GRPO run on several domains lets them compete for the same updates. MOPD trains one
expert per domain instead, then distills them into one student: the student answers prompts from
every domain, and each answer is scored token by token by the expert for that prompt's domain.
Every expert, teacher and the student is a LoRA session on the same Qwen3.8-27B resource.

1. Experts: one GRPO expert each for math (DeepMath-103K), instruction following (AllenAI
   RLVR-IFEval) and Game of 24, trained concurrently, plus a GRPO baseline on all three domains
   pooled. Each saves a training checkpoint every 10 steps and at the end.
2. Fusion: a fresh student plus one frozen teacher session per domain, started from the experts'
   final training checkpoints. Each step the student samples 4 answers to 16 prompts per domain,
   the routed teacher scores every token with ``prompt_logprobs``, and the student trains on the
   per-token reverse KL with the importance-sampling loss.
3. Evaluation: every model on every domain's held-out test split (500 / 464 / 362 prompts), greedy.

Our run: the student scored 77.8 / 92.9 / 89.5 on math / instruction following / Game of 24,
within 1.5 points of each expert on its own domain (79.0 / 94.4 / 90.3), against 77.8 / 89.0 /
90.6 for the pooled GRPO baseline and 54.2 / 80.6 / 72.1 for the base model. The IFEval expert
learned to game one constraint ("the letter e should appear 3 times" answered with ``e e e``) and
the student inherited it; read your experts' outputs before you distill them. Experts and
baseline took 132 minutes, fusion 66 and evaluation 9: about $250 at $72/h.

Requires TOGETHER_API_KEY. The resource is stopped when the script exits, even on error.

Run: uv run python recipes/mopd/math_ifeval_game24.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from together.post_training import LoraConfig

from post_training_cookbook.codecs import get_codec
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.mopd.config import MOPDConfig
from post_training_cookbook.mopd.domains import default_domains, load_splits
from post_training_cookbook.mopd.evaluate import build_eval_items, evaluate_checkpoints
from post_training_cookbook.mopd.experts import MIXED, expert_configs, train_experts
from post_training_cookbook.mopd.train import train
from post_training_cookbook.multi_lora import model_resource

MODEL = "Qwen/Qwen3.8-27B"
RUN_PREFIX = "mopd"
# Teachers are started from the experts' checkpoints, and a session doesn't inherit its LoRA
# settings from a checkpoint, so experts and teachers must share these.
LORA_RANK, LORA_ALPHA = 32, 64
MAX_MODEL_LENGTH, MAX_PROMPT_LENGTH, MAX_SAMPLE_TOKENS = 4096, 1024, 2048

# math (DeepMath-103K), ifeval (RLVR-IFEval) and game24, zero-shot, with disjoint train/test splits.
DOMAINS = default_domains()

expert = GRPOConfig(
    model=MODEL,
    num_steps=50,
    batch_size=16,
    group_size=8,
    learning_rate=5e-5,
    temperature=1.0,
    # Re-draw prompts whose 8 answers all scored the same: 70% of the base model's IFEval groups did.
    max_sampling_attempts=2,
    max_model_length=MAX_MODEL_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_sample_tokens=MAX_SAMPLE_TOKENS,
    # Thinking off in every stage keeps answers short; experts, student and evaluation must match.
    enable_thinking=False,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    num_val_examples=100,
    validation_interval=10,
    # A stuck operation raises after 20 min instead of waiting the 2 h default.
    timeout=1200.0,
    wandb_project=None,
)


def student_config(model_resources_id: str, teacher_checkpoints: dict[str, str]) -> MOPDConfig:
    return MOPDConfig(
        model_resources_id=model_resources_id,
        teacher_checkpoints=teacher_checkpoints,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        teacher_lora_rank=LORA_RANK,
        teacher_lora_alpha=LORA_ALPHA,
        num_steps=50,
        prompts_per_domain=16,
        samples_per_prompt=4,
        learning_rate=5e-5,
        # "domain" gives each domain a third of the loss weight; in our run it made no difference.
        balance="token",
        max_model_length=MAX_MODEL_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_sample_tokens=MAX_SAMPLE_TOKENS,
        enable_thinking=False,
        eval_interval=10,
        timeout=1200.0,
        wandb_run_name=f"{RUN_PREFIX}-student",
    )


def evaluate(resources, models: dict[str, str | None]) -> dict:
    """Score every model on each domain's full test split, up to 4 sessions at a time."""
    codec = get_codec(MODEL, enable_thinking=False)
    items, skipped = {}, {}
    for domain in DOMAINS:
        _, test = load_splits(domain)
        items[domain.name], skipped[domain.name] = build_eval_items(
            domain, test, codec, max_prompt_length=MAX_PROMPT_LENGTH, max_model_length=MAX_MODEL_LENGTH
        )
    return evaluate_checkpoints(
        resources,
        models,
        DOMAINS,
        items,
        codec,
        lora_config=LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA),
        out_dir=Path("runs") / f"{RUN_PREFIX}-eval",
        max_sample_tokens=MAX_SAMPLE_TOKENS,
        skipped=skipped,
    )


if __name__ == "__main__":
    load_dotenv(override=True)
    # One generator replica is enough: the sessions' training steps queue on the shared trainer,
    # and that, not sampling, sets the pace. On exit the resource is stopped with force=True.
    with model_resource(model=MODEL, num_generator_replicas=1) as resources:
        rid = resources.model_resources_id

        # 1. Experts and the pooled baseline, one session each, concurrently.
        configs = expert_configs(
            DOMAINS, expert, model_resources_id=rid, run_prefix=RUN_PREFIX, include_mixed=True, checkpoint_interval=10
        )
        experts = train_experts(configs)
        failed = {name: run.error for name, run in experts.items() if run.error or not run.training_checkpoint_id}
        if failed:
            raise SystemExit(f"expert runs failed: {failed}")
        teachers = {domain.name: experts[domain.name].training_checkpoint_id for domain in DOMAINS}

        # 2. Fusion: the student plus one frozen teacher session per domain.
        student = train(student_config(rid, teachers), DOMAINS)

        # 3. Every model on every domain's test split.
        models = {"base": None, **teachers, MIXED: experts[MIXED].training_checkpoint_id}
        models["student"] = student.training_checkpoint_id
        summaries = evaluate(resources, models)

    (Path("runs") / f"{RUN_PREFIX}-eval" / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\n{'model':<10}" + "".join(f"{d.name:>10}" for d in DOMAINS))
    for label, by_domain in summaries.items():
        print(f"{label:<10}" + "".join(f"{by_domain[d.name]['accuracy']:>10.3f}" for d in DOMAINS))
    print(f"Student run artifacts: {student.run_dir}; inference checkpoint: {student.inference_checkpoint_id}")
