"""Dataset registry for GRPO training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from datasets import Dataset, load_dataset

from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.datasets.math_grader import boxed_math_reward
from post_training_cookbook.datasets.rewards import (
    game24_accuracy,
    game24_reward,
    gsm8k_reward,
    normalize_numeric,
)


@dataclass
class DatasetConfig:
    hf_name: str
    hf_config: str | None
    hf_split: str
    format_question: Callable[[dict], str]
    extract_target: Callable[[dict], Any]
    compute_reward: Callable[[str, Any], float]
    compute_metric: Callable[[str, Any], float] | None = None
    filter_fn: Callable[[dict], bool] | None = None
    few_shot_messages: list[dict[str, str]] = field(default_factory=list)


class PromptExample(NamedTuple):
    question: str
    target: Any
    prompt_tokens: list[int]


_BOXED_SUFFIX = " Show your work step by step and provide the final answer inside \\boxed{}."
_GSM8K_SUFFIX = (
    " Solve the problem step by step. End the final line with #### followed by the final"
    " numerical answer, for example #### 42."
)

DATASETS: dict[str, DatasetConfig] = {
    "deepmath": DatasetConfig(
        hf_name="zwhe99/DeepMath-103K",
        hf_config="default",
        hf_split="train",
        format_question=lambda row: row["question"] + _BOXED_SUFFIX,
        extract_target=lambda row: row["final_answer"],
        compute_reward=boxed_math_reward,
        few_shot_messages=[
            {
                "role": "user",
                "content": "Find the value of $2^3 + 3^2$." + _BOXED_SUFFIX,
            },
            {
                "role": "assistant",
                "content": ("We compute each term: $2^3 = 8$ and $3^2 = 9$. Adding them: $8 + 9 = 17$. \\boxed{17}"),
            },
        ],
    ),
    "deepmath-lv5": DatasetConfig(
        hf_name="zwhe99/DeepMath-103K",
        hf_config="default",
        hf_split="train",
        format_question=lambda row: row["question"] + _BOXED_SUFFIX,
        extract_target=lambda row: row["final_answer"],
        compute_reward=boxed_math_reward,
        filter_fn=lambda row: row["difficulty"] <= 5,
        few_shot_messages=[
            {
                "role": "user",
                "content": "Find the value of $2^3 + 3^2$." + _BOXED_SUFFIX,
            },
            {
                "role": "assistant",
                "content": ("We compute each term: $2^3 = 8$ and $3^2 = 9$. Adding them: $8 + 9 = 17$. \\boxed{17}"),
            },
        ],
    ),
    "gsm8k": DatasetConfig(
        hf_name="openai/gsm8k",
        hf_config="main",
        hf_split="train",
        format_question=lambda row: row["question"] + _GSM8K_SUFFIX,
        extract_target=lambda row: normalize_numeric(row["answer"].split("####")[-1].strip()),
        compute_reward=gsm8k_reward,
        few_shot_messages=[
            {
                "role": "user",
                "content": (
                    "A store sold 3 packs of pencils with 4 pencils in each pack, then sold 2"
                    " more single pencils. How many pencils were sold in total?" + _GSM8K_SUFFIX
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Three packs of 4 pencils is 3 * 4 = 12 pencils. Adding 2 more pencils gives 12 + 2 = 14.\n#### 14"
                ),
            },
        ],
    ),
    "game24": DatasetConfig(
        hf_name="nlile/24-game",
        hf_config="default",
        hf_split="train",
        filter_fn=lambda row: row["solvable"],
        format_question=lambda row: (
            f"Use the numbers {row['numbers']} with +, -, *, / and parentheses to make 24. "
            "Each number must be used exactly once. Think briefly, then end with a single "
            "boxed arithmetic expression using only those numbers. The final line must be "
            "\\boxed{<expression>} with no words inside the box."
        ),
        extract_target=lambda row: row["numbers"],
        compute_reward=game24_reward,
        compute_metric=game24_accuracy,
        few_shot_messages=[
            {
                "role": "user",
                "content": (
                    "Use the numbers [1, 3, 4, 6] with +, -, *, / and parentheses to make 24. "
                    "Each number must be used exactly once. Think briefly, then end with a single "
                    "boxed arithmetic expression using only those numbers. The final line must be "
                    "\\boxed{<expression>} with no words inside the box."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "We want an expression that becomes 24 while using each number once. "
                    "Since 1 - 3/4 = 1/4, dividing 6 by 1/4 gives 24.\n"
                    "\\boxed{6/(1-3/4)}"
                ),
            },
        ],
    ),
}


def load_grpo_dataset(
    cfg: DatasetConfig,
    num_train: int,
    num_val: int = 0,
    seed: int = 0,
) -> tuple[Dataset, Dataset | None]:
    if cfg.hf_config:
        raw = load_dataset(cfg.hf_name, cfg.hf_config)
    else:
        raw = load_dataset(cfg.hf_name)

    ds = raw[cfg.hf_split]
    if cfg.filter_fn:
        ds = ds.filter(cfg.filter_fn)
    ds = ds.shuffle(seed=seed)

    train = ds.select(range(min(num_train, len(ds))))
    val = None
    if num_val > 0:
        val_start = min(num_train, len(ds))
        val_end = min(val_start + num_val, len(ds))
        if val_end > val_start:
            val = ds.select(range(val_start, val_end))
    return train, val


def build_prompt_example(
    row: dict,
    cfg: DatasetConfig,
    codec: MessageCodec,
    *,
    max_prompt_length: int,
    max_model_length: int,
    min_response_tokens: int = 1,
) -> PromptExample:
    question = cfg.format_question(row)
    target = cfg.extract_target(row)
    messages = list(cfg.few_shot_messages) + [
        {"role": "user", "content": question},
    ]
    prompt_tokens = codec.encode_messages(messages, add_generation_prompt=True)
    if len(prompt_tokens) > max_prompt_length:
        raise ValueError(f"Prompt length {len(prompt_tokens)} exceeds max prompt length {max_prompt_length}")
    if max_model_length - len(prompt_tokens) < min_response_tokens:
        raise ValueError(
            f"Prompt leaves insufficient room for a response: "
            f"prompt_length={len(prompt_tokens)}, max_model_length={max_model_length}, "
            f"min_response_tokens={min_response_tokens}"
        )
    return PromptExample(question=question, target=target, prompt_tokens=prompt_tokens)


def filter_by_prompt_length(
    dataset,
    ds_cfg: DatasetConfig,
    codec: MessageCodec,
    max_prompt_length: int,
    max_model_length: int,
    min_response_tokens: int = 1,
) -> Any:
    keep = []
    dropped_for_prompt_length = 0
    dropped_for_budget = 0
    for i, row in enumerate(dataset):
        try:
            build_prompt_example(
                row,
                ds_cfg,
                codec,
                max_prompt_length=max_prompt_length,
                max_model_length=max_model_length,
                min_response_tokens=min_response_tokens,
            )
            keep.append(i)
        except ValueError as exc:
            error_text = str(exc)
            if "exceeds max prompt length" in error_text:
                dropped_for_prompt_length += 1
            elif "insufficient room for a response" in error_text:
                dropped_for_budget += 1
            else:
                raise
    filtered = dataset.select(keep)
    dropped = len(dataset) - len(filtered)
    if dropped:
        print(
            f"  Filtered {dropped}/{len(dataset)} rows "
            f"({dropped_for_prompt_length} prompt too long, "
            f"{dropped_for_budget} insufficient response budget)"
        )
    return filtered
