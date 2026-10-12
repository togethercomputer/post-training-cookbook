#!/usr/bin/env python3
"""Scaffold a new GRPO dataset: reward function, registry entry, and recipe.

Usage:
    python scaffold_dataset.py <dataset_key> <hf_repo> [--question-col question] [--answer-col answer]

Example:
    python scaffold_dataset.py trivia my-org/trivia-10k --question-col question --answer-col answer

Generates three files in the current directory:
    reward_<key>.py           — reward function stub
    register_<key>.py         — DatasetConfig entry to paste into registry.py
    recipe_<key>.py           — ready-to-run GRPO recipe
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new GRPO dataset")
    parser.add_argument("key", help="Dataset key for the DATASETS registry (e.g. 'trivia')")
    parser.add_argument("hf_repo", help="HuggingFace dataset repo ID (e.g. 'my-org/trivia-10k')")
    parser.add_argument("--question-col", default="question", help="Column name for the question field")
    parser.add_argument("--answer-col", default="answer", help="Column name for the answer/target field")
    parser.add_argument("--hf-config", default=None, help="HuggingFace dataset config name")
    parser.add_argument("--hf-split", default="train", help="Dataset split")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B", help="Base model for the recipe")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    key = args.key
    reward_fn_name = f"{key}_reward"
    hf_config_repr = f'"{args.hf_config}"' if args.hf_config else "None"

    # ---- Reward function ----
    reward_code = textwrap.dedent(f'''\
        """Reward function for {key} dataset."""

        from __future__ import annotations

        import re
        import string


        def normalize_answer(text: str) -> str:
            """Lowercase, strip articles, punctuation, and extra whitespace."""
            text = text.lower().strip()
            text = re.sub(r"\\b(a|an|the)\\b", " ", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub(r"\\s+", " ", text).strip()
            return text


        def extract_answer(text: str) -> str | None:
            """Extract the answer from model output.

            Looks for ``#### <answer>`` first, then falls back to the last
            non-empty line.
            """
            match = re.search(r"####\\s*(.+)", text)
            if match:
                return match.group(1).strip()
            lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
            return lines[-1] if lines else None


        def {reward_fn_name}(prediction: str, target: str) -> float:
            """Return 1.0 if the predicted answer matches the target, else 0.0.

            Both sides are normalised so minor formatting differences are tolerated.
            """
            pred = extract_answer(prediction)
            if pred is None or not target:
                return 0.0
            return 1.0 if normalize_answer(pred) == normalize_answer(target) else 0.0
    ''')
    reward_path = out / f"reward_{key}.py"
    reward_path.write_text(reward_code)
    print(f"  Created {reward_path}")

    # ---- Registry entry ----
    register_code = textwrap.dedent(f'''\
        """Dataset registration entry for {key}.

        To integrate:
          1. Copy the reward function from reward_{key}.py into
             post_training_cookbook/datasets/rewards.py
          2. Add the import and DATASETS entry below into
             post_training_cookbook/datasets/registry.py
        """

        from post_training_cookbook.datasets.registry import DATASETS, DatasetConfig
        from post_training_cookbook.datasets.rewards import {reward_fn_name}

        _SUFFIX = (
            " Think step by step, then give the final answer on the last line"
            " preceded by ####. For example: #### <your answer>"
        )

        DATASETS["{key}"] = DatasetConfig(
            hf_name="{args.hf_repo}",
            hf_config={hf_config_repr},
            hf_split="{args.hf_split}",
            format_question=lambda row: row["{args.question_col}"] + _SUFFIX,
            extract_target=lambda row: row["{args.answer_col}"],
            compute_reward={reward_fn_name},
            few_shot_messages=[
                {{
                    "role": "user",
                    "content": "What is the capital of France?" + _SUFFIX,
                }},
                {{
                    "role": "assistant",
                    "content": (
                        "France is a country in Western Europe. Its capital"
                        " city is Paris.\\n#### Paris"
                    ),
                }},
            ],
        )
    ''')
    register_path = out / f"register_{key}.py"
    register_path.write_text(register_code)
    print(f"  Created {register_path}")

    # ---- Recipe ----
    recipe_code = textwrap.dedent(f'''\
        """GRPO on {key} — edit config values below and run."""

        from post_training_cookbook.grpo.config import GRPOConfig
        from post_training_cookbook.grpo.train import train_single_turn

        config = GRPOConfig(
            model="{args.model}",
            dataset="{key}",
            num_examples=5000,
            num_val_examples=200,
            batch_size=16,
            group_size=8,
            num_epochs=3,
            learning_rate=1e-5,
            enable_thinking=False,
            max_prompt_length=512,
            max_model_length=1536,
            max_sample_tokens=512,
            # Overlong handling is controlled by overlong_penalty_factor alone
            # (set it to 0.0 to disable). overlong_penalty is a read-only
            # property, not a constructor argument.
            overlong_buffer_len=64,
            overlong_penalty_factor=0.05,
            warmup_ratio=0.05,
            min_lr_ratio=0.2,
            validation_interval=100,
            wandb_project="grpo-{key}",
        )

        if __name__ == "__main__":
            train_single_turn(config)
    ''')
    recipe_path = out / f"recipe_{key}.py"
    recipe_path.write_text(recipe_code)
    print(f"  Created {recipe_path}")

    print(f"\nDone! Three files created for dataset '{key}'.")
    print("Next steps:")
    print(f"  1. Edit reward_{key}.py to implement your scoring logic")
    print(f"  2. Edit register_{key}.py few-shot examples for your domain")
    print("  3. Copy reward function into post_training_cookbook/datasets/rewards.py")
    print("  4. Copy registry entry into post_training_cookbook/datasets/registry.py")
    print(f"  5. Move recipe_{key}.py to recipes/grpo/{key}.py")
    print(f"  6. Run: python recipes/grpo/{key}.py")


if __name__ == "__main__":
    main()
