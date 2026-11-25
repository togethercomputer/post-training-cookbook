"""Provision a shared model resource and leave it running.

A *model resource* is the GPU pods plus base weights that training sessions
attach to. This script stands one up and prints its ID, so later runs
can attach their own LoRA-adapter sessions to it via ``MODEL_RESOURCES_ID``.

Billing stops when model_resources are shut down.

python scripts/multi_lora/launch.py [--model MODEL] [--replicas N]

Only the ID goes to stdout, so scripts can capture it:
``RID=$(python scripts/multi_lora/launch.py --model zai-org/GLM-5.3)``.
"""

import argparse
import sys
from contextlib import redirect_stdout

from together.post_training import ComputeConfig, ModelResourcesClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision a shared LoRA model resource.")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="base model to serve")
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="generator replicas; 0 for a trainer-only resource (SFT/DPO, no sampling)",
    )
    args = parser.parse_args()

    with redirect_stdout(sys.stderr):
        resource = ModelResourcesClient.create(
            base_model=args.model,
            lora_enabled=True,
            compute_config=ComputeConfig(num_generator_replicas=args.replicas),
        )
    rid = resource.model_resources_id

    print(rid)
    print(
        f"Model resource ready: {rid}\n"
        "List your resources:\n"
        "    python scripts/multi_lora/fetch.py\n"
        f'Attach a training run by setting model_resources_id="{rid}" and leaving model unset.\n'
        "Delete it when finished:\n"
        f"    python scripts/multi_lora/clean_up.py {rid}",
        file=sys.stderr,
    )
