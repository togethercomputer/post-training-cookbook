"""Provision a shared model resource and leave it running.

A *model resource* is the GPU pods plus base weights that training sessions
attach to. This script stands one up and prints its ID, so later runs
can attach their own LoRA-adapter sessions to it via ``MODEL_RESOURCES_ID``.

Billing stops when model_resources are shut down.
"""

from together.post_training import ComputeConfig, ModelResourcesClient

MODEL = "Qwen/Qwen3.6-35B-A3B"
LORA_ENABLED = True
NUM_GENERATOR_REPLICAS = 1  # set 0 for a trainer-only resource (SFT/DPO, no sampling)

if __name__ == "__main__":
    resource = ModelResourcesClient.create(
        base_model=MODEL,
        lora_enabled=LORA_ENABLED,
        compute_config=ComputeConfig(num_generator_replicas=NUM_GENERATOR_REPLICAS),
    )
    rid = resource.model_resources_id

    print(f"\nModel resource ready: {rid}")
    print("List your resources:")
    print("    python scripts/multi_lora/fetch.py")
    print("Attach a training run:")
    print(f"    MODEL_RESOURCES_ID={rid} python recipes/grpo/deepmath_shared.py")
    print("Delete it when finished:")
    print(f"    python scripts/multi_lora/clean_up.py {rid}")
