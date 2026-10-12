"""Estimate a model resource's hourly cost without provisioning it.

uv run python scripts/estimate_model_resource_cost.py
"""

from together import Together

BASE_MODEL = "zai-org/GLM-5.3"
GPU_TYPE = "B200-SXM"
LORA_ENABLED = True
NUM_GENERATOR_REPLICAS = 1


def main() -> None:
    with Together() as client:
        estimate = client.post_training.model_resources.estimate_cost(
            base_model=BASE_MODEL,
            compute_config={"gpu_type": GPU_TYPE, "num_generator_replicas": NUM_GENERATOR_REPLICAS},
            lora_enabled=LORA_ENABLED,
        )
        print(f"Estimated cost for {BASE_MODEL} on {GPU_TYPE}: {estimate.price_per_hour:.2f} {estimate.currency}/hour")


if __name__ == "__main__":
    main()
