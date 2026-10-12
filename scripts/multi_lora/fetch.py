"""List your model resources so you can attach to one or free its GPUs.

A *model resource* is the GPU pods plus base weights that training sessions
attach to; it bills until deleted. This prints every resource you own — ID,
base model, status, and age — alongside the commands to attach a run or tear
one down.

    python scripts/multi_lora/fetch.py
"""

from datetime import UTC, datetime

from together import Together


def all_resources(client: Together) -> list:
    """Page through every model resource the caller owns."""
    resources, cursor = [], None
    while True:
        page = client.post_training.model_resources.list(**({"after": cursor} if cursor else {}))
        resources.extend(page.data)
        if not page.meta.has_more:
            return resources
        cursor = page.meta.next_cursor


def _age(created_at: datetime) -> str:
    hours = (datetime.now(UTC) - created_at).total_seconds() / 3600
    return f"{hours:.1f}h" if hours < 24 else f"{hours / 24:.1f}d"


if __name__ == "__main__":
    resources = all_resources(Together())
    if not resources:
        print("No model resources. Provision one: python scripts/multi_lora/launch.py")
        raise SystemExit

    for r in resources:
        status = r.status.removeprefix("MODEL_RESOURCES_STATUS_")
        print(f"{r.id}  {r.base_model:<24}  {status:<8}  age={_age(r.created_at)}")

    print("\nAttach a training run to one:")
    print(f"    MODEL_RESOURCES_ID={resources[0].id} python recipes/grpo/deepmath_shared.py")
    print("Free GPUs by deleting resources:")
    print("    python scripts/multi_lora/clean_up.py <id> [<id> ...]")
