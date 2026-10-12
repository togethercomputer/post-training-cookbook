"""Stop model resources to free their GPUs.

python scripts/multi_lora/clean_up.py <id> [<id> ...]   # stop specific resources
python scripts/multi_lora/clean_up.py --all             # stop every resource you own
"""

import sys

from fetch import all_resources
from together import Together


def stop(client: Together, resource_id: str) -> None:
    print(f"stopping {resource_id} ...", end=" ", flush=True)
    client.post_training.model_resources.stop(resource_id)
    print("done")


if __name__ == "__main__":
    client = Together()
    args = sys.argv[1:]

    if args == ["--all"]:
        ids = [r.id for r in all_resources(client)]
    elif args and not any(a.startswith("-") for a in args):
        ids = args
    else:
        print("Pass IDs to stop, or --all to stop everything above.")
        raise SystemExit

    for resource_id in ids:
        stop(client, resource_id)
