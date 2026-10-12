#!/usr/bin/env python3
"""Prepare a LoRA adapter for FP4 inference and print the prepared model to deploy.

Usage:
    python prepare_for_fp4.py <adapter_id> [--revision-id rv_...] [--calibration-file file-...] [--yes]

Example:
    ckpt = session.create_inference_checkpoint()
    # python prepare_for_fp4.py <ckpt.registration.adapter.id>

This script:
  1. Estimates the job's price and running time, which also checks that the adapter is eligible
  2. Starts the job (only with --yes, since every job is billed)
  3. Polls until the job reaches a terminal status and prints its events on failure
  4. Prints the prepared model ID and the command to deploy it

An adapter whose base model isn't served in FP4 is rejected at the estimate: deploy it with
`tg beta endpoints deploy <adapter_id> --merge` instead.

Requires: TOGETHER_API_KEY for the project that owns the adapter
"""

from __future__ import annotations

import argparse
import sys
import time

from together import BadRequestError, Together

_TERMINAL_STATUSES = {"completed", "cancelled", "error", "user_error"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a LoRA adapter for FP4 inference")
    parser.add_argument("adapter_id", help="Adapter registry ID (ml_...) from checkpoint.registration.adapter")
    parser.add_argument("--revision-id", default=None, help="Adapter revision (rv_...); default: latest")
    parser.add_argument(
        "--calibration-file", default=None, help="Uploaded calibration JSONL (file-...), purpose 'calibration'"
    )
    parser.add_argument("--yes", action="store_true", help="Start the billed job after printing the estimate")
    parser.add_argument("--poll-seconds", type=int, default=300, help="Seconds between status checks (default: 300)")
    args = parser.parse_args()

    inputs = {"adapter_object_id": args.adapter_id}
    if args.revision_id is not None:
        inputs["adapter_revision_id"] = args.revision_id
    if args.calibration_file is not None:
        inputs["calibration_file_id"] = args.calibration_file

    client = Together()
    fp4 = client.post_training.prepare_for_fp4_inference

    try:
        estimate = fp4.estimate_cost(inputs=inputs)
    except BadRequestError as exc:
        print(f"ERROR: the adapter can't be prepared: {exc.message}", file=sys.stderr)
        sys.exit(1)
    print(f"Estimated price ${estimate.price_usd:.2f}, about {estimate.time_hours:.1f}h excluding queue time")
    if not estimate.allowed_to_proceed:
        print(f"ERROR: the price exceeds your credit limit (${estimate.credit_limit:.2f})", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        print("Re-run with --yes to start the job.")
        return

    job = fp4.create(inputs=inputs)
    print(f"Started {job.id}")
    while job.status not in _TERMINAL_STATUSES:
        time.sleep(args.poll_seconds)
        job = fp4.retrieve(job.id)
        print(f"  {job.status}")

    if job.status != "completed":
        for event in fp4.list_events(job.id).data:
            print(f"  {event.created_at} {event.type} {event.message}", file=sys.stderr)
        sys.exit(1)

    model_id = job.results.api_model_object_id
    print(f"Prepared model: {model_id} (revision {job.results.api_model_revision_id})")
    print("Deploy it with a config whose adapter_mode is disabled:")
    print(f"  tg beta models configs {model_id}")
    print(f"  tg beta endpoints deploy {model_id} --endpoint <name> --config <config_id> --project <project_id>")


if __name__ == "__main__":
    main()
