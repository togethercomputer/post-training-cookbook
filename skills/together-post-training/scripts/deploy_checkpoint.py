#!/usr/bin/env python3
"""Deploy an inference checkpoint on Dedicated Model Inference.

Usage:
    python deploy_checkpoint.py <model_id> <revision_id> [--config cr_...] [--timeout 3600]

Example:
    ckpt = session.create_inference_checkpoint()
    artifact = ckpt.registration.adapter  # full-weight: ckpt.registration.model
    # python deploy_checkpoint.py <artifact.id> <artifact.revision_id>

This script:
  1. Picks a serving config for the checkpoint's base model
  2. Creates an endpoint and a deployment pinned to the checkpoint revision
  3. Routes all endpoint traffic to that deployment
  4. Waits for the deployment to become ready
  5. Sends a test chat completion and prints the endpoint name

Requires: TOGETHER_API_KEY and TOGETHER_PROJECT_ID environment variables
"""

from __future__ import annotations

import argparse
import sys
import time

from together import APIStatusError, Together

_RETRYABLE_STATUSES = {404, 429, 500, 502, 503, 504}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy an inference checkpoint on Dedicated Model Inference")
    parser.add_argument("model_id", help="Registry model ID (ml_...) from checkpoint.registration")
    parser.add_argument("revision_id", help="Registry revision ID (rv_...) from checkpoint.registration")
    parser.add_argument("--endpoint-name", default=None, help="Endpoint name (default: derived from the model ID)")
    parser.add_argument("--config", default=None, help="Config ID (cr_...) when the base model has several")
    parser.add_argument(
        "--inactive-timeout",
        type=int,
        default=30,
        help="Minutes of idle time before auto-stop: 0 disables it, otherwise 30-1440 (default: 30)",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Max seconds to wait for the deployment (default: 3600)"
    )
    parser.add_argument("--test-prompt", default="What is 2 + 2?", help="Test prompt to send after deployment")
    args = parser.parse_args()

    client = Together()

    # ---- Pick a serving config ----
    model = client.models.retrieve(args.model_id)
    configs = client.models.configs.list(reference_model_id=model.base_model_id or model.id).data or []
    if args.config is not None:
        configs = [c for c in configs if c.id == args.config]
    if len(configs) != 1:
        print(f"ERROR: expected one config, found {[c.id for c in configs]}. Pass --config.", file=sys.stderr)
        sys.exit(1)
    config = configs[0]
    print(f"Using config: {config.id}")

    # ---- Create endpoint and deployment ----
    endpoint_name = args.endpoint_name or f"rl-{args.model_id.lower().replace('_', '-')}"
    endpoint = client.endpoints.create(name=endpoint_name)
    deployment = client.endpoints.deployments.create(
        endpoint.id,
        name=f"{endpoint_name}-{args.revision_id.lower().replace('_', '-')}",
        model=f"projects/{model.project_id}/models/{model.id}/revisions/{args.revision_id}",
        config=f"projects/{config.project_id}/configs/{config.id}",
        autoscaling={"min_replicas": 1, "max_replicas": 1},
        inactive_timeout=args.inactive_timeout,
    )
    client.endpoints.update(
        endpoint.id,
        traffic_split=[{"deployment_id": deployment.id, "weight": 1.0}],
        update_mask="trafficSplit",
    )
    print(f"Endpoint {endpoint.id}, deployment {deployment.id}")

    # ---- Wait for the deployment ----
    deadline = time.monotonic() + args.timeout
    while deployment.status.state != "DEPLOYMENT_STATE_READY":
        if deployment.status.state == "DEPLOYMENT_STATE_FAILED":
            print(f"ERROR: deployment failed: {deployment.status.message}", file=sys.stderr)
            sys.exit(1)
        if time.monotonic() > deadline:
            print(f"ERROR: deployment not ready within {args.timeout}s", file=sys.stderr)
            sys.exit(1)
        time.sleep(15)
        deployment = client.endpoints.deployments.retrieve(deployment.id, endpoint_id=endpoint.id)
        print(f"  state: {deployment.status.state}")

    # ---- Test inference ----
    # Routing can lag the READY state by a minute or two.
    print(f"\nSending test prompt: {args.test_prompt!r}")
    while True:
        try:
            response = client.chat.completions.create(
                model=endpoint.name,
                messages=[{"role": "user", "content": args.test_prompt}],
                max_tokens=256,
            )
            break
        except APIStatusError as exc:
            if exc.status_code not in _RETRYABLE_STATUSES or time.monotonic() > deadline:
                raise
            time.sleep(10)
    print(f"Response: {response.choices[0].message.content}")

    print("\n--- Summary ---")
    print(f"  Checkpoint: {args.model_id}@{args.revision_id}")
    print(f"  Endpoint:   {endpoint.name}")
    print(f"\nTo delete: tg endpoints rm {endpoint.id}")


if __name__ == "__main__":
    main()
