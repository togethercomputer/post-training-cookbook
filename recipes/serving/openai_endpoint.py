"""Serve an existing training session to OpenAI-compatible HTTP clients."""

from __future__ import annotations

import argparse
import asyncio
import os

import uvicorn
from dotenv import load_dotenv
from together.post_training import SessionClient

from post_training_cookbook.codecs import get_codec
from post_training_cookbook.serving import create_app, validate_bind


async def serve(args: argparse.Namespace) -> None:
    token = validate_bind(args.host, args.api_key)
    session = await SessionClient.attach_async(
        session_id=args.session_id,
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url=os.environ.get("TOGETHER_BASE_URL"),
    )
    try:
        if not session.has_generator:
            raise ValueError("The session's model resource has no generator replicas")
        info = await session.retrieve_async()
        codec = await asyncio.to_thread(
            get_codec,
            info.base_model,
            hf_token=os.environ.get("HF_TOKEN"),
            enable_thinking=args.enable_thinking,
            preserve_thinking=args.preserve_thinking,
            thinking_effort=args.thinking_effort,
        )
        app = create_app(
            session.generator.sample_async,
            codec,
            model=info.base_model,
            session_id=session.session_id,
            max_model_length=args.max_model_length,
            default_max_tokens=args.default_max_tokens,
            max_concurrent=args.max_concurrent,
            sample_timeout=args.sample_timeout,
            api_key=token,
        )
        await uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port)).serve()
    finally:
        # This handle borrows the training session; stopping it would interrupt training.
        await session.detach_async()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--max-model-length", required=True, type=int, help="Context limit of the model resource")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("ROLLOUT_API_KEY"))
    parser.add_argument("--max-concurrent", type=int, default=256)
    parser.add_argument("--default-max-tokens", type=int, default=1024)
    parser.add_argument("--sample-timeout", type=float, default=7200)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preserve-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking-effort")
    args = parser.parse_args()
    validate_bind(args.host, args.api_key)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
