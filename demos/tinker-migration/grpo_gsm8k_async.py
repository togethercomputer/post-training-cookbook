"""GRPO on GSM8K — one-step overlap of sample with train."""

import argparse
import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

import datasets

# import tinker
import together.post_training.tinker as tinker
import wandb
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

_ = load_dotenv(override=True)

MAX_IN_FLIGHT_SAMPLES = 16

QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."

CONVO_PREFIX = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + QUESTION_SUFFIX,
    },
    {
        "role": "assistant",
        "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e"
        " 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
    },
]


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.8-27B"
    batch_size: int = 64
    group_size: int = 8
    max_steps: int = 30
    learning_rate: float = 4e-5
    lora_rank: int = 32
    max_tokens: int = 1024
    eval_every: int = 5
    eval_size: int = 64
    wandb_project: str | None = None


def reward(response: str, answer: str) -> float:
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def build_prompt(renderer: renderers.Renderer, question: str) -> tinker.types.ModelInput:
    convo = [*CONVO_PREFIX, {"role": "user", "content": question + QUESTION_SUFFIX}]
    return renderer.build_generation_prompt(convo)


async def sample_groups(
    sampling_client: tinker.SamplingClient,
    prompts: list[tinker.types.ModelInput],
    sampling_params: tinker.types.SamplingParams,
    group_size: int,
    max_in_flight: int = MAX_IN_FLIGHT_SAMPLES,
) -> list[tinker.types.SampleResponse]:
    """Fan out sample_async over prompts; sample_async returns the final SampleResponse.

    Each in-flight request polls its own operation, so an unbounded fan-out over a large
    batch turns into a request storm against one session; the semaphore caps it.
    """
    limit = asyncio.Semaphore(max_in_flight)

    async def sample(prompt: tinker.types.ModelInput) -> tinker.types.SampleResponse:
        async with limit:
            return await sampling_client.sample_async(
                prompt=prompt, num_samples=group_size, sampling_params=sampling_params
            )

    return await asyncio.gather(*[sample(prompt) for prompt in prompts])


def score_group(
    renderer: renderers.Renderer, sequences: list[tinker.types.SampledSequence], answer: str
) -> list[float]:
    rewards = []
    for sequence in sequences:
        message, _ = renderer.parse_response(sequence.tokens)
        rewards.append(reward(renderers.get_text_content(message), answer))
    return rewards


def length_metrics(results: Sequence[tinker.types.SampleResponse]) -> dict[str, float]:
    r"""Mean response length and how often the token budget ran out.

    A thinking renderer that overruns max_tokens never emits \boxed{}, so reward
    collapses to 0 with nothing else to point at; truncated_frac names that directly.
    """
    sequences = [sequence for result in results for sequence in result.sequences]
    lengths = [len(sequence.tokens) for sequence in sequences]
    truncated = [sequence.stop_reason == "length" for sequence in sequences]
    return {
        "len/mean": sum(lengths) / len(lengths),
        "len/truncated_frac": sum(truncated) / len(truncated),
    }


def group_advantages(rewards: list[float]) -> list[float] | None:
    """Center rewards on the group mean; None when the group has no learning signal."""
    mean = sum(rewards) / len(rewards)
    advantages = [reward_value - mean for reward_value in rewards]
    if all(advantage == 0.0 for advantage in advantages):
        return None
    return advantages


def build_datums(
    prompt: tinker.types.ModelInput,
    sequences: list[tinker.types.SampledSequence],
    advantages: list[float],
) -> list[tinker.types.Datum]:
    """One training datum per rollout.

    Arrays are pre-shifted: model_input drops the final sampled token and target_tokens
    start at the first sampled position. Plain lists throughout — Datum coerces them,
    so no torch here.
    """
    datums = []
    ob_len = prompt.length - 1
    for sequence, advantage in zip(sequences, advantages, strict=True):
        if sequence.logprobs is None:
            raise ValueError("sampled sequence is missing logprobs")
        model_input = prompt.append(tinker.types.EncodedTextChunk(tokens=sequence.tokens[:-1]))
        target_tokens = [0] * ob_len + sequence.tokens
        padded_logprobs = [0.0] * ob_len + sequence.logprobs
        padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
        assert model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)
        datums.append(
            tinker.types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "logprobs": padded_logprobs,
                    "advantages": padded_advantages,
                },
            )
        )
    return datums


async def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams,
) -> dict[str, float]:
    """Submit forward_backward and optim_step before awaiting either result_async.

    Training ``*_async`` returns a future (eager submit); ``result_async`` polls.
    Both polls are gathered because awaiting one to completion before starting the other
    adds a round trip to the step.
    optim_step's metrics are empty on both backends; forward-backward's are real.
    """
    fwd_bwd_future = await training_client.forward_backward_async(datums, loss_fn="ppo")
    optim_future = await training_client.optim_step_async(adam_params)
    fwd_bwd, _ = await asyncio.gather(fwd_bwd_future.result_async(), optim_future.result_async())
    return fwd_bwd.metrics


async def evaluate(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    questions: Sequence[str],
    answers: Sequence[str],
    eval_params: tinker.types.SamplingParams,
) -> float:
    """Mean greedy-decoded reward on a held-out slice."""
    prompts = [build_prompt(renderer, question) for question in questions]
    results = await sample_groups(sampling_client, prompts, eval_params, group_size=1)
    scores = [
        score_group(renderer, result.sequences, answer)[0] for result, answer in zip(results, answers, strict=True)
    ]
    return sum(scores) / len(scores)


async def main(config: Config) -> None:
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(config.model_name), tokenizer)

    dataset = datasets.load_dataset("openai/gsm8k", "main")
    train_rows = dataset["train"]
    eval_rows = dataset["test"].select(range(config.eval_size))

    n_train_batches = len(train_rows) // config.batch_size
    if n_train_batches == 0:
        raise SystemExit(f"--batch-size {config.batch_size} exceeds the {len(train_rows)}-row train split")
    n_steps = min(config.max_steps, n_train_batches)

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name, rank=config.lora_rank
    )

    stop = renderer.get_stop_sequences()
    sampling_params = tinker.types.SamplingParams(max_tokens=config.max_tokens, stop=stop)
    eval_params = tinker.types.SamplingParams(max_tokens=config.max_tokens, stop=stop, temperature=0.0)
    adam_params = tinker.types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    wandb_run = None
    if config.wandb_project is not None:
        wandb_run = wandb.init(project=config.wandb_project, config=vars(config))

    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    queued_datums: list[tinker.types.Datum] = []

    for step in range(n_steps):
        t_start = time.time()
        metrics: dict[str, float] = {}
        rows = train_rows.select(range(step * config.batch_size, (step + 1) * config.batch_size))

        if config.eval_every > 0 and config.eval_size > 0 and step % config.eval_every == 0:
            metrics["reward/eval"] = await evaluate(
                sampling_client, renderer, eval_rows["question"], eval_rows["answer"], eval_params
            )

        prompts = [build_prompt(renderer, question) for question in rows["question"]]
        sampling = sample_groups(sampling_client, prompts, sampling_params, config.group_size)
        # Empty queue: step 0, or last sample was zero-variance. Otherwise gather so
        # the LoRA publish cannot land under this sample. Loss is the queued batch's.
        if queued_datums:
            results, train_metrics = await asyncio.gather(
                sampling, train_step(training_client, queued_datums, adam_params)
            )
            metrics.update(train_metrics)
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        else:
            results = await sampling

        queued_datums = []
        mean_rewards = []
        for prompt, result, answer in zip(prompts, results, rows["answer"], strict=True):
            rewards = score_group(renderer, result.sequences, answer)
            mean_rewards.append(sum(rewards) / len(rewards))
            advantages = group_advantages(rewards)
            if advantages is not None:
                queued_datums.extend(build_datums(prompt, result.sequences, advantages))
        metrics["reward/train"] = sum(mean_rewards) / len(mean_rewards)
        metrics.update(length_metrics(results))
        if not queued_datums:
            print(f"step {step}: all groups zero-variance, not queuing an optimizer step")

        metrics["time/step"] = time.time() - t_start
        print(f"step {step}: {'  '.join(f'{k}={v:.4f}' for k, v in metrics.items())}")
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    if queued_datums:
        flush_metrics = await train_step(training_client, queued_datums, adam_params)
        print(f"flush: {'  '.join(f'{k}={v:.4f}' for k, v in flush_metrics.items())}")
        if wandb_run is not None:
            wandb_run.log(flush_metrics, step=n_steps)


if __name__ == "__main__":
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--group-size", type=int, default=defaults.group_size)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-size", type=int, default=defaults.eval_size)
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    asyncio.run(main(Config(**vars(parser.parse_args()))))
