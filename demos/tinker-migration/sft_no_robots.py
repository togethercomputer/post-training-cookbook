"""SFT on No Robots — a minimal rewrite of tinker_cookbook/recipes/sl_loop.py."""

import argparse
import time
from dataclasses import dataclass

import datasets

# import tinker
import together.post_training.tinker as tinker
import wandb
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

_ = load_dotenv(override=True)


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.8-27B"
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_length: int = 32768
    lora_rank: int = 32
    max_steps: int = 30
    wandb_project: str | None = None


def decayed_lr(learning_rate: float, step: int, n_train_batches: int) -> float:
    """Learning rate for a step of the linear decay.

    The schedule spans the full epoch, so a --max-steps-truncated run reproduces a prefix of
    it rather than compressing it.
    """
    return learning_rate * max(0.0, 1.0 - step / n_train_batches)


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams,
) -> dict[str, float]:
    """Submit forward_backward and optim_step before resolving either, then report the forward-backward metrics.

    optim_step's metrics are empty on both backends; forward-backward's are real.
    """
    fwd_bwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
    optim_future = training_client.optim_step(adam_params)
    fwd_bwd = fwd_bwd_future.result()
    optim_future.result()
    return fwd_bwd.metrics


def main(config: Config) -> None:
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(config.model_name), tokenizer)

    dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
    train_rows = dataset["train"].shuffle(seed=0)
    # Drop the last incomplete batch (like PyTorch's drop_last=True): a partial batch has a
    # different effective gradient magnitude, which can spike the loss.
    n_train_batches = len(train_rows) // config.batch_size
    if n_train_batches == 0:
        raise SystemExit(f"--batch-size {config.batch_size} exceeds the {len(train_rows)}-row train split")
    n_steps = min(config.max_steps, n_train_batches)

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=config.model_name, rank=config.lora_rank)

    wandb_run = None
    if config.wandb_project is not None:
        wandb_run = wandb.init(project=config.wandb_project, config=vars(config))

    for step in range(n_steps):
        t_start = time.time()
        rows = train_rows.select(range(step * config.batch_size, (step + 1) * config.batch_size))
        datums = [conversation_to_datum(row["messages"], renderer, config.max_length) for row in rows]

        lr = decayed_lr(config.learning_rate, step, n_train_batches)
        adam_params = tinker.types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

        metrics = dict(train_step(training_client, datums, adam_params))
        metrics.update(
            num_sequences=len(datums),
            num_tokens=sum(datum.model_input.length for datum in datums),
            learning_rate=lr,
        )
        metrics["time/step"] = time.time() - t_start
        print(f"step {step}: {'  '.join(f'{k}={v:.4f}' for k, v in metrics.items())}")
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)


if __name__ == "__main__":
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    main(Config(**vars(parser.parse_args())))
