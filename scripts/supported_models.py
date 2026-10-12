"""List supported post-training models and their hardware-specific capabilities.

Each printed row is one validated model/GPU combination with training modes and
generator context length.

    python scripts/supported_models.py
"""

from together import Together

_ROW = "{base:<40}  gpu={gpu:<12}  default={default:<3}  modes={modes:<20}  ctx={ctx}"


def _modes(compute_config) -> str:
    trainer = compute_config.trainer_config
    modes = []
    if trainer is not None:
        if trainer.full is not None:
            modes.append("full")
        if trainer.lora is not None:
            modes.append("lora")
    if compute_config.generator_config is not None:
        modes.append("sampling")
    return ",".join(modes) or "-"


if __name__ == "__main__":
    models = Together().post_training.supported_models.get().data
    for model in models:
        for compute_config in model.compute_configs:
            generator = compute_config.generator_config
            context_length = generator.context_length if generator is not None else "-"
            print(
                _ROW.format(
                    base=model.base_model,
                    gpu=compute_config.gpu_type,
                    default="yes" if compute_config.gpu_type == model.default_gpu_type else "no",
                    modes=_modes(compute_config),
                    ctx=context_length,
                )
            )
