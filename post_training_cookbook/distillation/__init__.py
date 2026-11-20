"""Off-policy distillation: a hosted teacher writes the data, a post-training student learns it.

The pipeline has three parts:

- :mod:`~post_training_cookbook.distillation.teacher` calls the teacher through serverless chat
  completions, with a resumable on-disk cache, concurrency per API key and an optional budget.
- :mod:`~post_training_cookbook.distillation.data` turns teacher outputs into chat SFT rows, dropping
  failed, truncated or filtered outputs, and mixes several sets (for example a retention language).
- :mod:`~post_training_cookbook.distillation.train` fine-tunes the student with cross-entropy on a
  resource with no generator replica, and saves an inference checkpoint for deployment.

The package is task-agnostic. The FLORES-200 translation helpers the recipes use (prompts, filters,
eval items, chrF++ and script-rate metrics) live in ``recipes/distillation/translation.py``.
"""
