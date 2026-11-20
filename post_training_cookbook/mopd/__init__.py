"""Multi-teacher on-policy distillation (MOPD) on one model resource.

1. :mod:`~post_training_cookbook.mopd.experts` trains one GRPO expert per domain, concurrently.
2. :mod:`~post_training_cookbook.mopd.train` distills them into one student: the student samples,
   each completion's domain teacher scores it token by token, and the student trains on the
   per-token reverse KL.
3. :mod:`~post_training_cookbook.mopd.evaluate` scores every model on every domain.
"""
