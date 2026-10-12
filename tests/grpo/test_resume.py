"""Client progress comes from checkpoint records, not optimizer counters."""

import json

import pytest

from post_training_cookbook.grpo.resume import ResumePoint, load_resume_point
from tests.grpo._fakes import _base_config

_STATE = {"version": 1, "source": {}, "next_index": 24, "pending": []}


def test_continuation_requires_run_dir():
    with pytest.raises(ValueError, match="requires resume_run_dir"):
        _base_config(resume_from_checkpoint_id="checkpoint")


@pytest.mark.parametrize("checkpoint,load_optimizer", [(None, None), ("checkpoint", False)])
def test_fresh_starts_ignore_records(tmp_path, checkpoint, load_optimizer):
    config = _base_config(resume_from_checkpoint_id=checkpoint, load_optimizer=load_optimizer)
    assert load_resume_point(config) is None
    with pytest.raises(ValueError, match="resume_run_dir requires"):
        _base_config(resume_from_checkpoint_id=checkpoint, load_optimizer=load_optimizer, resume_run_dir=str(tmp_path))


def test_matches_training_record(tmp_path):
    records = [
        {"kind": "training", "ref": "other", "step": 90, "data_state": {}},
        {"kind": "inference", "ref": "checkpoint", "step": 99},
        {"kind": "training", "ref": "checkpoint", "step": 12, "data_state": _STATE},
    ]
    (tmp_path / "checkpoints.jsonl").write_text("\n".join(json.dumps(record) for record in records))
    config = _base_config(resume_from_checkpoint_id="checkpoint", resume_run_dir=str(tmp_path))
    assert load_resume_point(config) == ResumePoint(step=12, data_state=_STATE)


@pytest.mark.parametrize(
    "records,message",
    [
        (None, "No training checkpoint"),
        ([{"kind": "inference", "ref": "checkpoint", "step": 12}], "No training checkpoint"),
        ([{"kind": "training", "ref": "checkpoint", "step": 12}], "no data_state"),
    ],
)
def test_unusable_record_rejected(tmp_path, records, message):
    if records is not None:
        (tmp_path / "checkpoints.jsonl").write_text("\n".join(json.dumps(record) for record in records))
    config = _base_config(resume_from_checkpoint_id="checkpoint", resume_run_dir=str(tmp_path))
    with pytest.raises(ValueError, match=message):
        load_resume_point(config)
