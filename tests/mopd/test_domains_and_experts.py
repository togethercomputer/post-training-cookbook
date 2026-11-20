import pytest

from post_training_cookbook.datasets.registry import DATASETS
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.mopd import experts
from post_training_cookbook.mopd.domains import default_domains, load_splits, register

from ._fakes import toy_domain


def _rows(prefix, n):
    return [{"q": f"{prefix} question {i}", "a": f"{prefix}{i}"} for i in range(n)]


def test_splits_are_disjoint_and_sized():
    domain = toy_domain("alpha", _rows("a", 20), num_train=12, num_test=5)
    train, test = load_splits(domain)
    assert len(train) == 12 and len(test) == 5
    assert not {r["q"] for r in train} & {r["q"] for r in test}


def test_splits_reject_a_dataset_that_is_too_small():
    with pytest.raises(ValueError, match="too small"):
        load_splits(toy_domain("alpha", _rows("a", 10), num_train=8, num_test=5))


def test_domain_names_are_validated_and_registered_under_mopd_keys():
    with pytest.raises(ValueError):
        toy_domain("a/b", _rows("a", 4), 2, 1)
    with pytest.raises(ValueError):
        toy_domain("ok", _rows("a", 4), 0, 1)
    domain = toy_domain("beta", _rows("b", 4), 2, 1)
    register([domain])
    assert DATASETS["mopd/beta"] is domain.dataset
    with pytest.raises(ValueError, match="unique"):
        register([domain, domain])


def test_default_domains_are_zero_shot_and_uniquely_named():
    domains = default_domains()
    assert [d.name for d in domains] == ["math", "ifeval", "game24"]
    assert all(not d.dataset.few_shot_messages for d in domains)
    # The shipped deepmath / game24 entries keep their few-shot examples.
    assert DATASETS["deepmath"].few_shot_messages and DATASETS["game24"].few_shot_messages


def _base():
    return GRPOConfig(model="Qwen/Qwen3.8-27B", num_steps=7, batch_size=4, num_val_examples=50, wandb_project=None)


def test_expert_configs_attach_every_expert_and_checkpoint_at_the_last_step(tmp_path):
    a = toy_domain("alpha", _rows("a", 30), 20, 5)
    b = toy_domain("beta", _rows("b", 30), 20, 8)
    configs = experts.expert_configs([a, b], _base(), model_resources_id="mr-1", run_prefix="exp", runs_root=tmp_path)
    assert set(configs) == {"alpha", "beta"}
    for name, cfg in configs.items():
        assert cfg.model is None and cfg.model_resources_id == "mr-1"
        assert cfg.training_checkpoint_interval == 7
    assert configs["alpha"].dataset == "mopd/alpha" and configs["alpha"].num_examples == 20
    assert configs["alpha"].num_val_examples == 5  # capped by the test split
    assert configs["alpha"].wandb_run_name == "exp-expert-alpha"
    assert configs["beta"].batch_size == 4  # each expert keeps the base batch


def test_checkpoint_interval_must_divide_num_steps(tmp_path):
    a = toy_domain("alpha", _rows("a", 30), 20, 5)
    configs = experts.expert_configs([a], _base(), model_resources_id="mr", run_prefix="p", checkpoint_interval=7)
    assert configs["alpha"].training_checkpoint_interval == 7
    with pytest.raises(ValueError, match="divide"):
        experts.expert_configs([a], _base(), model_resources_id="mr", run_prefix="q", checkpoint_interval=3)


def test_expert_configs_need_num_steps_and_a_fresh_prefix(tmp_path):
    a = toy_domain("alpha", _rows("a", 30), 20, 5)
    with pytest.raises(ValueError, match="num_steps"):
        experts.expert_configs([a], GRPOConfig(model="m", wandb_project=None), model_resources_id="mr", run_prefix="p")
    (tmp_path / "p-expert-alpha").mkdir()
    with pytest.raises(FileExistsError):
        experts.expert_configs([a], _base(), model_resources_id="mr", run_prefix="p", runs_root=tmp_path)


def test_train_experts_collects_checkpoints_and_isolates_failures(tmp_path, monkeypatch):
    a = toy_domain("alpha", _rows("a", 30), 20, 5)
    b = toy_domain("beta", _rows("b", 30), 20, 5)
    configs = experts.expert_configs([a, b], _base(), model_resources_id="mr", run_prefix="p", runs_root=tmp_path)

    def fake_train(config):
        run_dir = tmp_path / config.wandb_run_name
        run_dir.mkdir()
        if config.dataset == "mopd/beta":
            raise RuntimeError("boom")
        (run_dir / "checkpoints.jsonl").write_text(
            '{"step": 7, "kind": "training", "ref": "ckpt-t"}\n{"step": 7, "kind": "inference", "ref": "ckpt-i"}\n'
        )

    monkeypatch.setattr(experts, "train_single_turn", fake_train)
    results = experts.train_experts(configs, runs_root=tmp_path)
    assert results["alpha"].training_checkpoint_id == "ckpt-t"
    assert results["alpha"].inference_checkpoint_id == "ckpt-i"
    assert results["alpha"].error is None
    assert results["beta"].training_checkpoint_id is None
    assert results["beta"].error == "RuntimeError: boom"
