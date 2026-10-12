"""Tests for GRPO config loading and derived defaults."""

import os
import tempfile

import pytest

from post_training_cookbook.grpo.config import GRPOConfig, policy_loss


def test_from_yaml_loads_values():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model: test-model\nlearning_rate: 0.001\nbatch_size: 4\n")
        f.flush()
        config = GRPOConfig.from_yaml(f.name)
    os.unlink(f.name)
    assert config.model == "test-model"
    assert config.learning_rate == 0.001
    assert config.batch_size == 4


def test_from_yaml_rejects_unknown_keys():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model: test-model\nunknown_field: 42\n")
        f.flush()
        with pytest.raises(ValueError, match="unknown_field"):
            GRPOConfig.from_yaml(f.name)
    os.unlink(f.name)


class TestDerivedDefaults:
    """Each row exercises one derivation rule: given these explicit fields,
    the named attribute should equal ``expected``. Whichever fields are absent
    are derived per the config's __post_init__ rules."""

    @pytest.mark.parametrize(
        ("kwargs", "attr", "expected"),
        [
            # max_sample_tokens defaults to max_model_length
            ({"max_model_length": 4096}, "max_sample_tokens", 4096),
            ({"max_model_length": 4096, "max_sample_tokens": 1024}, "max_sample_tokens", 1024),
            # max_prompt_length = max_model_length - max_sample_tokens
            ({"max_model_length": 2048, "max_sample_tokens": 512}, "max_prompt_length", 2048 - 512),
            ({"max_model_length": 2048}, "max_prompt_length", 0),  # sample fills context
            ({"max_model_length": 2048, "max_prompt_length": 768}, "max_prompt_length", 768),
            # overlong_buffer_len = max_sample_tokens // 8
            ({"max_model_length": 2048}, "overlong_buffer_len", 2048 // 8),
            ({"max_model_length": 2048, "max_sample_tokens": 512}, "overlong_buffer_len", 512 // 8),
            ({"overlong_buffer_len": 48}, "overlong_buffer_len", 48),
        ],
    )
    def test_derived(self, kwargs, attr, expected):
        config = GRPOConfig(model="test-model", **kwargs)
        assert getattr(config, attr) == expected

    @pytest.mark.parametrize(("factor", "expected"), [(0.1, True), (0.0, False)])
    def test_overlong_penalty_property(self, factor, expected):
        assert GRPOConfig(model="test-model", overlong_penalty_factor=factor).overlong_penalty is expected


class TestValidation:
    def test_rejects_sample_tokens_exceeding_model_length(self):
        with pytest.raises(ValueError, match="max_sample_tokens.*exceeds.*max_model_length"):
            GRPOConfig(model="test-model", max_model_length=1024, max_sample_tokens=2048)

    def test_rejects_prompt_plus_min_response_exceeding_model_length(self):
        with pytest.raises(ValueError, match="max_prompt_length.*min_response_tokens.*exceeds.*max_model_length"):
            GRPOConfig(model="test-model", max_model_length=1024, max_sample_tokens=512, max_prompt_length=1024)


def test_from_yaml_loads_loss_knobs(tmp_path):
    path = tmp_path / "ppo.yaml"
    path.write_text("loss: ppo\nclip_high_threshold: 1.2\n")
    config = GRPOConfig.from_yaml(str(path))
    assert config.loss == "ppo"
    assert config.clip_high_threshold == 1.2
    assert policy_loss(config) == {
        "type": "LOSS_TYPE_PPO",
        "ppo_params": {"clip_high_threshold": 1.2},
    }


def test_from_yaml_rejects_unknown_fields(tmp_path):
    path = tmp_path / "unknown.yaml"
    path.write_text("unknown_parameter: 0.04\n")
    with pytest.raises(ValueError, match="unknown_parameter"):
        GRPOConfig.from_yaml(str(path))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"loss": "cross_entropy"}, "unsupported loss"),
        ({"loss": "LOSS_TYPE_CROSS_ENTROPY"}, "unsupported loss"),
        ({"loss": "dro"}, "dro_beta is required"),
        ({"loss": "ppo", "dro_beta": 0.1}, "dro_beta is only valid"),
        ({"loss": "ppo", "loss_agg_type": "GRPO_LOSS_AGGREGATION_TYPE_TOKEN_MEAN"}, "loss_agg_type"),
        ({"loss": "ppo", "ratio_type": "GRPO_LOSS_RATIO_TYPE_SEQUENCE"}, "ratio_type"),
        ({"loss": "dppo", "clip_high_threshold": 1.5}, "clip thresholds"),
    ],
)
def test_rejects_invalid_loss_knobs(kwargs, error):
    with pytest.raises(ValueError, match=error):
        GRPOConfig(**kwargs)


class TestDynamicSampling:
    def test_disabled_by_default(self):
        config = GRPOConfig(model="test-model")
        assert config.max_sampling_attempts == 1

    def test_explicit_cap(self):
        config = GRPOConfig(model="test-model", max_sampling_attempts=4)
        assert config.max_sampling_attempts == 4

    def test_rejects_zero_max_attempts(self):
        with pytest.raises(ValueError, match="max_sampling_attempts"):
            GRPOConfig(model="test-model", max_sampling_attempts=0)


def test_load_optimizer_false_requires_checkpoint():
    with pytest.raises(ValueError, match="load_optimizer=False requires resume_from_checkpoint_id"):
        GRPOConfig(model="test-model", load_optimizer=False)


def test_weights_only_resume_accepted():
    config = GRPOConfig(model="test-model", resume_from_checkpoint_id="tckpt-1", load_optimizer=False)
    assert config.resume_from_checkpoint_id == "tckpt-1"
    assert config.load_optimizer is False


class TestAsyncCollection:
    @pytest.mark.parametrize("budget", [None, 0, 1])
    def test_defaults_and_explicit_on_policy_budget(self, budget):
        config = GRPOConfig(batch_size=4, episodes_per_task=2, max_steps_off_policy=budget)
        assert config.max_steps_off_policy == budget
        assert config.groups_in_flight == 4
        assert config.weight_sync_type == "WEIGHT_SYNC_TYPE_SYNCHRONOUS"

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"max_steps_off_policy": -1}, "max_steps_off_policy"),
            ({"max_steps_off_policy": 0.5}, "must be an integer"),
            ({"max_steps_off_policy": False}, "must be an integer"),
            ({"groups_in_flight": 0}, "groups_in_flight"),
            ({"groups_in_flight": -1}, "groups_in_flight"),
            ({"batch_size": 0}, "batch_size"),
            ({"weight_sync_type": "BACKGROUND_PUBLISH"}, "weight_sync_type"),
            ({"weight_sync_type": "WEIGHT_SYNC_TYPE_PIPELINE", "max_steps_off_policy": 0}, "use_lora=False"),
            ({"weight_sync_type": "WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH"}, "requires async collection"),
        ],
    )
    def test_rejects_invalid_config(self, kwargs, error):
        with pytest.raises(ValueError, match=error):
            GRPOConfig(**kwargs)

    @pytest.mark.parametrize("mode", ["SYNCHRONOUS", "BACKGROUND_PUBLISH", "PIPELINE"])
    def test_supported_weight_sync_modes(self, mode):
        """Deferred publishes are only reachable from the loop that accounts for them."""
        deferred = mode != "SYNCHRONOUS"
        config = GRPOConfig(
            weight_sync_type=f"WEIGHT_SYNC_TYPE_{mode}",
            use_lora=mode != "PIPELINE",
            episodes_per_task=2 if deferred else 1,
            max_steps_off_policy=1 if deferred else None,
        )
        assert config.weight_sync_type == f"WEIGHT_SYNC_TYPE_{mode}"

    def test_async_fields_load_from_yaml(self, tmp_path):
        path = tmp_path / "async.yaml"
        path.write_text("max_steps_off_policy: 0\ngroups_in_flight: 3\nepisodes_per_task: 2\n")
        config = GRPOConfig.from_yaml(str(path))
        assert config.max_steps_off_policy == 0
        assert config.groups_in_flight == 3


class TestGeneratorReplicas:
    @pytest.mark.parametrize("count", [0, -1])
    def test_requires_positive_count(self, count):
        with pytest.raises(ValueError, match=">= 1"):
            GRPOConfig(num_generator_replicas=count)

    @pytest.mark.parametrize("count", [1.5, True, "2", None])
    def test_rejects_non_integers(self, count):
        """from_yaml does not enforce annotations, so __post_init__ must."""
        with pytest.raises(ValueError, match="must be an integer"):
            GRPOConfig(num_generator_replicas=count)

    def test_replica_count_loads_from_yaml(self, tmp_path):
        path = tmp_path / "replicas.yaml"
        path.write_text("model: test-model\nnum_generator_replicas: 2\n")
        assert GRPOConfig.from_yaml(str(path)).num_generator_replicas == 2
