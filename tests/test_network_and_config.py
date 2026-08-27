"""Network shape contracts and configuration behaviour."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from dqn.config import DEFAULT_CONFIG, Config
from dqn.network import DQNetwork

STATE_DIM, ACTION_DIM = 8, 4


class TestNetwork:
    @pytest.fixture
    def net(self):
        torch.manual_seed(0)
        return DQNetwork(STATE_DIM, ACTION_DIM)

    def test_emits_one_q_value_per_action(self, net):
        out = net(torch.randn(1, STATE_DIM))
        assert out.shape == (1, ACTION_DIM)

    def test_handles_a_batch(self, net):
        out = net(torch.randn(32, STATE_DIM))
        assert out.shape == (32, ACTION_DIM)

    def test_q_values_are_unbounded(self, net):
        """No activation on the output layer. Q-values are returns, which in
        LunarLander are negative as often as positive; a ReLU or sigmoid here
        would make negative returns unrepresentable."""
        out = net(torch.randn(256, STATE_DIM) * 10)
        assert out.min() < 0 < out.max()

    def test_is_deterministic_for_a_fixed_input(self, net):
        state = torch.randn(1, STATE_DIM)
        net.eval()
        with torch.no_grad():
            assert torch.equal(net(state), net(state))

    def test_gradients_reach_every_layer(self, net):
        net(torch.randn(8, STATE_DIM)).sum().backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"no gradient reached {name}"
            assert torch.any(param.grad != 0), f"zero gradient at {name}"


class TestConfig:
    def test_is_immutable(self):
        """Frozen so a run cannot silently mutate its own settings mid-training
        and invalidate the config recorded in the summary."""
        with pytest.raises(FrozenInstanceError):
            DEFAULT_CONFIG.learning_rate = 1.0  # type: ignore[misc]

    def test_replace_produces_an_independent_config(self):
        modified = replace(DEFAULT_CONFIG, num_episodes=10)
        assert modified.num_episodes == 10
        assert DEFAULT_CONFIG.num_episodes != 10

    def test_serialises_to_json(self):
        """Every run records its config, so it has to survive json.dumps."""
        assert json.loads(json.dumps(DEFAULT_CONFIG.to_dict())) == DEFAULT_CONFIG.to_dict()

    def test_epsilon_schedule_is_well_formed(self):
        assert DEFAULT_CONFIG.epsilon_start > DEFAULT_CONFIG.epsilon_end
        assert 0.0 < DEFAULT_CONFIG.epsilon_decay < 1.0

    def test_discount_is_a_proper_fraction(self):
        assert 0.0 < DEFAULT_CONFIG.gamma <= 1.0

    def test_buffer_warmup_is_reachable(self):
        assert DEFAULT_CONFIG.min_buffer_size <= DEFAULT_CONFIG.buffer_size
        assert DEFAULT_CONFIG.batch_size <= DEFAULT_CONFIG.min_buffer_size

    def test_tau_is_a_small_interpolation_weight(self):
        assert 0.0 < DEFAULT_CONFIG.tau < 1.0

    def test_epsilon_floor_is_actually_reached_within_the_run(self):
        """Documents a real property of the default schedule: at 0.995 decay
        over 700 episodes epsilon lands near 0.03, so the 0.01 floor never
        engages. If the episode count or decay changes, this should be revisited.
        """
        eps = DEFAULT_CONFIG.epsilon_start
        for _ in range(DEFAULT_CONFIG.num_episodes):
            eps = max(DEFAULT_CONFIG.epsilon_end, eps * DEFAULT_CONFIG.epsilon_decay)
        assert eps > DEFAULT_CONFIG.epsilon_end
        assert eps == pytest.approx(0.03, abs=0.01)


class TestCustomConfig:
    def test_a_short_run_config_is_valid(self):
        cfg = Config(num_episodes=5, batch_size=8, min_buffer_size=8, buffer_size=100)
        assert cfg.num_episodes == 5
        assert cfg.batch_size <= cfg.min_buffer_size <= cfg.buffer_size
