"""Replay buffer: capacity, sampling shapes, and dtype correctness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dqn.replay_buffer import ReplayBuffer

DEVICE = torch.device("cpu")
STATE_DIM = 8


def make_transition(value: float = 0.0):
    state = np.full(STATE_DIM, value, dtype=np.float32)
    next_state = np.full(STATE_DIM, value + 1, dtype=np.float32)
    return state, 1, 0.5, next_state, False


class TestCapacity:
    def test_starts_empty(self):
        assert len(ReplayBuffer(10, DEVICE)) == 0

    def test_grows_with_each_push(self):
        buffer = ReplayBuffer(10, DEVICE)
        for i in range(5):
            buffer.push(*make_transition(i))
        assert len(buffer) == 5

    def test_never_exceeds_capacity(self):
        buffer = ReplayBuffer(10, DEVICE)
        for i in range(50):
            buffer.push(*make_transition(i))
        assert len(buffer) == 10

    def test_evicts_oldest_first(self):
        """The buffer is a sliding window over recent experience, not a sample
        of all of it. Old transitions must fall off the front."""
        buffer = ReplayBuffer(3, DEVICE)
        for i in range(5):
            buffer.push(*make_transition(i))

        first_values = sorted(float(e.state[0]) for e in buffer.buffer)
        assert first_values == [2.0, 3.0, 4.0]


class TestSampling:
    @pytest.fixture
    def full_buffer(self):
        buffer = ReplayBuffer(100, DEVICE)
        for i in range(100):
            buffer.push(*make_transition(i))
        return buffer

    def test_returns_five_stacked_tensors(self, full_buffer):
        batch = full_buffer.sample(16)
        assert len(batch) == 5
        assert all(isinstance(t, torch.Tensor) for t in batch)

    def test_shapes_match_batch_size(self, full_buffer):
        states, actions, rewards, next_states, dones = full_buffer.sample(16)
        assert states.shape == (16, STATE_DIM)
        assert next_states.shape == (16, STATE_DIM)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert dones.shape == (16,)

    def test_actions_are_int64_for_gather(self, full_buffer):
        """gather() indexes with int64. A float action tensor raises at runtime
        deep inside the Bellman update, which is a confusing place to find it."""
        _, actions, _, _, _ = full_buffer.sample(16)
        assert actions.dtype == torch.int64

    def test_continuous_fields_are_float32(self, full_buffer):
        states, _, rewards, next_states, dones = full_buffer.sample(16)
        assert states.dtype == torch.float32
        assert rewards.dtype == torch.float32
        assert next_states.dtype == torch.float32
        assert dones.dtype == torch.float32

    def test_dones_are_numeric_not_boolean(self, full_buffer):
        """The target multiplies by (1 - dones), so dones must be arithmetic."""
        _, _, _, _, dones = full_buffer.sample(16)
        assert torch.all((dones == 0.0) | (dones == 1.0))
