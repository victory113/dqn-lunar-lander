"""Experience replay.

Author: Pacifique (Member 3).

Consecutive transitions in an episode are highly correlated, and training a
network on them in order violates the i.i.d. assumption gradient descent relies
on. Sampling uniformly at random from a large buffer decorrelates the batch and
is what makes DQN stable enough to converge at all.
"""

from __future__ import annotations

import random
from collections import deque, namedtuple

import numpy as np
import torch

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        # deque(maxlen=...) evicts the oldest entry on overflow, so memory is
        # bounded without any explicit bookkeeping.
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Return a uniformly sampled batch as stacked tensors on the device.

        Each field is stacked into a single numpy array before conversion.
        Building one array and converting once is markedly faster than letting
        torch convert a list of arrays element by element.
        """
        experiences = random.sample(self.buffer, batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences], dtype=np.float32)).to(
            self.device
        )
        actions = torch.from_numpy(np.array([e.action for e in experiences], dtype=np.int64)).to(
            self.device
        )
        rewards = torch.from_numpy(np.array([e.reward for e in experiences], dtype=np.float32)).to(
            self.device
        )
        next_states = torch.from_numpy(
            np.array([e.next_state for e in experiences], dtype=np.float32)
        ).to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences], dtype=np.float32)).to(
            self.device
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
