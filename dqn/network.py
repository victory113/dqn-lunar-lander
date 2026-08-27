"""Q-network architecture.

Author: Victory Orobosa (Member 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DQNetwork(nn.Module):
    """Maps an 8-dimensional LunarLander state to one Q-value per action.

    A plain MLP is sufficient here: the observation is already a low-dimensional
    physics vector (position, velocity, angle, angular velocity, two contact
    flags), so there is no spatial structure for a convolutional stack to
    exploit. The 256-128-64 taper was chosen to give the first layer enough
    width to separate states while keeping the parameter count small enough to
    train on CPU in a few minutes.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
