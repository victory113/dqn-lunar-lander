"""DQN agent: action selection, Bellman update, target-network maintenance.

Author: Victory Orobosa (Member 1).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.config import Config
from dqn.network import DQNetwork
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network agent with experience replay and a soft-updated target net.

    Two networks are maintained. ``policy_net`` is the one being trained;
    ``target_net`` supplies the bootstrapped value in the Bellman target. Using
    a single network for both makes the regression target move every time the
    weights update, which is the classic "chasing a moving target" instability.
    """

    def __init__(
        self, state_size: int, action_size: int, config: Config, device: torch.device
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.cfg = config
        self.device = device
        self.epsilon = config.epsilon_start

        self.policy_net = DQNetwork(state_size, action_size).to(device)
        self.target_net = DQNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        # Smooth L1 (Huber) rather than MSE: TD errors are occasionally huge
        # early in training, and squaring them produces gradients large enough
        # to destabilise the network. Huber is linear past delta=1.
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(config.buffer_size, device)

    def select_action(self, state, *, greedy: bool = False) -> int:
        """Epsilon-greedy action selection. ``greedy=True`` disables exploration."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            return int(self.policy_net(state_tensor).argmax(dim=1).item())

    def learn(self) -> float | None:
        """One gradient step on a replayed minibatch. ``None`` until warm.

        Holding off until the buffer has ``min_buffer_size`` transitions avoids
        overfitting the first few hundred steps, which are almost pure noise
        while epsilon is near 1.
        """
        if len(self.memory) < self.cfg.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)

        # Q(s, a) for the actions actually taken.
        q_predicted = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target. no_grad because the target network is not being
        # trained here; letting gradients flow into it would defeat its purpose.
        # (1 - dones) zeroes the bootstrap on terminal transitions, where there
        # is no next state to value.
        with torch.no_grad():
            q_next = self._next_state_values(next_states)
            q_target = rewards + (self.cfg.gamma * q_next * (1.0 - dones))

        loss = self.loss_fn(q_predicted, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=self.cfg.grad_clip_norm
        )
        self.optimizer.step()

        return float(loss.item())

    def _next_state_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """max_a' Q(s', a') for the Bellman target, vanilla or Double DQN.

        Vanilla: the target network both selects and evaluates the best next
        action. Because ``max`` always picks the largest estimate, any upward
        noise is preferentially selected, so targets are biased high and the
        bias compounds through bootstrapping.

        Double DQN: the *policy* network selects the argmax action and the
        *target* network reports its value. The two networks' errors are not
        perfectly correlated, so a spuriously high estimate in one is unlikely
        to be echoed by the other, and the overestimation largely cancels.
        """
        if self.cfg.double_dqn:
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            return self.target_net(next_states).gather(1, next_actions).squeeze(1)
        return self.target_net(next_states).max(dim=1)[0]

    def soft_update_target(self) -> None:
        """Polyak averaging: target <- tau * policy + (1 - tau) * target."""
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters(), strict=True
            ):
                target_param.data.copy_(
                    self.cfg.tau * policy_param.data + (1.0 - self.cfg.tau) * target_param.data
                )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    # --- persistence ---

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
