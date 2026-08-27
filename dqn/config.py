"""Hyperparameters and training configuration.

Every tunable value lives here so a run can be described by this file alone.
Originally authored by Nicola (Member 2) as a flat constants block; kept in one
place during the refactor for the same reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Config:
    # --- Environment ---
    env_id: str = "LunarLander-v3"
    seed: int = 42

    # --- Training length ---
    num_episodes: int = 700
    max_steps_per_episode: int = 1000

    # --- Optimisation ---
    batch_size: int = 64
    learning_rate: float = 5e-4
    gamma: float = 0.99
    grad_clip_norm: float = 1.0

    # --- Exploration (epsilon-greedy) ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995  # multiplicative, applied once per episode

    # --- Replay buffer ---
    buffer_size: int = 100_000
    min_buffer_size: int = 1_000  # no gradient steps until the buffer holds this many

    # --- Target network ---
    # The target network is soft-updated (Polyak averaging) after every gradient
    # step rather than hard-copied every N steps. With tau this small the target
    # moves slowly enough to stay a stable regression target, and updating every
    # step avoids the discontinuity a periodic hard copy introduces.
    tau: float = 0.005

    # --- Algorithm variant ---
    # Vanilla DQN takes max_a' Q_target(s', a'), so the same network both picks
    # the best next action and estimates its value. Any positive noise in the
    # estimate gets selected by the max, which biases targets upward and
    # compounds through bootstrapping. Double DQN splits the two: the policy
    # network chooses the action, the target network scores it.
    double_dqn: bool = True

    # --- Reporting ---
    solved_threshold: float = 200.0  # LunarLander is "solved" at 200 avg reward
    solved_window: int = 100  # ...averaged over this many consecutive episodes
    plot_moving_avg_window: int = 25
    log_every: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = Config()
