"""Seeding, device selection, and plotting helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG that affects a run.

    The environment is seeded separately, per-episode, in the training loop.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_results(scores, avg_scores, losses, config, out_path: Path) -> None:
    """Save the two-panel training figure: scores left, loss and epsilon right."""
    import matplotlib

    matplotlib.use("Agg")  # write to file without needing a display
    import matplotlib.pyplot as plt

    n_episodes = len(scores)
    episodes = np.arange(1, n_episodes + 1)

    window = min(config.plot_moving_avg_window, n_episodes)
    score_ma = np.convolve(scores, np.ones(window) / window, mode="valid")

    # Early episodes record a zero placeholder because no gradient step ran
    # while the buffer was warming up; averaging those in would distort the
    # loss curve downward.
    nonzero_losses = [loss for loss in losses if loss > 0]
    if nonzero_losses:
        loss_window = min(config.plot_moving_avg_window, len(nonzero_losses))
        loss_ma = np.convolve(nonzero_losses, np.ones(loss_window) / loss_window, mode="valid")
    else:
        loss_window, loss_ma = 0, []

    # Rebuild the epsilon schedule so exploration lines up against the curves.
    epsilons, eps = [], config.epsilon_start
    for _ in range(n_episodes):
        epsilons.append(eps)
        eps = max(config.epsilon_end, eps * config.epsilon_decay)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(episodes, scores, alpha=0.25, color="steelblue", label="Episode score")
    axes[0].plot(
        episodes,
        avg_scores,
        color="crimson",
        linewidth=2,
        label=f"Avg score ({config.solved_window} eps)",
    )
    axes[0].plot(
        np.arange(window, n_episodes + 1),
        score_ma,
        color="navy",
        linewidth=2,
        label=f"Score MA ({window})",
    )
    axes[0].axhline(
        y=config.solved_threshold,
        color="forestgreen",
        linestyle="--",
        label=f"Solved ({config.solved_threshold:.0f})",
    )

    solved_idx = next((i for i, s in enumerate(avg_scores) if s >= config.solved_threshold), None)
    if solved_idx is not None:
        axes[0].annotate(
            f"Solved @ ep {solved_idx + 1}",
            xy=(solved_idx + 1, avg_scores[solved_idx]),
            xytext=(solved_idx + 1, avg_scores[solved_idx] + 50),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
        )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Training progress")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, losses, alpha=0.35, color="darkorange", label="Episode avg loss")
    if len(loss_ma) > 0:
        axes[1].plot(
            np.arange(loss_window, loss_window + len(loss_ma)),
            loss_ma,
            color="saddlebrown",
            linewidth=2,
            label=f"Loss MA ({loss_window})",
        )

    ax_eps = axes[1].twinx()
    ax_eps.plot(episodes, epsilons, color="teal", linestyle=":", linewidth=2, label="Epsilon")
    ax_eps.set_ylabel("Epsilon", color="teal")
    ax_eps.tick_params(axis="y", labelcolor="teal")

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg loss")
    axes[1].set_title("Training loss and exploration decay")
    lines_l, labels_l = axes[1].get_legend_handles_labels()
    lines_r, labels_r = ax_eps.get_legend_handles_labels()
    axes[1].legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
