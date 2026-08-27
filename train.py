"""Train a DQN agent on LunarLander-v3.

    python train.py                      # defaults from dqn/config.py
    python train.py --episodes 300       # shorter run
    python train.py --seed 7 --out runs/seed7

Writes three artifacts to the output directory: the trained weights, the
training figure, and a JSON of the run's config and final metrics. The JSON is
what the README numbers are quoted from, so a claim can always be traced to a
specific run.

Training loop author: Victory Orobosa (Member 1).
Plotting: Nicola (Member 2). Replay buffer: Pacifique (Member 3).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from dqn.agent import DQNAgent
from dqn.config import DEFAULT_CONFIG, Config
from dqn.utils import get_device, plot_training_results, set_seed


def train(config: Config, out_dir: Path) -> dict:
    device = get_device()
    set_seed(config.seed)
    print(f"Device: {device}")

    env = gym.make(config.env_id)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config, device)

    scores: list[float] = []
    avg_scores: list[float] = []
    losses: list[float] = []
    started = time.perf_counter()

    print(f"Training {config.num_episodes} episodes on {config.env_id}")
    print("-" * 66)

    for episode in range(1, config.num_episodes + 1):
        # Seed each episode deterministically so a run is reproducible end to end.
        state, _ = env.reset(seed=config.seed + episode)
        episode_reward = 0.0
        episode_losses: list[float] = []

        for _ in range(config.max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)
                # Soft update after every gradient step (see Config.tau).
                agent.soft_update_target()

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.decay_epsilon()

        scores.append(episode_reward)
        avg_scores.append(float(np.mean(scores[-config.solved_window :])))
        losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)

        if episode % config.log_every == 0:
            print(
                f"Episode {episode:4d} | Score: {episode_reward:7.1f} | "
                f"Avg({config.solved_window}): {avg_scores[-1]:7.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | Loss: {losses[-1]:.4f}"
            )

    env.close()
    elapsed = time.perf_counter() - started

    solved_episode = next(
        (i + 1 for i, s in enumerate(avg_scores) if s >= config.solved_threshold), None
    )
    summary = {
        "config": config.to_dict(),
        "wall_clock_seconds": round(elapsed, 1),
        "best_episode_score": round(max(scores), 1),
        "worst_episode_score": round(min(scores), 1),
        "final_avg_score": round(avg_scores[-1], 1),
        "solved": solved_episode is not None,
        "solved_at_episode": solved_episode,
        "episodes_above_threshold": sum(1 for s in scores if s >= config.solved_threshold),
    }

    print("-" * 66)
    print("TRAINING SUMMARY")
    for key, value in summary.items():
        if key != "config":
            print(f"  {key}: {value}")

    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(out_dir / "dqn_lunarlander.pth"))
    plot_training_results(scores, avg_scores, losses, config, out_dir / "training_curve.png")
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "training_history.npz",
        scores=np.array(scores),
        avg_scores=np.array(avg_scores),
        losses=np.array(losses),
    )
    print(f"\nArtifacts written to {out_dir.resolve()}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN on LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG.num_episodes)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    args = parser.parse_args()

    config = replace(
        DEFAULT_CONFIG,
        num_episodes=args.episodes,
        seed=args.seed,
        learning_rate=args.lr,
    )
    train(config, args.out)


if __name__ == "__main__":
    main()
