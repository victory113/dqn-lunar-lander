"""Evaluate a trained DQN checkpoint with exploration disabled.

    python evaluate.py --checkpoint runs/latest/dqn_lunarlander.pth
    python evaluate.py --checkpoint runs/latest/dqn_lunarlander.pth --episodes 100
    python evaluate.py --checkpoint runs/latest/dqn_lunarlander.pth --record

Evaluation episodes are seeded from a fixed offset, so the reported average is
reproducible rather than a different sample of the environment each run. The
original assignment evaluated on 10 unseeded episodes; 100 seeded episodes is a
tighter estimate and is what the README quotes.

Evaluation harness: Nicola (Member 2), extended during the refactor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from dqn.agent import DQNAgent
from dqn.config import DEFAULT_CONFIG, Config
from dqn.utils import get_device, set_seed

EVAL_SEED_OFFSET = 10_000  # keeps eval episodes disjoint from training seeds


def evaluate(
    checkpoint: Path,
    config: Config,
    num_episodes: int = 100,
    record: bool = False,
    out_dir: Path | None = None,
) -> dict:
    device = get_device()
    set_seed(config.seed)

    render_mode = "rgb_array" if record else None
    env = gym.make(config.env_id, render_mode=render_mode)

    if record:
        video_dir = (out_dir or checkpoint.parent) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        # Record the first three episodes only — enough for a README clip.
        env = gym.wrappers.RecordVideo(
            env, str(video_dir), episode_trigger=lambda ep: ep < 3, disable_logger=True
        )

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config, device)
    agent.load(str(checkpoint))

    scores: list[float] = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed=EVAL_SEED_OFFSET + episode)
        total_reward, done = 0.0, False

        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        scores.append(total_reward)

    env.close()

    arr = np.array(scores, dtype=np.float32)
    solved = int(np.sum(arr >= config.solved_threshold))
    results = {
        "checkpoint": str(checkpoint),
        "episodes": num_episodes,
        "mean": round(float(arr.mean()), 1),
        "std": round(float(arr.std()), 1),
        "median": round(float(np.median(arr)), 1),
        "min": round(float(arr.min()), 1),
        "max": round(float(arr.max()), 1),
        "episodes_at_or_above_threshold": solved,
        "success_rate_pct": round(100.0 * solved / num_episodes, 1),
    }

    print("EVALUATION")
    for key, value in results.items():
        print(f"  {key}: {value}")

    target = (out_dir or checkpoint.parent) / "evaluation.json"
    target.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {target.resolve()}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--record", action="store_true", help="save mp4s of the first 3 episodes")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    evaluate(args.checkpoint, DEFAULT_CONFIG, args.episodes, args.record, args.out)


if __name__ == "__main__":
    main()
