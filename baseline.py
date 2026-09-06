"""Random-policy baseline.

A learned agent's score means nothing without a floor to compare it to. This
runs a uniformly random policy over the same seeded episodes the evaluation
harness uses, so "our agent scores X" becomes "our agent scores X against a
random baseline of Y on identical starting conditions".

    python baseline.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np

from dqn.config import DEFAULT_CONFIG
from evaluate import EVAL_SEED_OFFSET


def run_baseline(episodes: int = 100, seed: int = 42, out: Path | None = None) -> dict:
    random.seed(seed)
    env = gym.make(DEFAULT_CONFIG.env_id)
    env.action_space.seed(seed)  # sample() has its own RNG; reset(seed=) does not cover it

    scores: list[float] = []
    for episode in range(episodes):
        # Same seeds as evaluate.py, so the comparison is like for like.
        state, _ = env.reset(seed=EVAL_SEED_OFFSET + episode)
        total, done, steps = 0.0, False, 0
        while not done and steps < DEFAULT_CONFIG.max_steps_per_episode:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
            steps += 1
        scores.append(total)

    env.close()

    arr = np.array(scores, dtype=np.float32)
    results = {
        "policy": "uniform_random",
        "episodes": episodes,
        "mean": round(float(arr.mean()), 1),
        "std": round(float(arr.std()), 1),
        "median": round(float(np.median(arr)), 1),
        "min": round(float(arr.min()), 1),
        "max": round(float(arr.max()), 1),
        "episodes_at_or_above_threshold": int(np.sum(arr >= DEFAULT_CONFIG.solved_threshold)),
    }

    print("RANDOM BASELINE")
    for key, value in results.items():
        print(f"  {key}: {value}")

    target = (out or Path("runs")) / "baseline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {target.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random-policy baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_baseline(args.episodes, args.seed, args.out)
