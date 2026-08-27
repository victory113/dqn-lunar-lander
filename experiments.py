"""Multi-seed experiment runner and ablation harness.

Reinforcement learning results vary enormously between random seeds. A single
run that reaches 250 tells you almost nothing about whether the method works —
the next seed might plateau at 40. Reporting a mean and spread across several
seeds is the difference between a demo and a result.

    python experiments.py --seeds 42 43 44 --episodes 700
    python experiments.py --ablate --seeds 42 43 44

``--ablate`` trains Double DQN and vanilla DQN under identical seeds and
episode budgets, so the only difference between the two arms is the Bellman
target. Anything else would confound the comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from dqn.config import DEFAULT_CONFIG, Config
from evaluate import evaluate
from train import train


def _aggregate(label: str, runs: list[dict]) -> dict:
    """Collapse per-seed results into mean/std/min/max."""
    finals = np.array([r["eval_mean"] for r in runs], dtype=np.float64)
    solved_at = [r["solved_at_episode"] for r in runs if r["solved_at_episode"] is not None]

    return {
        "variant": label,
        "seeds": [r["seed"] for r in runs],
        "n_seeds": len(runs),
        "eval_mean_across_seeds": round(float(finals.mean()), 1),
        "eval_std_across_seeds": round(float(finals.std()), 1),
        "eval_min_seed": round(float(finals.min()), 1),
        "eval_max_seed": round(float(finals.max()), 1),
        "seeds_solved": len(solved_at),
        "median_episode_to_solve": int(np.median(solved_at)) if solved_at else None,
        "per_seed": runs,
    }


def run_variant(
    label: str, base: Config, seeds: list[int], out_root: Path, eval_episodes: int
) -> dict:
    runs = []
    for seed in seeds:
        print(f"\n{'=' * 66}\n{label} | seed {seed}\n{'=' * 66}")
        config = replace(base, seed=seed)
        out_dir = out_root / label / f"seed_{seed}"

        summary = train(config, out_dir)
        evaluation = evaluate(
            out_dir / "dqn_lunarlander.pth", config, eval_episodes, False, out_dir
        )

        runs.append(
            {
                "seed": seed,
                "solved_at_episode": summary["solved_at_episode"],
                "final_train_avg": summary["final_avg_score"],
                "eval_mean": evaluation["mean"],
                "eval_success_rate_pct": evaluation["success_rate_pct"],
            }
        )

    aggregate = _aggregate(label, runs)
    (out_root / label / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed experiments and ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG.num_episodes)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--ablate", action="store_true", help="also train vanilla DQN for comparison"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/experiments"))
    args = parser.parse_args()

    base = replace(DEFAULT_CONFIG, num_episodes=args.episodes)
    results = [
        run_variant(
            "double_dqn", replace(base, double_dqn=True), args.seeds, args.out, args.eval_episodes
        )
    ]

    if args.ablate:
        results.append(
            run_variant(
                "vanilla_dqn",
                replace(base, double_dqn=False),
                args.seeds,
                args.out,
                args.eval_episodes,
            )
        )

    print(f"\n{'=' * 66}\nRESULTS ACROSS {len(args.seeds)} SEEDS\n{'=' * 66}")
    print(f"{'variant':<14}{'eval mean':>12}{'std':>8}{'solved':>9}{'med. ep':>10}")
    for r in results:
        med = r["median_episode_to_solve"]
        print(
            f"{r['variant']:<14}"
            f"{r['eval_mean_across_seeds']:>12.1f}"
            f"{r['eval_std_across_seeds']:>8.1f}"
            f"{str(r['seeds_solved']) + '/' + str(r['n_seeds']):>9}"
            f"{(med if med is not None else '-'):>10}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {(args.out / 'comparison.json').resolve()}")


if __name__ == "__main__":
    main()
