"""
Gaussian/OU exploration-noise ablation for the current Keras-3 DDPG environment.

Place this file in the same directory as:
    ddpg_ga_keras3_same_env.py

It compares:
    no_noise
    gaussian_only
    ou_only
    gaussian_ou

The four variants share:
- the same DDPG actor/critic implementation;
- the same fixed environment trajectories;
- the same replay warm-up, reward, hyperparameters, and training horizon;
- the same total target exploration RMS and decay schedule.

Outputs are written to:
    data/noise_ablation/
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ddpg_ga_keras3_same_env import (
    Config as BaseConfig,
    DDPGAgent,
    FixedTrajectoryBank,
    energy_consumption,
    jain_fairness_index,
    reward_function,
    set_global_seed,
)

NoiseMode = Literal[
    "no_noise",
    "gaussian_only",
    "ou_only",
    "gaussian_ou",
]

NOISE_MODES: tuple[NoiseMode, ...] = (
    "no_noise",
    "gaussian_only",
    "ou_only",
    "gaussian_ou",
)

EPS = 1e-8


@dataclass(frozen=True)
class AblationConfig:
    # Use 3 seeds as a practical minimum. Change to (0, 1, 2, 3, 4) for 5 seeds.
    seeds: tuple[int, ...] = (0, 1, 2)

    # Matched total policy-noise RMS, linearly decayed during training.
    total_rms_start: float = 0.20
    total_rms_end: float = 0.05

    # OU process parameters.
    ou_theta: float = 0.15
    ou_dt: float = 1.0

    # Analysis settings.
    rolling_window: int = 50
    final_window: int = 100
    convergence_fraction: float = 0.95
    convergence_hold: int = 100

    # Independent random streams.
    environment_seed: int = 0
    warmup_seed_offset: int = 20_000
    exploration_seed_offset: int = 30_000
    replay_seed_offset: int = 40_000

    output_dir: str = "data/noise_ablation"
    quick_test: bool = False


ABL = AblationConfig()


def linear_schedule(
    start: float,
    end: float,
    step: int,
    total_steps: int,
) -> float:
    ratio = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return float(start + ratio * (end - start))


class MatchedPolicyNoise:
    """Gaussian, OU, or matched Gaussian+OU policy noise.

    The composite variant allocates half of the target total variance to each
    component, so it is not favored merely because two noises are added.

    The discrete OU process is:
        x_(t+1) = (1-theta*dt)x_t + sigma_diff*sqrt(dt)*epsilon_t

    Its stationary variance is:
        sigma_diff^2*dt / (1-(1-theta*dt)^2)
    """

    def __init__(
        self,
        mode: NoiseMode,
        action_dim: int,
        total_steps: int,
        seed: int,
        cfg: AblationConfig,
    ) -> None:
        self.mode = mode
        self.action_dim = int(action_dim)
        self.total_steps = max(int(total_steps), 1)
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.ou_state = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.ou_state.fill(0.0)

    def current_total_rms(self, global_step: int) -> float:
        return linear_schedule(
            self.cfg.total_rms_start,
            self.cfg.total_rms_end,
            global_step,
            self.total_steps,
        )

    def _ou_diffusion_sigma(self, stationary_std: float) -> float:
        theta_dt = self.cfg.ou_theta * self.cfg.ou_dt
        denominator = max(1.0 - (1.0 - theta_dt) ** 2, EPS)
        return float(
            stationary_std
            * math.sqrt(denominator / self.cfg.ou_dt)
        )

    def sample(self, global_step: int) -> np.ndarray:
        if self.mode == "no_noise":
            return np.zeros(self.action_dim, dtype=np.float32)

        total_rms = self.current_total_rms(global_step)

        if self.mode == "gaussian_only":
            gaussian_std = total_rms
            ou_stationary_std = 0.0
        elif self.mode == "ou_only":
            gaussian_std = 0.0
            ou_stationary_std = total_rms
        elif self.mode == "gaussian_ou":
            component_std = total_rms / math.sqrt(2.0)
            gaussian_std = component_std
            ou_stationary_std = component_std
        else:
            raise ValueError(f"Unsupported noise mode: {self.mode}")

        gaussian = self.rng.normal(
            loc=0.0,
            scale=gaussian_std,
            size=self.action_dim,
        ).astype(np.float32)

        if ou_stationary_std <= 0.0:
            return gaussian

        diffusion_sigma = self._ou_diffusion_sigma(ou_stationary_std)
        random_term = self.rng.randn(self.action_dim).astype(np.float32)
        drift = (
            -self.cfg.ou_theta
            * self.ou_state
            * self.cfg.ou_dt
        )
        diffusion = (
            diffusion_sigma
            * math.sqrt(self.cfg.ou_dt)
            * random_term
        )
        self.ou_state = self.ou_state + drift + diffusion
        return gaussian + self.ou_state


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    valid = np.convolve(values, kernel, mode="valid")
    return np.concatenate([
        np.full(window - 1, np.nan, dtype=np.float64),
        valid,
    ])


def convergence_episode(
    rewards: np.ndarray,
    cfg: AblationConfig,
) -> float:
    rewards = np.asarray(rewards, dtype=np.float64)
    if len(rewards) < cfg.final_window:
        return float("nan")

    final_mean = float(np.mean(rewards[-cfg.final_window:]))
    threshold = cfg.convergence_fraction * final_mean
    smooth = rolling_mean(rewards, cfg.rolling_window)

    for start in range(len(smooth)):
        end = min(start + cfg.convergence_hold, len(smooth))
        segment = smooth[start:end]
        segment = segment[~np.isnan(segment)]
        if len(segment) < min(cfg.convergence_hold, len(smooth) - start):
            continue
        if np.all(segment >= threshold):
            return float(start + 1)

    return float("nan")


def normalized_auc(rewards: np.ndarray) -> float:
    rewards = np.asarray(rewards, dtype=np.float64)
    if len(rewards) < 2:
        return float("nan")
    return float(np.trapz(rewards) / (len(rewards) - 1))


def run_one(
    mode: NoiseMode,
    seed: int,
    base_cfg: BaseConfig,
    ablation_cfg: AblationConfig,
    trajectories: FixedTrajectoryBank,
) -> dict[str, object]:
    set_global_seed(seed)

    max_episodes = 20 if ablation_cfg.quick_test else base_cfg.max_episodes
    max_steps = 20 if ablation_cfg.quick_test else base_cfg.max_episode_steps
    warmup_steps = 100 if ablation_cfg.quick_test else base_cfg.warmup_steps
    total_steps = max_episodes * max_steps

    original_state_dim = 4 + 2 * base_cfg.num_users
    combined_state_dim = 2 * original_state_dim

    # Give each seed an independent replay RNG while keeping it matched across modes.
    run_cfg = BaseConfig(
        **{
            **base_cfg.__dict__,
            "master_seed": seed,
            "environment_seed": ablation_cfg.environment_seed,
            "exploration_seed": ablation_cfg.exploration_seed_offset + seed,
            "replay_seed": ablation_cfg.replay_seed_offset + seed,
            "ga_seed": 50_000 + seed,
            "quick_test": ablation_cfg.quick_test,
        }
    )

    agent = DDPGAgent(
        state_dim=combined_state_dim,
        action_dim=run_cfg.action_dim,
        cfg=run_cfg,
    )

    policy_noise = MatchedPolicyNoise(
        mode=mode,
        action_dim=run_cfg.action_dim,
        total_steps=total_steps,
        seed=ablation_cfg.exploration_seed_offset + seed,
        cfg=ablation_cfg,
    )
    warmup_rng = np.random.RandomState(
        ablation_cfg.warmup_seed_offset + seed
    )

    episode_rewards: list[float] = []
    episode_fairness_mean: list[float] = []
    episode_energy_mean: list[float] = []
    episode_action_variation: list[float] = []
    episode_clip_rate: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []

    global_step = 0
    start_time = time.time()

    for episode in range(max_episodes):
        user_positions, other_state = trajectories.reset(episode)
        current_state = np.concatenate([
            other_state.flatten(),
            user_positions.flatten(),
        ]).astype(np.float32)
        previous_state = np.zeros_like(current_state)
        combined_state = np.concatenate([
            current_state,
            previous_state,
        ]).astype(np.float32)

        policy_noise.reset()
        episode_reward = 0.0
        fairness_values: list[float] = []
        energy_values: list[float] = []
        action_deltas: list[float] = []
        clipped_flags: list[float] = []
        previous_action: np.ndarray | None = None
        latest_losses: dict[str, float] | None = None

        for step in range(max_steps):
            if global_step < warmup_steps:
                action = warmup_rng.uniform(
                    low=0.0,
                    high=run_cfg.action_bound,
                    size=(run_cfg.action_dim,),
                ).astype(np.float32)
                clipped = False
            else:
                deterministic_action = agent.choose_action(combined_state)
                unbounded_action = (
                    deterministic_action
                    + policy_noise.sample(global_step)
                )
                action = np.clip(
                    unbounded_action,
                    0.0,
                    run_cfg.action_bound,
                ).astype(np.float32)
                clipped = bool(
                    np.any(np.abs(action - unbounded_action) > 1e-7)
                )

            user_delta, other_delta = trajectories.step_delta(episode, step)
            user_positions = user_positions + user_delta
            other_state = other_state + other_delta

            next_current_state = np.concatenate([
                other_state.flatten(),
                user_positions.flatten(),
            ]).astype(np.float32)
            next_combined_state = np.concatenate([
                next_current_state,
                current_state,
            ]).astype(np.float32)

            distances = np.sqrt(
                np.sum(user_positions ** 2, axis=1)
            )
            channel_gains = 1.0 / (1.0 + distances)

            reward = reward_function(
                channel_gains,
                action,
                run_cfg.energy_weight,
            )
            fairness = jain_fairness_index(channel_gains)
            energy = energy_consumption(action)
            done = step == max_steps - 1

            agent.store_transition(
                combined_state,
                action,
                reward,
                next_combined_state,
                done,
            )

            if (
                agent.memory.size >= max(warmup_steps, run_cfg.batch_size)
                and global_step % 1 == 0
            ):
                latest_losses = agent.learn()
                actor_losses.append(latest_losses["actor_loss"])
                critic_losses.append(latest_losses["critic_loss"])

            if previous_action is not None:
                action_deltas.append(
                    float(np.mean(np.abs(action - previous_action)))
                )
            previous_action = action.copy()

            episode_reward += reward
            fairness_values.append(fairness)
            energy_values.append(energy)
            clipped_flags.append(float(clipped))

            current_state = next_current_state.copy()
            combined_state = next_combined_state.copy()
            global_step += 1

        episode_rewards.append(float(episode_reward))
        episode_fairness_mean.append(float(np.mean(fairness_values)))
        episode_energy_mean.append(float(np.mean(energy_values)))
        episode_action_variation.append(
            float(np.mean(action_deltas)) if action_deltas else 0.0
        )
        episode_clip_rate.append(float(np.mean(clipped_flags)))

        loss_text = (
            "warming up"
            if latest_losses is None
            else (
                f"actor={latest_losses['actor_loss']:.4f}, "
                f"critic={latest_losses['critic_loss']:.4f}"
            )
        )
        print(
            f"[{mode:13s}] seed={seed} "
            f"Episode {episode + 1:4d}/{max_episodes} | "
            f"Reward={episode_reward:10.4f} | "
            f"Fairness={episode_fairness_mean[-1]:.4f} | "
            f"Energy={episode_energy_mean[-1]:.4f} | "
            f"ActionDelta={episode_action_variation[-1]:.4f} | "
            f"{loss_text}"
        )

    runtime = time.time() - start_time
    rewards = np.asarray(episode_rewards, dtype=np.float32)
    final_count = min(ablation_cfg.final_window, len(rewards))

    return {
        "noise_mode": np.asarray(mode),
        "seed": np.int32(seed),
        "ep_rewardall": rewards,
        "ep_fairness_mean": np.asarray(
            episode_fairness_mean, dtype=np.float32
        ),
        "ep_energy_mean": np.asarray(
            episode_energy_mean, dtype=np.float32
        ),
        "ep_action_variation": np.asarray(
            episode_action_variation, dtype=np.float32
        ),
        "ep_action_clip_rate": np.asarray(
            episode_clip_rate, dtype=np.float32
        ),
        "actor_losses": np.asarray(actor_losses, dtype=np.float32),
        "critic_losses": np.asarray(critic_losses, dtype=np.float32),
        "last100_reward_mean": np.float32(
            np.mean(rewards[-final_count:])
        ),
        "last100_reward_std": np.float32(
            np.std(rewards[-final_count:], ddof=1)
        ),
        "convergence_episode": np.float32(
            convergence_episode(rewards, ablation_cfg)
        ),
        "normalized_auc": np.float32(normalized_auc(rewards)),
        "runtime_seconds": np.float32(runtime),
    }


def save_run(output_dir: Path, result: dict[str, object]) -> Path:
    mode = str(np.asarray(result["noise_mode"]).item())
    seed = int(np.asarray(result["seed"]).item())
    run_dir = output_dir / "runs" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"seed_{seed}.npz"
    np.savez_compressed(path, **result)
    return path


def summarize(
    results: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, float | int | str]] = []

    for result in results:
        run_rows.append({
            "Noise configuration": str(
                np.asarray(result["noise_mode"]).item()
            ),
            "Seed": int(np.asarray(result["seed"]).item()),
            "Last-100 reward": float(result["last100_reward_mean"]),
            "Last-100 reward std": float(result["last100_reward_std"]),
            "Convergence episode": float(result["convergence_episode"]),
            "Normalized AUC": float(result["normalized_auc"]),
            "Mean Jain fairness": float(
                np.mean(result["ep_fairness_mean"])
            ),
            "Mean energy": float(np.mean(result["ep_energy_mean"])),
            "Mean action variation": float(
                np.mean(result["ep_action_variation"])
            ),
            "Mean action clipping rate": float(
                np.mean(result["ep_action_clip_rate"])
            ),
            "Runtime (s)": float(result["runtime_seconds"]),
        })

    run_df = pd.DataFrame(run_rows)
    summary_rows: list[dict[str, float | int | str]] = []

    metric_columns = [
        column
        for column in run_df.columns
        if column not in ("Noise configuration", "Seed")
    ]

    for mode in NOISE_MODES:
        group = run_df[run_df["Noise configuration"] == mode]
        row: dict[str, float | int | str] = {
            "Noise configuration": mode,
            "Runs": len(group),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column} mean"] = float(values.mean())
            row[f"{column} std"] = float(values.std(ddof=1))
        summary_rows.append(row)

    return run_df, pd.DataFrame(summary_rows)


def plot_metric(
    results: list[dict[str, object]],
    key: str,
    ylabel: str,
    output_path: Path,
    rolling_window_size: int,
) -> None:
    plt.figure(figsize=(9, 6))

    for mode in NOISE_MODES:
        runs = [
            np.asarray(result[key], dtype=np.float64)
            for result in results
            if str(np.asarray(result["noise_mode"]).item()) == mode
        ]
        matrix = np.stack(runs, axis=0)

        if rolling_window_size > 1:
            matrix = np.stack([
                rolling_mean(row, rolling_window_size)
                for row in matrix
            ])

        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0, ddof=1)
        episodes = np.arange(1, len(mean) + 1)

        plt.plot(episodes, mean, label=mode)
        plt.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.18,
        )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    base_cfg = BaseConfig()
    output_dir = Path(ABL.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_episodes = 20 if ABL.quick_test else base_cfg.max_episodes
    max_steps = 20 if ABL.quick_test else base_cfg.max_episode_steps

    trajectory_path = output_dir / (
        f"fixed_trajectories_K{base_cfg.num_users}_"
        f"E{max_episodes}_T{max_steps}_"
        f"seed{ABL.environment_seed}.npz"
    )
    trajectories = FixedTrajectoryBank(
        path=str(trajectory_path),
        episodes=max_episodes,
        steps=max_steps,
        num_users=base_cfg.num_users,
        seed=ABL.environment_seed,
    )

    all_results: list[dict[str, object]] = []

    for mode in NOISE_MODES:
        for seed in ABL.seeds:
            print("\n" + "=" * 80)
            print(f"Running noise mode={mode}, seed={seed}")
            print("=" * 80)

            result = run_one(
                mode=mode,
                seed=seed,
                base_cfg=base_cfg,
                ablation_cfg=ABL,
                trajectories=trajectories,
            )
            path = save_run(output_dir, result)
            print("Saved run:", path)
            all_results.append(result)

    run_df, summary_df = summarize(all_results)
    run_csv = output_dir / "noise_ablation_run_summary.csv"
    summary_csv = output_dir / "noise_ablation_summary.csv"
    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    plot_metric(
        results=all_results,
        key="ep_rewardall",
        ylabel="Episode return",
        output_path=output_dir / "noise_ablation_reward_curve.png",
        rolling_window_size=ABL.rolling_window,
    )
    plot_metric(
        results=all_results,
        key="ep_action_variation",
        ylabel="Mean consecutive-action variation",
        output_path=(
            output_dir / "noise_ablation_action_variation_curve.png"
        ),
        rolling_window_size=ABL.rolling_window,
    )

    print("\nRun-level results:")
    print(run_df.to_string(index=False))
    print("\nConfiguration-level summary:")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(run_csv)
    print(summary_csv)
    print(output_dir / "noise_ablation_reward_curve.png")
    print(output_dir / "noise_ablation_action_variation_curve.png")


if __name__ == "__main__":
    main()
