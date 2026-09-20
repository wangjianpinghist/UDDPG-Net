"""
Fast EAM reward-weight sensitivity experiment for the user's Keras-3 DDPG codebase.

Place this file in the same directory as:
    ddpg_ga_keras3_same_env.py

It reuses:
- DDPGAgent
- FixedTrajectoryBank
- Config
- set_global_seed

Why this script does not directly reuse reward_function() from the current baseline:
The current simplified baseline computes Jain fairness from channel gains only and uses one
scalar action. In that setting, changing a fairness weight cannot meaningfully change the
allocation fairness. This runner therefore keeps the same fixed-trajectory and actor-critic
infrastructure, but uses a compact per-user power + subband action and computes Jain fairness
from the achieved per-link rates.

Profiles:
    smoke : dependency/compatibility check
    fast  : short 3-seed sensitivity study (default)
    paper : longer 5-seed study

Outputs:
    data/eam_weight_sensitivity_fast/run_level_results.csv
    data/eam_weight_sensitivity_fast/table6_summary.csv
    data/eam_weight_sensitivity_fast/table6_paper.csv
    data/eam_weight_sensitivity_fast/throughput_fairness_tradeoff.csv
    data/eam_weight_sensitivity_fast/weight_sensitivity.png
    data/eam_weight_sensitivity_fast/throughput_fairness_tradeoff.png
    data/eam_weight_sensitivity_fast/runs/*.npz

Example:
    python fast_eam_weight_sensitivity.py --profile fast

Important:
1. Set BASE_ENERGY_WEIGHT, BASE_FAIRNESS_WEIGHT, BASE_THROUGHPUT_WEIGHT,
   and BASE_VIOLATION_WEIGHT to the exact default values used in the manuscript.
2. The "fast" profile is intended for a rapid controlled sensitivity check. Before final
   submission, rerun with --profile paper if time permits.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ddpg_ga_keras3_same_env import (
    Config as BaseConfig,
    DDPGAgent,
    FixedTrajectoryBank,
    set_global_seed,
)

EPS = 1e-8

# Replace these values with the exact default EAM coefficients used by UDDPG-Net.
BASE_THROUGHPUT_WEIGHT = 1.0
BASE_ENERGY_WEIGHT = 0.10
BASE_FAIRNESS_WEIGHT = 1.0
BASE_VIOLATION_WEIGHT = 1.0


@dataclass(frozen=True)
class Profile:
    seeds: tuple[int, ...]
    train_episodes: int
    train_steps: int
    eval_episodes: int
    eval_steps: int
    hidden_dim: int
    warmup_steps: int
    update_every: int
    final_window: int
    rolling_window: int
    convergence_hold: int


PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        seeds=(0,), train_episodes=12, train_steps=10,
        eval_episodes=3, eval_steps=10, hidden_dim=32,
        warmup_steps=40, update_every=2, final_window=5,
        rolling_window=3, convergence_hold=3,
    ),
    "fast": Profile(
        seeds=(0, 1, 2), train_episodes=120, train_steps=25,
        eval_episodes=12, eval_steps=25, hidden_dim=128,
        warmup_steps=200, update_every=2, final_window=20,
        rolling_window=15, convergence_hold=15,
    ),
    "paper": Profile(
        seeds=(0, 1, 2, 3, 4), train_episodes=400, train_steps=50,
        eval_episodes=30, eval_steps=50, hidden_dim=128,
        warmup_steps=1000, update_every=1, final_window=50,
        rolling_window=30, convergence_hold=30,
    ),
}


@dataclass(frozen=True)
class WeightSetting:
    key: str
    group: str
    multiplier: float
    throughput_weight: float
    energy_weight: float
    fairness_weight: float
    violation_weight: float


def build_weight_settings(
    throughput_weight: float,
    energy_weight: float,
    fairness_weight: float,
    violation_weight: float,
) -> tuple[WeightSetting, ...]:
    return (
        WeightSetting("energy_0.5x", "Energy weight", 0.5,
                      throughput_weight, 0.5 * energy_weight,
                      fairness_weight, violation_weight),
        WeightSetting("default", "Default", 1.0,
                      throughput_weight, energy_weight,
                      fairness_weight, violation_weight),
        WeightSetting("energy_1.5x", "Energy weight", 1.5,
                      throughput_weight, 1.5 * energy_weight,
                      fairness_weight, violation_weight),
        WeightSetting("fairness_0.0x", "Fairness weight", 0.0,
                      throughput_weight, energy_weight, 0.0,
                      violation_weight),
        WeightSetting("fairness_0.5x", "Fairness weight", 0.5,
                      throughput_weight, energy_weight,
                      0.5 * fairness_weight, violation_weight),
        WeightSetting("fairness_1.5x", "Fairness weight", 1.5,
                      throughput_weight, energy_weight,
                      1.5 * fairness_weight, violation_weight),
        WeightSetting("fairness_2.0x", "Fairness weight", 2.0,
                      throughput_weight, energy_weight,
                      2.0 * fairness_weight, violation_weight),
    )


class CompositeNoise:
    def __init__(self, dim: int, total_steps: int, seed: int,
                 std_start: float = 0.18, std_end: float = 0.04,
                 theta: float = 0.15) -> None:
        self.dim = int(dim)
        self.total_steps = max(int(total_steps), 1)
        self.std_start = float(std_start)
        self.std_end = float(std_end)
        self.theta = float(theta)
        self.rng = np.random.RandomState(seed)
        self.ou_state = np.zeros(self.dim, dtype=np.float32)

    def reset(self) -> None:
        self.ou_state.fill(0.0)

    def sample(self, global_step: int) -> np.ndarray:
        ratio = min(max(global_step / self.total_steps, 0.0), 1.0)
        total_std = self.std_start + ratio * (self.std_end - self.std_start)
        component_std = total_std / math.sqrt(2.0)
        gaussian = self.rng.normal(0.0, component_std, size=self.dim).astype(np.float32)
        self.ou_state = (
            self.ou_state + self.theta * (-self.ou_state)
            + component_std * self.rng.randn(self.dim).astype(np.float32)
        )
        return gaussian + self.ou_state


class FastUnderwaterAllocation:
    """Compact rate-based adapter with per-user power and subband decisions."""

    def __init__(self, num_users: int, num_subbands: int = 4,
                 bandwidth_khz: float = 10.0, noise_power: float = 0.06,
                 circuit_energy: float = 0.003,
                 tx_energy_scale: float = 0.020,
                 initial_battery: float = 1.0) -> None:
        self.num_users = int(num_users)
        self.num_subbands = int(num_subbands)
        self.bandwidth_khz = float(bandwidth_khz)
        self.noise_power = float(noise_power)
        self.circuit_energy = float(circuit_energy)
        self.tx_energy_scale = float(tx_energy_scale)
        self.initial_battery = float(initial_battery)
        self.battery = np.full(self.num_users, self.initial_battery, dtype=np.float32)

    @property
    def action_dim(self) -> int:
        return 2 * self.num_users

    def reset(self) -> None:
        self.battery.fill(self.initial_battery)

    def battery_summary(self, other_state: np.ndarray) -> np.ndarray:
        x = np.asarray(other_state, dtype=np.float32).copy()
        x[0] = float(np.mean(self.battery))
        x[1] = float(np.min(self.battery))
        return x

    def _map_action(self, raw_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action.size}.")
        candidate_power = np.clip(action[:self.num_users], 0.0, 1.0)
        subband_scores = np.clip(action[self.num_users:], 0.0, 1.0 - 1e-7)
        subbands = np.floor(subband_scores * self.num_subbands).astype(np.int32)

        predicted_energy = self.circuit_energy + self.tx_energy_scale * candidate_power ** 2
        violation = np.maximum(predicted_energy - self.battery, 0.0)
        violation_rate = float(np.mean(violation > 0.0))

        feasible_tx_energy = np.maximum(self.battery - self.circuit_energy, 0.0)
        feasible_power = np.sqrt(feasible_tx_energy / max(self.tx_energy_scale, EPS))
        executed_power = np.minimum(candidate_power, feasible_power)
        return np.clip(executed_power, 0.0, 1.0).astype(np.float32), subbands, violation_rate

    def step_metrics(self, user_positions: np.ndarray, raw_action: np.ndarray,
                     setting: WeightSetting) -> dict[str, float]:
        powers, subbands, violation_rate = self._map_action(raw_action)
        positions = np.asarray(user_positions, dtype=np.float32)
        distances_to_receiver = np.sqrt(np.sum(positions ** 2, axis=1))
        direct_gains = 1.0 / (1.0 + distances_to_receiver)

        pairwise = positions[:, None, :] - positions[None, :, :]
        pairwise_dist = np.sqrt(np.sum(pairwise ** 2, axis=2))
        cross_gains = 1.0 / (1.0 + pairwise_dist)
        np.fill_diagonal(cross_gains, 0.0)

        rates = np.zeros(self.num_users, dtype=np.float32)
        for i in range(self.num_users):
            same_band = (subbands == subbands[i])
            same_band[i] = False
            interference = float(np.sum(0.20 * powers[same_band] * cross_gains[i, same_band]))
            signal = float(powers[i] * direct_gains[i])
            sinr = signal / (self.noise_power + interference + EPS)
            rates[i] = self.bandwidth_khz * np.log2(1.0 + sinr)

        throughput = float(np.sum(rates))
        average_rate = float(np.mean(rates))
        fairness = float((np.sum(rates) ** 2) /
                         (self.num_users * np.sum(rates ** 2) + EPS))
        energy_per_user = self.circuit_energy + self.tx_energy_scale * powers ** 2
        mean_energy = float(np.mean(energy_per_user))
        self.battery = np.maximum(self.battery - energy_per_user, 0.0).astype(np.float32)

        throughput_norm = float(np.tanh(throughput / (self.num_users * self.bandwidth_khz)))
        energy_norm = float(np.mean(powers ** 2))
        reward = float(
            setting.throughput_weight * throughput_norm
            + setting.fairness_weight * fairness
            - setting.energy_weight * energy_norm
            - setting.violation_weight * violation_rate
        )
        return {
            "reward": reward,
            "throughput_kbps": throughput,
            "average_rate_kbps": average_rate,
            "jain_fairness": fairness,
            "mean_energy": mean_energy,
            "pre_projection_violation_rate": violation_rate,
            "remaining_energy_ratio": float(np.mean(self.battery) / self.initial_battery),
        }


def build_state(env: FastUnderwaterAllocation, user_positions: np.ndarray,
                other_state: np.ndarray) -> np.ndarray:
    return np.concatenate([
        env.battery_summary(other_state).flatten(),
        user_positions.flatten(),
    ]).astype(np.float32)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=np.float64)
    valid = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), valid])


def convergence_episode(rewards: np.ndarray, profile: Profile) -> float:
    rewards = np.asarray(rewards, dtype=np.float64)
    if len(rewards) < profile.final_window:
        return float("nan")
    threshold = 0.95 * float(np.mean(rewards[-profile.final_window:]))
    smooth = rolling_mean(rewards, profile.rolling_window)
    for start in range(len(smooth)):
        end = min(start + profile.convergence_hold, len(smooth))
        segment = smooth[start:end]
        segment = segment[np.isfinite(segment)]
        required = min(profile.convergence_hold, max(len(smooth) - start, 0))
        if required > 0 and len(segment) >= required and np.all(segment >= threshold):
            return float(start + 1)
    return float("nan")


def make_base_config(seed: int, profile: Profile, action_dim: int) -> BaseConfig:
    base = BaseConfig()
    return BaseConfig(**{
        **base.__dict__,
        "max_episodes": profile.train_episodes,
        "max_episode_steps": profile.train_steps,
        "hidden_dim": profile.hidden_dim,
        "action_dim": action_dim,
        "replay_capacity": 20000,
        "batch_size": 64,
        "warmup_steps": profile.warmup_steps,
        "master_seed": seed,
        "environment_seed": 0,
        "exploration_seed": 10000 + seed,
        "replay_seed": 20000 + seed,
        "ga_seed": 30000 + seed,
        "quick_test": False,
    })


def train_one(setting: WeightSetting, seed: int, profile: Profile,
              train_trajectories: FixedTrajectoryBank) -> tuple[DDPGAgent, dict[str, object]]:
    set_global_seed(seed)
    env = FastUnderwaterAllocation(num_users=train_trajectories.num_users)
    cfg = make_base_config(seed, profile, env.action_dim)
    original_state_dim = 4 + 2 * cfg.num_users
    combined_state_dim = 2 * original_state_dim
    agent = DDPGAgent(state_dim=combined_state_dim, action_dim=env.action_dim, cfg=cfg)

    total_steps = profile.train_episodes * profile.train_steps
    noise = CompositeNoise(env.action_dim, total_steps, 10000 + seed)
    warmup_rng = np.random.RandomState(40000 + seed)

    episode_rewards, episode_throughput = [], []
    episode_fairness, episode_energy, episode_violation = [], [], []
    global_step = 0
    start_time = time.time()

    for episode in range(profile.train_episodes):
        user_positions, other_state = train_trajectories.reset(episode)
        env.reset(); noise.reset()
        current_state = build_state(env, user_positions, other_state)
        combined_state = np.concatenate([current_state, np.zeros_like(current_state)]).astype(np.float32)

        reward_values, throughput_values = [], []
        fairness_values, energy_values, violation_values = [], [], []

        for step in range(profile.train_steps):
            if global_step < profile.warmup_steps:
                action = warmup_rng.uniform(0.0, 1.0, size=env.action_dim).astype(np.float32)
            else:
                action = np.clip(agent.choose_action(combined_state) + noise.sample(global_step),
                                 0.0, 1.0).astype(np.float32)

            user_delta, other_delta = train_trajectories.step_delta(episode, step)
            user_positions = user_positions + user_delta
            other_state = other_state + other_delta
            metrics = env.step_metrics(user_positions, action, setting)

            next_current_state = build_state(env, user_positions, other_state)
            next_combined_state = np.concatenate([next_current_state, current_state]).astype(np.float32)
            done = step == profile.train_steps - 1
            agent.store_transition(combined_state, action, metrics["reward"], next_combined_state, done)

            if (agent.memory.size >= max(profile.warmup_steps, cfg.batch_size)
                    and global_step % profile.update_every == 0):
                agent.learn()

            reward_values.append(metrics["reward"])
            throughput_values.append(metrics["throughput_kbps"])
            fairness_values.append(metrics["jain_fairness"])
            energy_values.append(metrics["mean_energy"])
            violation_values.append(metrics["pre_projection_violation_rate"])
            current_state = next_current_state.copy()
            combined_state = next_combined_state.copy()
            global_step += 1

        episode_rewards.append(float(np.sum(reward_values)))
        episode_throughput.append(float(np.mean(throughput_values)))
        episode_fairness.append(float(np.mean(fairness_values)))
        episode_energy.append(float(np.mean(energy_values)))
        episode_violation.append(float(np.mean(violation_values)))

        if (episode == 0 or (episode + 1) % max(profile.train_episodes // 4, 1) == 0
                or episode + 1 == profile.train_episodes):
            print(f"[{setting.key:15s}] seed={seed} episode={episode+1:4d}/{profile.train_episodes} "
                  f"reward={episode_rewards[-1]:8.3f} thr={episode_throughput[-1]:8.3f} kbps "
                  f"Jain={episode_fairness[-1]:.4f} energy={episode_energy[-1]:.5f}")

    rewards = np.asarray(episode_rewards, dtype=np.float32)
    result = {
        "setting": np.asarray(setting.key), "group": np.asarray(setting.group),
        "multiplier": np.float32(setting.multiplier), "seed": np.int32(seed),
        "throughput_weight": np.float32(setting.throughput_weight),
        "energy_weight": np.float32(setting.energy_weight),
        "fairness_weight": np.float32(setting.fairness_weight),
        "violation_weight": np.float32(setting.violation_weight),
        "ep_reward": rewards,
        "ep_throughput_kbps": np.asarray(episode_throughput, dtype=np.float32),
        "ep_jain_fairness": np.asarray(episode_fairness, dtype=np.float32),
        "ep_mean_energy": np.asarray(episode_energy, dtype=np.float32),
        "ep_violation_rate": np.asarray(episode_violation, dtype=np.float32),
        "convergence_episode": np.float32(convergence_episode(rewards, profile)),
        "runtime_seconds": np.float32(time.time() - start_time),
    }
    return agent, result


def evaluate_one(agent: DDPGAgent, setting: WeightSetting,
                 eval_trajectories: FixedTrajectoryBank,
                 profile: Profile) -> dict[str, float]:
    env = FastUnderwaterAllocation(num_users=eval_trajectories.num_users)
    throughput_values, average_rate_values = [], []
    fairness_values, energy_values, violation_values, remaining_values = [], [], [], []

    for episode in range(profile.eval_episodes):
        user_positions, other_state = eval_trajectories.reset(episode)
        env.reset()
        current_state = build_state(env, user_positions, other_state)
        combined_state = np.concatenate([current_state, np.zeros_like(current_state)]).astype(np.float32)

        for step in range(profile.eval_steps):
            action = agent.choose_action(combined_state)
            user_delta, other_delta = eval_trajectories.step_delta(episode, step)
            user_positions = user_positions + user_delta
            other_state = other_state + other_delta
            metrics = env.step_metrics(user_positions, action, setting)
            throughput_values.append(metrics["throughput_kbps"])
            average_rate_values.append(metrics["average_rate_kbps"])
            fairness_values.append(metrics["jain_fairness"])
            energy_values.append(metrics["mean_energy"])
            violation_values.append(metrics["pre_projection_violation_rate"])
            remaining_values.append(metrics["remaining_energy_ratio"])
            next_current_state = build_state(env, user_positions, other_state)
            combined_state = np.concatenate([next_current_state, current_state]).astype(np.float32)
            current_state = next_current_state.copy()

    return {
        "Average throughput (kbps)": float(np.mean(throughput_values)),
        "Average per-link rate (kbps)": float(np.mean(average_rate_values)),
        "Jain fairness index": float(np.mean(fairness_values)),
        "Average energy consumption": float(np.mean(energy_values)),
        "Pre-projection violation rate": float(np.mean(violation_values)),
        "Final remaining-energy ratio": float(np.mean(remaining_values[-profile.eval_steps:])),
    }


def fmt_mean_std(mean_value: float, std_value: float, digits: int) -> str:
    return (f"{mean_value:.{digits}f}" if not np.isfinite(std_value)
            else f"{mean_value:.{digits}f} ± {std_value:.{digits}f}")


def summarize(run_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "Average throughput (kbps)", "Average per-link rate (kbps)",
        "Jain fairness index", "Average energy consumption",
        "Convergence episode", "Pre-projection violation rate",
        "Final remaining-energy ratio", "Runtime (s)",
    ]
    summary_rows, paper_rows = [], []
    for setting in list(dict.fromkeys(run_df["Setting"].tolist())):
        group = run_df[run_df["Setting"] == setting]
        row = {"Setting": setting, "Group": group["Group"].iloc[0],
               "Multiplier": float(group["Multiplier"].iloc[0]), "Runs": len(group)}
        paper = {"Weight setting": setting, "Runs": len(group)}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            mean_value, std_value = float(values.mean()), float(values.std(ddof=1))
            row[f"{column} mean"] = mean_value
            row[f"{column} std"] = std_value
            digits = 2 if ("throughput" in column.lower() or "rate" in column.lower()) else 4
            if "Convergence" in column or "Runtime" in column:
                digits = 1
            paper[column] = fmt_mean_std(mean_value, std_value, digits)
        summary_rows.append(row); paper_rows.append(paper)
    return pd.DataFrame(summary_rows), pd.DataFrame(paper_rows)


def save_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    energy = summary_df[summary_df["Setting"].isin(["energy_0.5x", "default", "energy_1.5x"])].copy()
    energy["x"] = energy["Setting"].map({"energy_0.5x": 0.5, "default": 1.0, "energy_1.5x": 1.5})
    energy = energy.sort_values("x")
    plt.figure(figsize=(7, 5))
    plt.errorbar(energy["x"], energy["Average energy consumption mean"],
                 yerr=energy["Average energy consumption std"].fillna(0.0),
                 marker="o", capsize=3)
    plt.xlabel("Energy-weight multiplier"); plt.ylabel("Average energy consumption")
    plt.tight_layout(); plt.savefig(output_dir / "weight_sensitivity.png", dpi=300, bbox_inches="tight"); plt.close()

    fairness_order = {"fairness_0.0x": 0.0, "fairness_0.5x": 0.5,
                      "default": 1.0, "fairness_1.5x": 1.5, "fairness_2.0x": 2.0}
    tradeoff = summary_df[summary_df["Setting"].isin(fairness_order)].copy()
    tradeoff["Fairness-weight multiplier"] = tradeoff["Setting"].map(fairness_order)
    tradeoff = tradeoff.sort_values("Fairness-weight multiplier")
    plt.figure(figsize=(7, 5))
    plt.plot(tradeoff["Average throughput (kbps) mean"], tradeoff["Jain fairness index mean"], marker="o")
    for _, row in tradeoff.iterrows():
        plt.annotate(f"{row['Fairness-weight multiplier']:.1f}×",
                     (row["Average throughput (kbps) mean"], row["Jain fairness index mean"]),
                     xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Average throughput (kbps)"); plt.ylabel("Jain fairness index")
    plt.tight_layout(); plt.savefig(output_dir / "throughput_fairness_tradeoff.png", dpi=300, bbox_inches="tight"); plt.close()
    tradeoff[["Setting", "Fairness-weight multiplier",
              "Average throughput (kbps) mean", "Average throughput (kbps) std",
              "Jain fairness index mean", "Jain fairness index std",
              "Average energy consumption mean", "Average energy consumption std"]].to_csv(
        output_dir / "throughput_fairness_tradeoff.csv", index=False)


def parse_seed_list(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--seeds cannot be empty.")
    return tuple(int(item) for item in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="fast")
    parser.add_argument("--output-dir", default="data/eam_weight_sensitivity_fast")
    parser.add_argument("--seeds", default=None, help="Comma-separated override, e.g. 0,1,2")
    parser.add_argument("--base-throughput-weight", type=float, default=BASE_THROUGHPUT_WEIGHT)
    parser.add_argument("--base-energy-weight", type=float, default=BASE_ENERGY_WEIGHT)
    parser.add_argument("--base-fairness-weight", type=float, default=BASE_FAIRNESS_WEIGHT)
    parser.add_argument("--base-violation-weight", type=float, default=BASE_VIOLATION_WEIGHT)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    seed_override = parse_seed_list(args.seeds)
    if seed_override is not None:
        profile = Profile(**{**profile.__dict__, "seeds": seed_override})

    output_dir = Path(args.output_dir); run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = build_weight_settings(args.base_throughput_weight, args.base_energy_weight,
                                     args.base_fairness_weight, args.base_violation_weight)

    train_trajectories = FixedTrajectoryBank(
        path=str(output_dir / f"fixed_train_env_seed0_ep{profile.train_episodes}_steps{profile.train_steps}_K10.npz"),
        episodes=profile.train_episodes, steps=profile.train_steps, num_users=10, seed=0)
    eval_trajectories = FixedTrajectoryBank(
        path=str(output_dir / f"fixed_eval_env_seed999_ep{profile.eval_episodes}_steps{profile.eval_steps}_K10.npz"),
        episodes=profile.eval_episodes, steps=profile.eval_steps, num_users=10, seed=999)

    print("=" * 72)
    print("Fast EAM weight-sensitivity experiment")
    print("Profile:", args.profile, "Seeds:", profile.seeds)
    print("Training budget per run:", f"{profile.train_episodes} episodes × {profile.train_steps} steps")
    print("Configurations:", len(settings), "Total runs:", len(settings) * len(profile.seeds))
    print("=" * 72)

    run_rows = []
    for setting in settings:
        for seed in profile.seeds:
            agent, train_result = train_one(setting, seed, profile, train_trajectories)
            eval_result = evaluate_one(agent, setting, eval_trajectories, profile)
            np.savez_compressed(run_dir / f"{setting.key}_seed{seed}.npz", **train_result,
                                eval_metrics=np.asarray(list(eval_result.values()), dtype=np.float32),
                                eval_metric_names=np.asarray(list(eval_result.keys())))
            row = {
                "Setting": setting.key, "Group": setting.group, "Multiplier": setting.multiplier,
                "Seed": seed, "Throughput weight": setting.throughput_weight,
                "Energy weight": setting.energy_weight, "Fairness weight": setting.fairness_weight,
                "Violation weight": setting.violation_weight, **eval_result,
                "Convergence episode": float(np.asarray(train_result["convergence_episode"]).item()),
                "Runtime (s)": float(np.asarray(train_result["runtime_seconds"]).item()),
            }
            run_rows.append(row)
            pd.DataFrame(run_rows).to_csv(output_dir / "run_level_results.csv", index=False)
            print(f"Completed {setting.key}, seed={seed}: throughput={eval_result['Average throughput (kbps)']:.3f}, "
                  f"Jain={eval_result['Jain fairness index']:.4f}, energy={eval_result['Average energy consumption']:.5f}")

    run_df = pd.DataFrame(run_rows)
    summary_df, paper_df = summarize(run_df)
    summary_df.to_csv(output_dir / "table6_summary.csv", index=False)
    paper_df.to_csv(output_dir / "table6_paper.csv", index=False)
    save_plots(summary_df, output_dir)
    print("\nFinished.")
    print("Paper-ready table:", output_dir / "table6_paper.csv")


if __name__ == "__main__":
    main()
