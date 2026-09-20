#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Five-seed SHAP analysis for the PyTorch UDDPG-Net described in the manuscript.

Defaults:
- K=10
- state_dim=4+2K=24
- action_dim=10
- Actor: 24-128-128-128-10, ReLU + tanh
- Critic: (24+10)-128-128-128-1, ReLU + linear

Primary target:
    f_Q(s) = Q(s, mu(s))
This explains which state variables most strongly affect the critic-estimated
long-term utility under the deterministic policy.

Optional target:
    Sum of normalized actor outputs at known power-control indices.
Only enable this after confirming the true power indices in the action vector.
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim=24, action_dim=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim=24, action_dim=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class PolicyValueModel(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        return self.critic(state, self.actor(state))


class ActorPowerModel(nn.Module):
    def __init__(self, actor, power_indices):
        super().__init__()
        self.actor = actor
        self.register_buffer(
            "power_indices",
            torch.as_tensor(power_indices, dtype=torch.long),
            persistent=False,
        )

    def forward(self, state):
        action = self.actor(state)
        power_raw = action.index_select(1, self.power_indices)
        return ((power_raw + 1.0) / 2.0).sum(dim=1, keepdim=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and obj and all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    )


def find_state_dict(checkpoint, keys):
    if is_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in keys:
            value = checkpoint.get(key)
            if is_state_dict(value):
                return value
    raise KeyError(f"Cannot find state_dict. Tried keys: {keys}")


def strip_prefixes(state_dict):
    output = dict(state_dict)
    prefixes = ("module.", "actor.", "critic.", "policy.",
                "online_actor.", "online_critic.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if output and all(k.startswith(prefix) for k in output):
                output = {k[len(prefix):]: v for k, v in output.items()}
                changed = True
                break
    return output


def load_models(path, state_dim, action_dim, hidden_dim, device):
    checkpoint = torch.load(path, map_location="cpu")
    actor_sd = find_state_dict(
        checkpoint,
        ["actor_state_dict", "actor", "online_actor",
         "policy_state_dict", "policy", "actor_net"],
    )
    critic_sd = find_state_dict(
        checkpoint,
        ["critic_state_dict", "critic", "online_critic",
         "q_state_dict", "q_network", "critic_net"],
    )

    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, action_dim, hidden_dim)
    actor.load_state_dict(strip_prefixes(actor_sd), strict=True)
    critic.load_state_dict(strip_prefixes(critic_sd), strict=True)
    actor.to(device).eval()
    critic.to(device).eval()
    return actor, critic


def load_states(path, state_key, state_dim):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        x = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if state_key not in data:
            raise KeyError(f"{path}: missing key '{state_key}', keys={list(data.keys())}")
        x = data[state_key]
    else:
        raise ValueError("State files must be .npy or .npz")

    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.ndim != 2 or x.shape[1] != state_dim:
        raise ValueError(f"{path}: expected (*,{state_dim}), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: NaN or Inf detected")
    return x


def choose_rows(x, n, rng):
    idx = rng.choice(len(x), size=n, replace=len(x) < n)
    return x[idx]


def normalize_shap(values, n, d):
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected a scalar model output")
        values = values[0]
    values = np.asarray(values).squeeze()
    if n == 1 and values.shape == (d,):
        values = values[None, :]
    if values.shape != (n, d):
        raise ValueError(f"Unexpected SHAP shape {values.shape}, expected {(n,d)}")
    return values


def run_deep_shap(model, background_np, explain_np, device):
    background = torch.as_tensor(background_np, dtype=torch.float32, device=device)
    explain = torch.as_tensor(explain_np, dtype=torch.float32, device=device)
    # Do not use torch.no_grad(): SHAP needs gradients with respect to inputs.
    explainer = shap.DeepExplainer(model, background)
    values = explainer.shap_values(explain, check_additivity=False)
    return normalize_shap(values, len(explain_np), explain_np.shape[1])


def load_names(path, state_dim):
    if path is None:
        return [f"state_{i:02d}" for i in range(state_dim)]
    path = Path(path)
    if path.suffix.lower() == ".json":
        names = json.loads(path.read_text(encoding="utf-8"))
    else:
        names = [v.strip() for v in path.read_text(encoding="utf-8").splitlines() if v.strip()]
    if len(names) != state_dim:
        raise ValueError(f"Need exactly {state_dim} feature names")
    return list(map(str, names))


def load_groups(path, state_dim):
    if path is None:
        return {}
    groups = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, indices in groups.items():
        if not indices or min(indices) < 0 or max(indices) >= state_dim:
            raise ValueError(f"Invalid indices in group: {name}")
    return groups


def save_importance(seed_importance, names, path):
    mean = seed_importance.mean(0)
    sd = seed_importance.std(0, ddof=1)
    df = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": mean,
        "sd_across_seeds": sd,
    }).sort_values("mean_abs_shap", ascending=False)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def save_group_importance(seed_importance, groups, path):
    if not groups:
        return None
    rows = []
    for name, indices in groups.items():
        per_seed = seed_importance[:, indices].sum(1)
        rows.append({
            "group": name,
            "mean_abs_shap": float(per_seed.mean()),
            "sd_across_seeds": float(per_seed.std(ddof=1)),
        })
    df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def bar_plot(df, label, xlabel, path, top_n):
    data = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4.5, 0.38 * len(data) + 1.5)))
    plt.barh(data[label], data["mean_abs_shap"], xerr=data["sd_across_seeds"])
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def beeswarm_plot(values, states, names, path, max_display):
    explanation = shap.Explanation(values=values, data=states, feature_names=names)
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
        plot_size=(9, max(5.5, 0.38 * max_display + 1.5)),
    )
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def parse_indices(text):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--state-key", default="states")
    parser.add_argument("--state-dim", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--explain-size", type=int, default=300)
    parser.add_argument("--feature-names", default=None)
    parser.add_argument("--groups", default=None)
    parser.add_argument("--power-indices", default="")
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument("--output-dir", default="shap_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if len(args.checkpoints) != 5:
        raise ValueError("Provide exactly five checkpoints")
    if len(args.states) not in (1, 5):
        raise ValueError("Provide one common state file or five seed-specific files")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    names = load_names(args.feature_names, args.state_dim)
    groups = load_groups(args.groups, args.state_dim)
    power_indices = parse_indices(args.power_indices)

    seed_q_importance = []
    all_q_values = []
    all_explain_states = []
    seed_power_importance = []
    all_power_values = []

    for i, ckpt in enumerate(args.checkpoints):
        state_file = args.states[0] if len(args.states) == 1 else args.states[i]
        states = load_states(state_file, args.state_key, args.state_dim)
        background = choose_rows(states, args.background_size, rng)
        explain_states = choose_rows(states, args.explain_size, rng)

        actor, critic = load_models(
            ckpt, args.state_dim, args.action_dim, args.hidden_dim, device
        )

        q_model = PolicyValueModel(actor, critic).to(device).eval()
        q_values = run_deep_shap(q_model, background, explain_states, device)
        q_importance = np.abs(q_values).mean(0)

        seed_q_importance.append(q_importance)
        all_q_values.append(q_values)
        all_explain_states.append(explain_states)

        power_values = np.empty((0,))
        power_importance = np.empty((0,))
        if power_indices:
            if min(power_indices) < 0 or max(power_indices) >= args.action_dim:
                raise ValueError("Power index outside action dimension")
            power_model = ActorPowerModel(actor, power_indices).to(device).eval()
            power_values = run_deep_shap(power_model, background, explain_states, device)
            power_importance = np.abs(power_values).mean(0)
            seed_power_importance.append(power_importance)
            all_power_values.append(power_values)

        np.savez_compressed(
            out / f"seed_{i}_shap.npz",
            states=explain_states,
            q_shap=q_values,
            q_feature_importance=q_importance,
            power_shap=power_values,
            power_feature_importance=power_importance,
        )
        print(f"[seed {i}] explained {len(explain_states)} states")

    seed_q_importance = np.stack(seed_q_importance)
    q_df = save_importance(seed_q_importance, names, out / "q_feature_importance.csv")
    bar_plot(
        q_df, "feature",
        "Mean |SHAP value| for critic-estimated long-term utility",
        out / "q_global_importance_bar.png",
        args.top_features,
    )

    q_group_df = save_group_importance(
        seed_q_importance, groups, out / "q_group_importance.csv"
    )
    if q_group_df is not None:
        bar_plot(
            q_group_df, "group",
            "Grouped mean |SHAP value| for critic-estimated long-term utility",
            out / "q_group_importance_bar.png",
            len(q_group_df),
        )

    all_q_values = np.concatenate(all_q_values)
    all_explain_states = np.concatenate(all_explain_states)
    beeswarm_plot(
        all_q_values, all_explain_states, names,
        out / "q_beeswarm.png", args.top_features
    )

    if power_indices:
        seed_power_importance = np.stack(seed_power_importance)
        p_df = save_importance(
            seed_power_importance, names, out / "power_feature_importance.csv"
        )
        bar_plot(
            p_df, "feature",
            "Mean |SHAP value| for normalized power tendency",
            out / "power_global_importance_bar.png",
            args.top_features,
        )
        p_group_df = save_group_importance(
            seed_power_importance, groups, out / "power_group_importance.csv"
        )
        if p_group_df is not None:
            bar_plot(
                p_group_df, "group",
                "Grouped mean |SHAP value| for normalized power tendency",
                out / "power_group_importance_bar.png",
                len(p_group_df),
            )
        beeswarm_plot(
            np.concatenate(all_power_values), all_explain_states, names,
            out / "power_beeswarm.png", args.top_features
        )

    metadata = {
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "hidden_dim": args.hidden_dim,
        "background_size_per_seed": args.background_size,
        "explain_size_per_seed": args.explain_size,
        "number_of_seeds": 5,
        "power_indices": power_indices,
        "device": str(device),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
