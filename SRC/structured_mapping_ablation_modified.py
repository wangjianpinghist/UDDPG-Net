#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structured_mapping_ablation.py

Controlled ablation for:
    Fig. 8. Ablation of the structured hybrid-action mapping in UDDPG-Net.

This version fixes result bookkeeping, safe output filenames, and
Table-6 consistency reporting. It does NOT force experimental values
to match the manuscript reference table.

Current manuscript setting:
    K = 10 active links
    B = 5 acoustic subbands
    raw action dimension = K(2B+1) = 110

Raw action:
    [power-control scores (K*B),
     subband-preference scores (K*B),
     time-sharing scores (K)]

Full UDDPG-Net:
    tanh power normalization
    -> softmax subband preference
    -> discrete subband selection
    -> selected power/subband coupling
    -> energy-feasibility projection

w/o Structured Mapping:
    direct power/subband post-processing
    -> SAME energy-feasibility projection

The two variants use the same:
    actor/critic architecture
    optimizer
    reward
    fixed trajectories
    exploration
    training budget
    evaluation protocol
    random seed set

IMPORTANT:
This is a controlled mapping-ablation runner. Its compact environment
reproduces the manuscript's action structure and the main
throughput/interference/energy/fairness coupling, but it is not a replacement
for the final physical-layer environment. Numerical values should be used in
the manuscript only after checking that the physical-channel implementation,
units, constants, and training budget match the main UDDPG-Net experiment.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Manuscript setting
    K: int = 10
    B: int = 5

    # Fast controlled ablation budget. For the final manuscript, use the
    # same training budget as the main UDDPG-Net experiment if required.
    train_episodes: int = 300
    steps_per_episode: int = 100

    eval_episodes: int = 20
    eval_steps: int = 100

    # Same 3x128 FC architecture described in the manuscript.
    hidden: int = 128
    actor_lr: float = 0.002
    critic_lr: float = 0.004
    gamma: float = 0.96
    tau: float = 0.01

    replay_capacity: int = 50000
    batch_size: int = 128
    warmup_steps: int = 1000
    update_every: int = 1

    # Compact normalized resource/energy model.
    P_MAX: float = 1.0
    CIRCUIT_ENERGY: float = 0.010
    TX_ENERGY_SCALE: float = 0.025
    INITIAL_ENERGY: float = 1.0

    # Normalized channel/interference settings.
    SUBBAND_BW_KHZ: float = 2.0
    NOISE: float = 0.08
    INTERFERENCE_COEF: float = 0.25

    # Composite Gaussian + OU exploration.
    noise_start: float = 0.18
    noise_end: float = 0.04
    ou_theta: float = 0.15

    # EAM coefficients.
    throughput_weight: float = 1.0
    energy_weight: float = 0.10
    fairness_weight: float = 1.0
    violation_weight: float = 1.0

    # ------------------------------------------------------------
    # Manuscript Table-6 reference values.
    #
    # IMPORTANT:
    # These are validation targets, NOT values used to modify the
    # simulation output. The experiment must reproduce them from
    # the same physical-layer implementation used by the manuscript.
    # ------------------------------------------------------------
    table6_ref_uddpg_return: float = 426.17
    table6_ref_uddpg_return_std: float = 0.83
    table6_ref_uddpg_throughput: float = 105.75
    table6_ref_uddpg_throughput_std: float = 1.81
    table6_ref_uddpg_jain: float = 0.740
    table6_ref_uddpg_jain_std: float = 0.022

    table6_ref_abl_throughput: float = 101.86
    table6_ref_abl_throughput_std: float = 2.04
    table6_ref_abl_jain: float = 0.711
    table6_ref_abl_jain_std: float = 0.027

    # Numerical tolerance used only for a consistency report.
    table6_value_tolerance: float = 0.05

    @property
    def action_dim(self) -> int:
        # D_a = K(2B+1), exactly as in the manuscript.
        return self.K * (2 * self.B + 1)

    @property
    def state_dim(self) -> int:
        # 4K + 2 compact augmented state:
        # direct channel, interference, residual energy,
        # previous subband index, plus two global variables.
        return 4 * self.K + 2


# ============================================================
# Fixed trajectories
# ============================================================

class FixedTrajectories:
    """Algorithm-independent trajectories shared by both variants."""

    def __init__(self, cfg: Config, seed: int, episodes: int, steps: int):
        self.K = cfg.K
        self.episodes = episodes
        self.steps = steps
        rng = np.random.RandomState(seed)

        self.base_pos = rng.uniform(
            0.15, 1.0, size=(episodes, cfg.K, 2)
        ).astype(np.float32)

        self.channel_noise = (
            rng.normal(0.0, 0.025, size=(episodes, steps, cfg.K))
        ).astype(np.float32)

        self.interference_noise = (
            rng.normal(0.0, 0.025, size=(episodes, steps, cfg.K))
        ).astype(np.float32)

        self.load_noise = (
            rng.normal(0.0, 0.02, size=(episodes, steps))
        ).astype(np.float32)

    def initial(self, ep: int) -> np.ndarray:
        return self.base_pos[ep].copy()

    def at(self, ep: int, step: int):
        return (
            self.channel_noise[ep, step],
            self.interference_noise[ep, step],
            self.load_noise[ep, step],
        )


# ============================================================
# Compact underwater resource-allocation environment
# ============================================================

class FastUnderwaterEnv:
    """
    Compact environment used only for the controlled mapping ablation.

    The action interface follows the manuscript exactly:
        power scores: K*B
        subband scores: K*B
        time sharing: K
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.K = cfg.K
        self.B = cfg.B

        self.energy = np.full(
            self.K, cfg.INITIAL_ENERGY, dtype=np.float32
        )
        self.prev_band = np.zeros(
            self.K, dtype=np.int32
        )

    def reset(self):
        self.energy.fill(self.cfg.INITIAL_ENERGY)
        self.prev_band.fill(0)

    # --------------------------------------------------------
    # Full structured mapping
    # --------------------------------------------------------

    def structured_mapping(self, raw):
        """
        Structured hybrid-action mapping used by the complete UDDPG-Net.

        The softmax preference is not only used to choose the subband; it also
        participates in the power/subband coupling. The factor B preserves
        the power scale when a near-uniform preference is produced.
        """
        K, B = self.K, self.B
        x = np.asarray(raw, dtype=np.float32)

        power_raw = x[:K * B].reshape(K, B)
        band_raw = x[K * B:2 * K * B].reshape(K, B)
        alpha_raw = x[2 * K * B:2 * K * B + K]

        rho = (np.tanh(power_raw) + 1.0) / 2.0
        rho = np.clip(rho, 0.0, 1.0)
        candidate_power = rho * self.cfg.P_MAX

        z = band_raw - np.max(
            band_raw, axis=1, keepdims=True
        )
        pref = np.exp(z)
        pref /= np.sum(pref, axis=1, keepdims=True) + 1e-8

        # Hard execution decision.
        band = np.argmax(pref, axis=1).astype(np.int32)

        # Explicit power/subband coupling:
        # the selected power is modulated by the normalized subband
        # preference while retaining the nominal P_MAX scale.
        coupled_power = np.clip(
            candidate_power * B * pref,
            0.0,
            self.cfg.P_MAX,
        )
        selected_power = coupled_power[
            np.arange(K), band
        ].astype(np.float32)

        alpha = np.clip(
            (np.tanh(alpha_raw) + 1.0) / 2.0,
            0.0,
            1.0,
        ).astype(np.float32)

        selected_power = alpha * selected_power

        return selected_power, band, alpha, pref

    # --------------------------------------------------------
    # Ablated direct post-processing
    # --------------------------------------------------------

    def direct_mapping(self, raw):
        """
        Ablated mapping without the structured hybrid representation.

        Power and subband decisions are processed independently. The
        subband is selected from the raw preference scores and the
        corresponding power is used directly, with no normalized
        preference-to-power coupling.
        """
        K, B = self.K, self.B
        x = np.asarray(raw, dtype=np.float32)

        power_raw = x[:K * B].reshape(K, B)
        band_raw = x[K * B:2 * K * B].reshape(K, B)
        alpha_raw = x[2 * K * B:2 * K * B + K]

        rho = (np.tanh(power_raw) + 1.0) / 2.0
        rho = np.clip(rho, 0.0, 1.0)
        candidate_power = rho * self.cfg.P_MAX

        band = np.argmax(
            band_raw, axis=1
        ).astype(np.int32)

        selected_power = candidate_power[
            np.arange(K), band
        ].astype(np.float32)

        alpha = np.clip(
            (np.tanh(alpha_raw) + 1.0) / 2.0,
            0.0,
            1.0,
        ).astype(np.float32)

        selected_power = alpha * selected_power

        return selected_power, band, alpha, None

    # --------------------------------------------------------
    # SAME energy-feasibility projection
    # --------------------------------------------------------

    def energy_projection(self, power):
        cfg = self.cfg

        candidate_energy = (
            cfg.CIRCUIT_ENERGY
            + cfg.TX_ENERGY_SCALE * np.square(power)
        )

        available = np.maximum(
            self.energy - cfg.CIRCUIT_ENERGY,
            0.0,
        )

        max_power = np.sqrt(
            available / max(
                cfg.TX_ENERGY_SCALE, 1e-8
            )
        )

        scale = np.minimum(
            1.0,
            max_power / np.maximum(
                power, 1e-8
            ),
        )

        scale = np.where(
            power > 1e-8,
            scale,
            0.0,
        )

        executed = np.clip(
            power * scale,
            0.0,
            cfg.P_MAX,
        )

        actual_energy = (
            cfg.CIRCUIT_ENERGY
            + cfg.TX_ENERGY_SCALE
            * executed ** 2
        )

        violation = (
            candidate_energy
            > self.energy + 1e-10
        )

        self.energy = np.maximum(
            self.energy - actual_energy,
            0.0,
        ).astype(np.float32)

        return (
            executed.astype(np.float32),
            violation.astype(np.float32),
        )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    def state(
        self,
        gains,
        interference,
        load,
    ):
        e = (
            self.energy
            / self.cfg.INITIAL_ENERGY
        )
        occ = (
            self.prev_band
            / max(self.B - 1, 1)
        )

        s = np.concatenate([
            gains,
            interference,
            e,
            occ,
            np.array([
                float(np.mean(e)),
                float(load),
            ], dtype=np.float32),
        ])

        return np.nan_to_num(
            s,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).astype(np.float32)

    # --------------------------------------------------------
    # Environment transition
    # --------------------------------------------------------

    def step(
        self,
        raw_action,
        gains,
        env_interference,
        load,
        structured,
    ):
        if structured:
            power, band, alpha, _ = (
                self.structured_mapping(
                    raw_action
                )
            )
        else:
            power, band, alpha, _ = (
                self.direct_mapping(
                    raw_action
                )
            )

        # Both variants use exactly the same
        # energy-feasibility projection.
        power_exec, violation = (
            self.energy_projection(power)
        )

        rates = np.zeros(
            self.K,
            dtype=np.float32,
        )

        for i in range(self.K):
            same_band = (
                band == band[i]
            )
            same_band[i] = False

            other_ids = np.arange(
                self.K
            )[same_band]

            if len(other_ids) > 0:
                distance_term = (
                    1.0
                    + np.abs(
                        other_ids - i
                    )
                )

                interference = np.sum(
                    self.cfg.INTERFERENCE_COEF
                    * power_exec[same_band]
                    / distance_term
                )
            else:
                interference = 0.0

            channel = max(
                0.02,
                float(gains[i]),
            )

            sinr = (
                power_exec[i]
                * channel
                / (
                    self.cfg.NOISE
                    + interference
                    + 0.03 * max(
                        load, 0.0
                    )
                    + 1e-8
                )
            )

            rates[i] = (
                alpha[i]
                * self.cfg.SUBBAND_BW_KHZ
                * np.log2(
                    1.0 + sinr
                )
            )

        throughput = float(
            np.sum(rates)
        )

        fairness = float(
            np.sum(rates) ** 2
            / (
                self.K
                * np.sum(rates ** 2)
                + 1e-8
            )
        )

        mean_energy = float(
            np.mean(
                self.cfg.CIRCUIT_ENERGY
                + self.cfg.TX_ENERGY_SCALE
                * power_exec ** 2
            )
        )

        violation_rate = float(
            np.mean(violation)
        )

        throughput_norm = float(
            np.tanh(
                throughput
                / (
                    self.K
                    * self.cfg.SUBBAND_BW_KHZ
                )
            )
        )

        energy_norm = float(
            np.mean(
                power_exec ** 2
            )
        )

        reward = (
            self.cfg.throughput_weight
            * throughput_norm
            + self.cfg.fairness_weight
            * fairness
            - self.cfg.energy_weight
            * energy_norm
            - self.cfg.violation_weight
            * violation_rate
        )

        self.prev_band = band.copy()

        return {
            "reward": reward,
            "throughput": throughput,
            "fairness": fairness,
            "energy": mean_energy,
            "violation_rate": violation_rate,
            "remaining_energy": float(
                np.mean(self.energy)
            ),
        }


# ============================================================
# Actor / Critic
# ============================================================

class Actor(tf.keras.Model):
    def __init__(
        self,
        hidden,
        action_dim,
    ):
        super().__init__()

        self.fc1 = tf.keras.layers.Dense(
            hidden,
            activation="relu",
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden,
            activation="relu",
        )
        self.fc3 = tf.keras.layers.Dense(
            hidden,
            activation="relu",
        )
        self.out = tf.keras.layers.Dense(
            action_dim,
            activation="tanh",
        )

    def call(
        self,
        x,
        training=False,
    ):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)


class Critic(tf.keras.Model):
    def __init__(
        self,
        hidden,
    ):
        super().__init__()

        self.fc1 = tf.keras.layers.Dense(
            hidden,
            activation="relu",
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden,
            activation="relu",
        )
        self.fc3 = tf.keras.layers.Dense(
            hidden,
            activation="relu",
        )
        self.out = tf.keras.layers.Dense(
            1,
            activation=None,
        )

    def call(
        self,
        inputs,
        training=False,
    ):
        state, action = inputs
        x = tf.concat(
            [state, action],
            axis=-1,
        )
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.out(x)


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:
    def __init__(
        self,
        capacity,
        state_dim,
        action_dim,
        seed,
    ):
        self.capacity = capacity

        self.state = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.action = np.zeros(
            (capacity, action_dim),
            dtype=np.float32,
        )
        self.reward = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )
        self.next_state = np.zeros_like(
            self.state
        )
        self.done = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )

        self.ptr = 0
        self.size = 0
        self.rng = np.random.RandomState(
            seed
        )

    def add(
        self,
        s,
        a,
        r,
        ns,
        d,
    ):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[
            self.ptr, 0
        ] = r
        self.next_state[
            self.ptr
        ] = ns
        self.done[
            self.ptr, 0
        ] = float(d)

        self.ptr = (
            self.ptr + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch,
    ):
        idx = self.rng.choice(
            self.size,
            batch,
            replace=False,
        )

        return (
            tf.convert_to_tensor(
                self.state[idx]
            ),
            tf.convert_to_tensor(
                self.action[idx]
            ),
            tf.convert_to_tensor(
                self.reward[idx]
            ),
            tf.convert_to_tensor(
                self.next_state[idx]
            ),
            tf.convert_to_tensor(
                self.done[idx]
            ),
        )


# ============================================================
# Differentiable action representations for DDPG
# ============================================================

def structured_action_tensor(raw, K, B):
    """
    Differentiable surrogate used by the actor/critic update.

    A straight-through one-hot representation preserves the executed
    discrete subband decision in the forward pass while retaining the
    softmax derivative in the backward pass. The power branch is explicitly
    coupled to the subband preference.
    """
    power_raw = raw[:, :K * B]
    band_raw = raw[:, K * B:2 * K * B]
    alpha_raw = raw[:, 2 * K * B:2 * K * B + K]

    power_raw = tf.reshape(power_raw, (-1, K, B))
    band_raw = tf.reshape(band_raw, (-1, K, B))

    rho = (tf.tanh(power_raw) + 1.0) / 2.0
    pref = tf.nn.softmax(band_raw, axis=-1)

    hard = tf.one_hot(
        tf.argmax(pref, axis=-1),
        depth=B,
        dtype=raw.dtype,
    )
    st_pref = tf.stop_gradient(hard - pref) + pref

    coupled_power = tf.clip_by_value(
        rho * float(B) * st_pref,
        0.0,
        1.0,
    )

    alpha = (tf.tanh(alpha_raw) + 1.0) / 2.0

    return tf.concat(
        [
            tf.reshape(coupled_power, (-1, K * B)),
            tf.reshape(st_pref, (-1, K * B)),
            alpha,
        ],
        axis=-1,
    )


def direct_action_tensor(raw, K, B):
    """Direct continuous action representation used by the ablation."""
    return raw


# ============================================================
# DDPG
# ============================================================

class DDPG:
    def __init__(
        self,
        cfg,
        seed,
        structured,
    ):
        self.cfg = cfg
        self.structured = bool(structured)

        self.actor = Actor(
            cfg.hidden,
            cfg.action_dim,
        )
        self.target_actor = Actor(
            cfg.hidden,
            cfg.action_dim,
        )

        self.critic = Critic(
            cfg.hidden
        )
        self.target_critic = Critic(
            cfg.hidden
        )

        dummy_s = tf.zeros(
            (1, cfg.state_dim),
            dtype=tf.float32,
        )
        dummy_a = tf.zeros(
            (1, cfg.action_dim),
            dtype=tf.float32,
        )

        self.actor(dummy_s)
        self.target_actor(dummy_s)
        self.critic(
            (dummy_s, dummy_a)
        )
        self.target_critic(
            (dummy_s, dummy_a)
        )

        self.target_actor.set_weights(
            self.actor.get_weights()
        )
        self.target_critic.set_weights(
            self.critic.get_weights()
        )

        self.actor_opt = (
            tf.keras.optimizers.Adam(
                learning_rate=cfg.actor_lr
            )
        )

        self.critic_opt = (
            tf.keras.optimizers.Adam(
                learning_rate=cfg.critic_lr
            )
        )

        self.buffer = ReplayBuffer(
            cfg.replay_capacity,
            cfg.state_dim,
            cfg.action_dim,
            seed + 10000,
        )

    def action(
        self,
        state,
    ):
        s = tf.convert_to_tensor(
            state[None, :],
            dtype=tf.float32,
        )
        return self.actor(
            s,
            training=False,
        ).numpy()[0]

    def critic_action_tensor(self, raw):
        if self.structured:
            return structured_action_tensor(
                raw,
                self.cfg.K,
                self.cfg.B,
            )
        return direct_action_tensor(
            raw,
            self.cfg.K,
            self.cfg.B,
        )

    def critic_action_from_raw(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        tensor = tf.convert_to_tensor(
            raw[None, :],
            dtype=tf.float32,
        )
        return self.critic_action_tensor(tensor).numpy()[0]

    @tf.function
    def train_step(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ):
        with tf.GradientTape() as tape:
            next_raw_actions = self.target_actor(
                next_states,
                training=False,
            )
            next_actions = self.critic_action_tensor(
                next_raw_actions
            )

            target_q = (
                self.target_critic(
                    (
                        next_states,
                        next_actions,
                    ),
                    training=False,
                )
            )

            y = (
                rewards
                + self.cfg.gamma
                * (1.0 - dones)
                * target_q
            )

            q = self.critic(
                (
                    states,
                    actions,
                ),
                training=True,
            )

            critic_loss = (
                tf.reduce_mean(
                    tf.square(
                        q
                        - tf.stop_gradient(y)
                    )
                )
            )

        critic_grad = tape.gradient(
            critic_loss,
            self.critic.trainable_variables,
        )

        self.critic_opt.apply_gradients(
            zip(
                critic_grad,
                self.critic.trainable_variables,
            )
        )

        with tf.GradientTape() as tape:
            policy_raw_actions = self.actor(
                states,
                training=True,
            )
            policy_actions = self.critic_action_tensor(
                policy_raw_actions
            )

            q_pi = self.critic(
                (
                    states,
                    policy_actions,
                ),
                training=False,
            )

            actor_loss = (
                -tf.reduce_mean(q_pi)
            )

        actor_grad = tape.gradient(
            actor_loss,
            self.actor.trainable_variables,
        )

        self.actor_opt.apply_gradients(
            zip(
                actor_grad,
                self.actor.trainable_variables,
            )
        )

        for src, tgt in zip(
            self.actor.weights,
            self.target_actor.weights,
        ):
            tgt.assign(
                (1.0 - self.cfg.tau)
                * tgt
                + self.cfg.tau
                * src
            )

        for src, tgt in zip(
            self.critic.weights,
            self.target_critic.weights,
        ):
            tgt.assign(
                (1.0 - self.cfg.tau)
                * tgt
                + self.cfg.tau
                * src
            )

        return (
            actor_loss,
            critic_loss,
        )

    def update(self):
        if self.buffer.size < max(
            self.cfg.batch_size,
            self.cfg.warmup_steps,
        ):
            return

        return self.train_step(
            *self.buffer.sample(
                self.cfg.batch_size
            )
        )


# ============================================================
# Exploration
# ============================================================

class CompositeNoise:
    def __init__(
        self,
        dim,
        total_steps,
        seed,
        start=0.18,
        end=0.04,
        theta=0.15,
    ):
        self.dim = dim
        self.total = max(
            total_steps,
            1,
        )
        self.start = start
        self.end = end
        self.theta = theta

        self.rng = np.random.RandomState(
            seed
        )

        self.ou = np.zeros(
            dim,
            dtype=np.float32,
        )

    def reset(self):
        self.ou.fill(0.0)

    def sample(
        self,
        step,
    ):
        ratio = min(
            max(
                step / self.total,
                0.0,
            ),
            1.0,
        )

        std = (
            self.start
            + (
                self.end
                - self.start
            )
            * ratio
        )

        component_std = (
            std / math.sqrt(2.0)
        )

        gaussian = self.rng.normal(
            0.0,
            component_std,
            size=self.dim,
        ).astype(np.float32)

        self.ou += (
            -self.theta
            * self.ou
            + self.rng.normal(
                0.0,
                component_std,
                size=self.dim,
            ).astype(np.float32)
        )

        return (
            gaussian
            + self.ou
        )


# ============================================================
# State generation
# ============================================================

def make_state_variables(
    cfg,
    trajectories,
    ep,
    step,
):
    pos = trajectories.initial(
        ep
    )

    distance = np.sqrt(
        np.sum(
            pos ** 2,
            axis=1,
        )
    )

    gains = (
        1.0
        / (
            1.0
            + distance
        )
    ).astype(np.float32)

    channel_noise, int_noise, load_noise = (
        trajectories.at(
            ep,
            step,
        )
    )

    gains = np.clip(
        gains + channel_noise,
        0.03,
        1.2,
    )

    interference = np.clip(
        0.35
        + 0.10
        * np.roll(
            gains,
            1,
        )
        + int_noise,
        0.0,
        1.5,
    )

    load = float(
        np.clip(
            1.0 + load_noise,
            0.5,
            1.5,
        )
    )

    return (
        gains.astype(np.float32),
        interference.astype(np.float32),
        load,
    )


# ============================================================
# Training
# ============================================================

def train_one(
    cfg,
    seed,
    structured,
    trajectories,
):
    set_seed(seed)

    agent = DDPG(
        cfg,
        seed,
        structured,
    )
    env = FastUnderwaterEnv(
        cfg
    )

    noise = CompositeNoise(
        cfg.action_dim,
        cfg.train_episodes
        * cfg.steps_per_episode,
        seed + 30000,
        cfg.noise_start,
        cfg.noise_end,
        cfg.ou_theta,
    )

    warm_rng = np.random.RandomState(
        seed + 40000
    )

    ep_return = []
    ep_throughput = []
    ep_fairness = []
    ep_energy = []
    ep_violation = []

    total_step = 0
    start_time = time.time()

    for ep in range(
        cfg.train_episodes
    ):
        env.reset()
        noise.reset()

        gains, inter, load = (
            make_state_variables(
                cfg,
                trajectories,
                ep,
                0,
            )
        )

        state = env.state(
            gains,
            inter,
            load,
        )

        ret = []
        thr = []
        fair = []
        energy = []
        violation = []

        for step in range(
            cfg.steps_per_episode
        ):
            if (
                total_step
                < cfg.warmup_steps
            ):
                raw_action = (
                    warm_rng.uniform(
                        -1.0,
                        1.0,
                        size=cfg.action_dim,
                    ).astype(
                        np.float32
                    )
                )
            else:
                raw_action = np.clip(
                    agent.action(
                        state
                    )
                    + noise.sample(
                        total_step
                    ),
                    -1.0,
                    1.0,
                ).astype(
                    np.float32
                )

            gains, inter, load = (
                make_state_variables(
                    cfg,
                    trajectories,
                    ep,
                    step,
                )
            )

            metrics = env.step(
                raw_action,
                gains,
                inter,
                load,
                structured,
            )

            next_state = env.state(
                gains,
                inter,
                load,
            )

            done = (
                step
                == cfg.steps_per_episode - 1
            )

            # Store the same action representation used by the critic.
            # The environment still receives the raw actor output and applies
            # the selected mapping before feasibility projection.
            critic_action = agent.critic_action_from_raw(
                raw_action
            )

            agent.buffer.add(
                state,
                critic_action,
                metrics["reward"],
                next_state,
                done,
            )

            if (
                total_step
                >= cfg.warmup_steps
                and total_step
                % cfg.update_every
                == 0
            ):
                agent.update()

            ret.append(
                metrics["reward"]
            )
            thr.append(
                metrics["throughput"]
            )
            fair.append(
                metrics["fairness"]
            )
            energy.append(
                metrics["energy"]
            )
            violation.append(
                metrics["violation_rate"]
            )

            state = next_state
            total_step += 1

        ep_return.append(
            float(np.sum(ret))
        )
        ep_throughput.append(
            float(np.mean(thr))
        )
        ep_fairness.append(
            float(np.mean(fair))
        )
        ep_energy.append(
            float(np.mean(energy))
        )
        ep_violation.append(
            float(np.mean(violation))
        )

        if (
            ep == 0
            or (ep + 1)
            % max(
                cfg.train_episodes // 5,
                1,
            )
            == 0
        ):
            name = (
                "UDDPG-Net"
                if structured
                else "w/o Structured Mapping"
            )

            print(
                f"[{name:24s}] "
                f"seed={seed} "
                f"episode={ep+1:4d}/"
                f"{cfg.train_episodes} "
                f"return={ep_return[-1]:8.3f} "
                f"thr={ep_throughput[-1]:7.3f} "
                f"Jain={ep_fairness[-1]:.4f}"
            )

    return agent, {
        "episode_return": np.asarray(
            ep_return,
            dtype=np.float32,
        ),
        "episode_throughput": np.asarray(
            ep_throughput,
            dtype=np.float32,
        ),
        "episode_fairness": np.asarray(
            ep_fairness,
            dtype=np.float32,
        ),
        "episode_energy": np.asarray(
            ep_energy,
            dtype=np.float32,
        ),
        "episode_violation": np.asarray(
            ep_violation,
            dtype=np.float32,
        ),
        "runtime": float(
            time.time() - start_time
        ),
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_variant(
    cfg,
    agent,
    trajectories,
    structured,
):
    env = FastUnderwaterEnv(
        cfg
    )

    values = {
        "throughput": [],
        "fairness": [],
        "energy": [],
        "violation": [],
    }

    for ep in range(
        cfg.eval_episodes
    ):
        env.reset()

        for step in range(
            cfg.eval_steps
        ):
            gains, inter, load = (
                make_state_variables(
                    cfg,
                    trajectories,
                    ep,
                    step,
                )
            )

            state = env.state(
                gains,
                inter,
                load,
            )

            # Deterministic evaluation:
            # exploration noise is disabled.
            raw_action = agent.action(
                state
            )

            metrics = env.step(
                raw_action,
                gains,
                inter,
                load,
                structured,
            )

            values["throughput"].append(
                metrics["throughput"]
            )
            values["fairness"].append(
                metrics["fairness"]
            )
            values["energy"].append(
                metrics["energy"]
            )
            values["violation"].append(
                metrics["violation_rate"]
            )

    return {
        "throughput": float(
            np.mean(
                values["throughput"]
            )
        ),
        "fairness": float(
            np.mean(
                values["fairness"]
            )
        ),
        "energy": float(
            np.mean(
                values["energy"]
            )
        ),
        "violation": float(
            np.mean(
                values["violation"]
            )
        ),
    }


# ============================================================
# Plotting
# ============================================================

def mean_std_curves(
    records,
    key,
):
    arrays = [
        r[key]
        for r in records
    ]

    min_len = min(
        len(x)
        for x in arrays
    )

    x = np.stack(
        [
            x[:min_len]
            for x in arrays
        ],
        axis=0,
    )

    mean = np.mean(
        x,
        axis=0,
    )

    std = np.std(
        x,
        axis=0,
        ddof=1,
    )

    return mean, std


def save_fig8(
    full_records,
    ablated_records,
    out_path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.7),
        constrained_layout=True,
    )

    panels = [
        (
            "episode_return",
            "Episode",
            "Episode return",
            "a",
        ),
        (
            "episode_throughput",
            "Episode",
            "Throughput (kbps)",
            "b",
        ),
        (
            "episode_fairness",
            "Episode",
            "Jain fairness index",
            "c",
        ),
    ]

    for ax, (
        key,
        xlabel,
        ylabel,
        label,
    ) in zip(
        axes,
        panels,
    ):
        full_mean, full_std = (
            mean_std_curves(
                full_records,
                key,
            )
        )

        abl_mean, abl_std = (
            mean_std_curves(
                ablated_records,
                key,
            )
        )

        x = np.arange(
            1,
            len(full_mean) + 1,
        )

        # Matplotlib defaults are intentionally used;
        # no colors are hard-coded.
        ax.plot(
            x,
            full_mean,
            linewidth=1.8,
            label="UDDPG-Net",
        )

        ax.fill_between(
            x,
            full_mean - full_std,
            full_mean + full_std,
            alpha=0.16,
        )

        ax.plot(
            x,
            abl_mean,
            linewidth=1.8,
            linestyle="--",
            label="w/o Structured Mapping",
        )

        ax.fill_between(
            x,
            abl_mean - abl_std,
            abl_mean + abl_std,
            alpha=0.12,
        )

        ax.set_xlabel(
            xlabel,
            fontsize=10,
        )
        ax.set_ylabel(
            ylabel,
            fontsize=10,
        )

        ax.tick_params(
            labelsize=9,
        )

        ax.grid(
            alpha=0.22,
            linewidth=0.6,
        )

        ax.text(
            0.02,
            0.96,
            f"({label})",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )

    axes[0].legend(
        fontsize=8.5,
        frameon=True,
        loc="lower right",
    )

    fig.tight_layout(
        pad=0.8
    )

    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast controlled ablation of the "
            "structured hybrid-action mapping."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="data/structured_mapping_ablation",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Fast default: 300 episodes.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Fast default: 100 steps/episode.",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated seeds.",
    )

    parser.add_argument(
        "--single-seed",
        action="store_true",
        help="Smoke test with only seed 0.",
    )

    args = parser.parse_args()

    cfg = Config(
        train_episodes=args.episodes,
        steps_per_episode=args.steps,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
    )

    if args.single_seed:
        seeds = (0,)
    else:
        seeds = tuple(
            int(x.strip())
            for x in args.seeds.split(",")
            if x.strip()
        )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_traj = FixedTrajectories(
        cfg,
        seed=12345,
        episodes=cfg.train_episodes,
        steps=cfg.steps_per_episode,
    )

    eval_traj = FixedTrajectories(
        cfg,
        seed=54321,
        episodes=cfg.eval_episodes,
        steps=cfg.eval_steps,
    )

    print("=" * 76)
    print(
        "UDDPG-Net structured hybrid-action "
        "mapping ablation"
    )
    print("=" * 76)

    print(
        f"K={cfg.K}, B={cfg.B}"
    )

    print(
        "Raw action dimension: "
        f"K(2B+1) = {cfg.action_dim}"
    )

    print(
        f"Training: "
        f"{cfg.train_episodes} episodes × "
        f"{cfg.steps_per_episode} steps"
    )

    print(
        f"Seeds: {seeds}"
    )

    print(
        "Variants: UDDPG-Net vs. "
        "w/o Structured Mapping"
    )

    print("=" * 76)

    all_rows = []

    full_records = []
    ablated_records = []

    for structured, variant_name in [
        (True, "UDDPG-Net"),
        (False, "w/o Structured Mapping"),
    ]:

        for seed in seeds:

            agent, train = train_one(
                cfg,
                seed,
                structured,
                train_traj,
            )

            evaluation = evaluate_variant(
                cfg,
                agent,
                eval_traj,
                structured,
            )

            last_n = min(100, len(train["episode_return"]))
            last100_return = float(
                np.mean(train["episode_return"][-last_n:])
            )

            row = {
                "variant": variant_name,
                "seed": seed,
                "last100_return":
                    last100_return,
                "eval_throughput_kbps":
                    evaluation["throughput"],
                "eval_jain":
                    evaluation["fairness"],
                "eval_energy":
                    evaluation["energy"],
                "eval_violation_rate":
                    evaluation["violation"],
                "runtime_s":
                    train["runtime"],
            }

            all_rows.append(row)

            if structured:
                full_records.append(
                    train
                )
            else:
                ablated_records.append(
                    train
                )

            safe_variant_name = (
                variant_name
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            np.savez_compressed(
                out
                / (
                    safe_variant_name
                    + f"_seed{seed}.npz"
                ),
                episode_return=train[
                    "episode_return"
                ],
                episode_throughput=train[
                    "episode_throughput"
                ],
                episode_fairness=train[
                    "episode_fairness"
                ],
                episode_energy=train[
                    "episode_energy"
                ],
                episode_violation=train[
                    "episode_violation"
                ],
                eval_throughput=
                    evaluation["throughput"],
                eval_jain=
                    evaluation["fairness"],
                eval_energy=
                    evaluation["energy"],
                eval_violation=
                    evaluation["violation"],
            )

    # --------------------------------------------------------
    # Run-level CSV
    # --------------------------------------------------------

    csv_path = (
        out
        / "run_level_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                all_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            all_rows
        )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary = []

    for variant in [
        "UDDPG-Net",
        "w/o Structured Mapping",
    ]:
        rows = [
            r
            for r in all_rows
            if r["variant"] == variant
        ]

        def mean_std(key):
            values = np.asarray(
                [
                    r[key]
                    for r in rows
                ],
                dtype=float,
            )

            mean = float(
                np.mean(values)
            )

            std = float(
                np.std(
                    values,
                    ddof=1,
                )
            ) if len(values) > 1 else 0.0

            return mean, std

        r100_m, r100_s = mean_std(
            "last100_return"
        )
        thr_m, thr_s = mean_std(
            "eval_throughput_kbps"
        )
        j_m, j_s = mean_std(
            "eval_jain"
        )
        e_m, e_s = mean_std(
            "eval_energy"
        )
        v_m, v_s = mean_std(
            "eval_violation_rate"
        )
        r_m, r_s = mean_std(
            "runtime_s"
        )

        summary.append({
            "Variant": variant,
            "Last-100 return mean": r100_m,
            "Last-100 return std": r100_s,
            "Throughput mean": thr_m,
            "Throughput std": thr_s,
            "Jain mean": j_m,
            "Jain std": j_s,
            "Energy mean": e_m,
            "Energy std": e_s,
            "Violation mean": v_m,
            "Violation std": v_s,
            "Runtime mean (s)": r_m,
            "Runtime std (s)": r_s,
        })

    summary_path = (
        out
        / "table6_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            summary
        )

    # --------------------------------------------------------
    # Fig. 8
    # --------------------------------------------------------

    fig_path = (
        out
        / "Fig8_structured_mapping_ablation.png"
    )

    save_fig8(
        full_records,
        ablated_records,
        fig_path,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)

    for row in summary:
        print(
            f"{row['Variant']:24s} | "
            f"Last-100 return = "
            f"{row['Last-100 return mean']:.4f} "
            f"± {row['Last-100 return std']:.4f} | "
            f"Thr = "
            f"{row['Throughput mean']:.4f} "
            f"± {row['Throughput std']:.4f} | "
            f"Jain = "
            f"{row['Jain mean']:.4f} "
            f"± {row['Jain std']:.4f} | "
            f"Energy = "
            f"{row['Energy mean']:.6f}"
        )

    # ------------------------------------------------------------
    # Manuscript consistency report
    # ------------------------------------------------------------
    print("\n" + "-" * 76)
    print("TABLE-6 CONSISTENCY CHECK")
    print("-" * 76)
    print(
        "Reference values are reported for validation only; "
        "the code never overwrites experimental outputs."
    )

    ref = {
        "UDDPG-Net": {
            "last100": cfg.table6_ref_uddpg_return,
            "thr": cfg.table6_ref_uddpg_throughput,
            "jain": cfg.table6_ref_uddpg_jain,
        },
        "w/o Structured Mapping": {
            "last100": None,
            "thr": cfg.table6_ref_abl_throughput,
            "jain": cfg.table6_ref_abl_jain,
        },
    }

    for row in summary:
        name = row["Variant"]
        rr = ref[name]

        print(f"\n{name}")
        print(
            f"  Last-100 return : "
            f"{row['Last-100 return mean']:.4f}"
            + (
                f"  (Table-6 ref {rr['last100']:.2f})"
                if rr["last100"] is not None else
                "  (Table-6 reference not supplied)"
            )
        )
        print(
            f"  Throughput      : "
            f"{row['Throughput mean']:.4f} kbps"
            f"  (Table-6 ref {rr['thr']:.2f})"
        )
        print(
            f"  Jain fairness   : "
            f"{row['Jain mean']:.4f}"
            f"  (Table-6 ref {rr['jain']:.3f})"
        )

    print("\nNOTE:")
    print(
        "If the current compact environment does not reproduce the "
        "manuscript-scale Table-6 values, DO NOT rescale or overwrite "
        "the results. Replace the compact environment with the exact "
        "physical-layer environment used by the main UDDPG-Net experiment."
    )

    print("\nOutput files:")
    print(csv_path)
    print(summary_path)
    print(fig_path)
    print("=" * 76)


if __name__ == "__main__":
    main()
