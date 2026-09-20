"""
SAC baseline for the user's current simplified DDPG environment.

Compatibility:
- TensorFlow 2.16+ / Keras 3
- Does NOT use tf.compat.v1, tf.Session, tf.layers, or disable_v2_behavior
- Uses tf.keras.Model and tf.GradientTape

Important:
This script intentionally preserves the current DDPG environment:
1 scalar action, random user motion, channel_gain = 1 / (1 + distance),
and reward = channel_reward * Jain(channel gains) - 0.1 * action^2.
It is suitable for a preliminary same-environment SAC-vs-DDPG comparison,
not yet for the paper's full joint power-subband allocation task.
"""

import os
import time
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# Experiment configuration
# ============================================================
@dataclass(frozen=True)
class Config:
    pn: int = 1
    num_users: int = 10

    max_episodes: int = 2000
    max_episode_steps: int = 100

    gamma: float = 0.9
    tau: float = 0.01

    replay_capacity: int = 100_000
    batch_size: int = 64
    warmup_steps: int = 1_000

    actor_lr: float = 0.002
    critic_lr: float = 0.004
    alpha_lr: float = 0.0003

    hidden_dim: int = 128
    action_dim: int = 1
    action_bound: float = 1.0

    log_std_min: float = -20.0
    log_std_max: float = 2.0

    energy_weight: float = 0.1
    seed: int = 0

    # Set True only for a quick compatibility test.
    quick_test: bool = False


CFG = Config()
EPS = 1e-6


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================
# Replay buffer
# ============================================================
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity = int(capacity)
        self.pointer = 0
        self.size = 0

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )
        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.dones = np.zeros(
            (capacity, 1),
            dtype=np.float32,
        )

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.pointer % self.capacity

        self.states[index] = np.asarray(
            state,
            dtype=np.float32,
        ).reshape(-1)
        self.actions[index] = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)
        self.rewards[index, 0] = float(reward)
        self.next_states[index] = np.asarray(
            next_state,
            dtype=np.float32,
        ).reshape(-1)
        self.dones[index, 0] = float(done)

        self.pointer += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.size < batch_size:
            raise RuntimeError(
                f"Replay buffer has {self.size} samples, "
                f"but batch_size={batch_size}."
            )

        indices = np.random.choice(
            self.size,
            size=batch_size,
            replace=False,
        )

        return (
            tf.convert_to_tensor(self.states[indices]),
            tf.convert_to_tensor(self.actions[indices]),
            tf.convert_to_tensor(self.rewards[indices]),
            tf.convert_to_tensor(self.next_states[indices]),
            tf.convert_to_tensor(self.dones[indices]),
        )


# ============================================================
# Neural networks compatible with Keras 3
# ============================================================
class GaussianActor(tf.keras.Model):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float,
        log_std_max: float,
        name: str = "GaussianActor",
    ) -> None:
        super().__init__(name=name)

        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.fc1 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="fc1",
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="fc2",
        )
        self.fc3 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="fc3",
        )
        self.mean_layer = tf.keras.layers.Dense(
            action_dim,
            activation=None,
            name="mean",
        )
        self.log_std_layer = tf.keras.layers.Dense(
            action_dim,
            activation=None,
            name="log_std",
        )

    def call(
        self,
        states: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del training

        x = self.fc1(states)
        x = self.fc2(x)
        x = self.fc3(x)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(
            log_std,
            self.log_std_min,
            self.log_std_max,
        )
        return mean, log_std

    def sample(
        self,
        states: tf.Tensor,
        action_bound: float,
        deterministic: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        mean, log_std = self(states, training=True)
        std = tf.exp(log_std)

        if deterministic:
            pre_tanh = mean
        else:
            noise = tf.random.normal(tf.shape(mean))
            pre_tanh = mean + std * noise

        squashed = tf.tanh(pre_tanh)

        # Map tanh output from [-1, 1] to [0, action_bound].
        scale = tf.cast(0.5 * action_bound, tf.float32)
        actions = scale * (squashed + 1.0)

        # Gaussian log probability before the tanh transformation.
        gaussian_log_prob = -0.5 * (
            tf.square((pre_tanh - mean) / (std + EPS))
            + 2.0 * log_std
            + np.log(2.0 * np.pi)
        )
        gaussian_log_prob = tf.reduce_sum(
            gaussian_log_prob,
            axis=1,
            keepdims=True,
        )

        # Change-of-variables correction for:
        # action = 0.5 * bound * (tanh(u) + 1).
        correction = tf.reduce_sum(
            tf.math.log(
                scale * (1.0 - tf.square(squashed)) + EPS
            ),
            axis=1,
            keepdims=True,
        )
        log_prob = gaussian_log_prob - correction

        return actions, log_prob


class QNetwork(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int,
        name: str,
    ) -> None:
        super().__init__(name=name)

        self.fc1 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="fc1",
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="fc2",
        )
        self.fc3 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="fc3",
        )
        self.q_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            name="q",
        )

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        del training

        states, actions = inputs
        x = tf.concat([states, actions], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.q_layer(x)


# ============================================================
# Soft Actor-Critic
# ============================================================
class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: Config,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg

        self.actor = GaussianActor(
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )

        self.q1 = QNetwork(cfg.hidden_dim, name="Q1")
        self.q2 = QNetwork(cfg.hidden_dim, name="Q2")
        self.target_q1 = QNetwork(
            cfg.hidden_dim,
            name="TargetQ1",
        )
        self.target_q2 = QNetwork(
            cfg.hidden_dim,
            name="TargetQ2",
        )

        # Build every model before copying weights.
        dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, action_dim), dtype=tf.float32)

        self.actor(dummy_state)
        self.q1((dummy_state, dummy_action))
        self.q2((dummy_state, dummy_action))
        self.target_q1((dummy_state, dummy_action))
        self.target_q2((dummy_state, dummy_action))

        self.target_q1.set_weights(self.q1.get_weights())
        self.target_q2.set_weights(self.q2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.actor_lr,
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.critic_lr,
        )
        self.alpha_optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.alpha_lr,
        )

        self.log_alpha = tf.Variable(
            0.0,
            trainable=True,
            dtype=tf.float32,
            name="log_alpha",
        )
        self.target_entropy = tf.constant(
            -float(action_dim),
            dtype=tf.float32,
        )

        self.memory = ReplayBuffer(
            cfg.replay_capacity,
            state_dim,
            action_dim,
        )

    @property
    def alpha(self) -> tf.Tensor:
        return tf.exp(self.log_alpha)

    def choose_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        state_tensor = tf.convert_to_tensor(
            np.asarray(state, dtype=np.float32).reshape(1, -1)
        )
        action, _ = self.actor.sample(
            state_tensor,
            action_bound=self.cfg.action_bound,
            deterministic=deterministic,
        )
        return np.clip(
            action.numpy()[0],
            0.0,
            self.cfg.action_bound,
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.store(
            state,
            action,
            reward,
            next_state,
            done,
        )

    @tf.function(reduce_retracing=True)
    def _train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # ----------------------------------------------------
        # Update twin critics
        # ----------------------------------------------------
        with tf.GradientTape() as critic_tape:
            next_actions, next_log_prob = self.actor.sample(
                next_states,
                action_bound=self.cfg.action_bound,
                deterministic=False,
            )

            target_q1 = self.target_q1(
                (next_states, next_actions),
                training=False,
            )
            target_q2 = self.target_q2(
                (next_states, next_actions),
                training=False,
            )
            target_min_q = tf.minimum(target_q1, target_q2)

            target_value = (
                target_min_q
                - tf.stop_gradient(self.alpha) * next_log_prob
            )
            q_target = rewards + self.cfg.gamma * (
                1.0 - dones
            ) * target_value
            q_target = tf.stop_gradient(q_target)

            current_q1 = self.q1(
                (states, actions),
                training=True,
            )
            current_q2 = self.q2(
                (states, actions),
                training=True,
            )

            q1_loss = tf.reduce_mean(
                tf.square(current_q1 - q_target)
            )
            q2_loss = tf.reduce_mean(
                tf.square(current_q2 - q_target)
            )
            critic_loss = q1_loss + q2_loss

        critic_variables = (
            self.q1.trainable_variables
            + self.q2.trainable_variables
        )
        critic_gradients = critic_tape.gradient(
            critic_loss,
            critic_variables,
        )
        critic_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                critic_gradients,
                critic_variables,
            )
            if gradient is not None
        ]
        self.critic_optimizer.apply_gradients(critic_pairs)

        # ----------------------------------------------------
        # Update stochastic actor
        # ----------------------------------------------------
        with tf.GradientTape() as actor_tape:
            sampled_actions, log_prob = self.actor.sample(
                states,
                action_bound=self.cfg.action_bound,
                deterministic=False,
            )
            q1_pi = self.q1(
                (states, sampled_actions),
                training=False,
            )
            q2_pi = self.q2(
                (states, sampled_actions),
                training=False,
            )
            min_q_pi = tf.minimum(q1_pi, q2_pi)

            actor_loss = tf.reduce_mean(
                tf.stop_gradient(self.alpha) * log_prob
                - min_q_pi
            )

        actor_gradients = actor_tape.gradient(
            actor_loss,
            self.actor.trainable_variables,
        )
        actor_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                actor_gradients,
                self.actor.trainable_variables,
            )
            if gradient is not None
        ]
        self.actor_optimizer.apply_gradients(actor_pairs)

        # ----------------------------------------------------
        # Automatically update entropy temperature alpha
        # ----------------------------------------------------
        with tf.GradientTape() as alpha_tape:
            _, alpha_log_prob = self.actor.sample(
                states,
                action_bound=self.cfg.action_bound,
                deterministic=False,
            )
            alpha_loss = -tf.reduce_mean(
                self.log_alpha
                * tf.stop_gradient(
                    alpha_log_prob + self.target_entropy
                )
            )

        alpha_gradient = alpha_tape.gradient(
            alpha_loss,
            [self.log_alpha],
        )
        self.alpha_optimizer.apply_gradients(
            [(alpha_gradient[0], self.log_alpha)]
        )

        # ----------------------------------------------------
        # Soft-update target critics
        # ----------------------------------------------------
        for source, target in zip(
            self.q1.variables,
            self.target_q1.variables,
        ):
            target.assign(
                (1.0 - self.cfg.tau) * target
                + self.cfg.tau * source
            )

        for source, target in zip(
            self.q2.variables,
            self.target_q2.variables,
        ):
            target.assign(
                (1.0 - self.cfg.tau) * target
                + self.cfg.tau * source
            )

        return (
            actor_loss,
            critic_loss,
            alpha_loss,
            self.alpha,
        )

    def learn(self) -> dict[str, float]:
        batch = self.memory.sample(self.cfg.batch_size)
        (
            actor_loss,
            critic_loss,
            alpha_loss,
            alpha,
        ) = self._train_step(*batch)

        return {
            "actor_loss": float(actor_loss.numpy()),
            "critic_loss": float(critic_loss.numpy()),
            "alpha_loss": float(alpha_loss.numpy()),
            "alpha": float(alpha.numpy()),
        }


# ============================================================
# The same simplified environment/reward as the user's DDPG
# ============================================================
def jain_fairness_index(
    channel_gains: np.ndarray,
) -> float:
    channel_gains = np.asarray(
        channel_gains,
        dtype=np.float32,
    )

    numerator = np.sum(channel_gains) ** 2
    denominator = (
        len(channel_gains)
        * np.sum(channel_gains ** 2)
        + EPS
    )
    return float(numerator / denominator)


def energy_consumption(
    action: np.ndarray,
) -> float:
    action = np.asarray(action, dtype=np.float32)
    return float(np.sum(action ** 2))


def reward_function(
    channel_gains: np.ndarray,
    action: np.ndarray,
    energy_weight: float,
) -> float:
    scalar_action = float(
        np.asarray(action, dtype=np.float32).reshape(-1)[0]
    )

    channel_reward = abs(
        np.sum(channel_gains * scalar_action)
    )
    fairness = jain_fairness_index(channel_gains)
    energy_penalty = energy_consumption(action)

    return float(
        channel_reward * fairness
        - energy_weight * energy_penalty
    )


# ============================================================
# Training
# ============================================================
def main() -> None:
    set_global_seed(CFG.seed)

    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)
    print("Eager execution:", tf.executing_eagerly())

    original_state_dim = 4 + 2 * CFG.num_users
    combined_state_dim = 2 * original_state_dim

    max_episodes = (
        20 if CFG.quick_test else CFG.max_episodes
    )
    max_episode_steps = (
        20 if CFG.quick_test else CFG.max_episode_steps
    )
    warmup_steps = (
        100 if CFG.quick_test else CFG.warmup_steps
    )

    agent = SACAgent(
        state_dim=combined_state_dim,
        action_dim=CFG.action_dim,
        cfg=CFG,
    )

    start_time = time.time()
    global_step = 0

    episode_rewards: list[float] = []
    episode_fairness_last: list[float] = []
    episode_energy_last: list[float] = []
    episode_fairness_mean: list[float] = []
    episode_energy_mean: list[float] = []

    actor_losses: list[float] = []
    critic_losses: list[float] = []
    alpha_losses: list[float] = []
    alpha_values: list[float] = []

    for episode in range(max_episodes):
        user_positions = np.random.randn(
            CFG.num_users,
            2,
        ).astype(np.float32)
        other_state = np.random.randn(
            1,
            4,
        ).astype(np.float32)

        current_state = np.concatenate(
            [
                other_state.flatten(),
                user_positions.flatten(),
            ]
        ).astype(np.float32)
        previous_state = np.zeros_like(current_state)

        combined_state = np.concatenate(
            [current_state, previous_state]
        ).astype(np.float32)

        episode_reward = 0.0
        fairness_values: list[float] = []
        energy_values: list[float] = []
        latest_losses: dict[str, float] | None = None

        for step in range(max_episode_steps):
            if global_step < warmup_steps:
                action = np.random.uniform(
                    low=0.0,
                    high=CFG.action_bound,
                    size=(CFG.action_dim,),
                ).astype(np.float32)
            else:
                # SAC has an intrinsically stochastic policy.
                # Do not add external Gaussian/OU noise here.
                action = agent.choose_action(
                    combined_state,
                    deterministic=False,
                )

            # Same random-walk state transition as the DDPG code.
            user_positions += (
                np.random.randn(CFG.num_users, 2)
                * 0.1
            ).astype(np.float32)
            other_state += (
                np.random.randn(1, 4)
                * 0.1
            ).astype(np.float32)

            next_current_state = np.concatenate(
                [
                    other_state.flatten(),
                    user_positions.flatten(),
                ]
            ).astype(np.float32)

            # Correct temporal stacking:
            # next input = [s_(t+1), s_t].
            next_combined_state = np.concatenate(
                [next_current_state, current_state]
            ).astype(np.float32)

            distances = np.sqrt(
                np.sum(user_positions ** 2, axis=1)
            )
            channel_gains = 1.0 / (1.0 + distances)

            reward = reward_function(
                channel_gains,
                action,
                CFG.energy_weight,
            )
            fairness = jain_fairness_index(channel_gains)
            energy = energy_consumption(action)

            done = step == max_episode_steps - 1

            agent.store_transition(
                combined_state,
                action,
                reward,
                next_combined_state,
                done,
            )

            if agent.memory.size >= max(
                warmup_steps,
                CFG.batch_size,
            ):
                latest_losses = agent.learn()
                actor_losses.append(
                    latest_losses["actor_loss"]
                )
                critic_losses.append(
                    latest_losses["critic_loss"]
                )
                alpha_losses.append(
                    latest_losses["alpha_loss"]
                )
                alpha_values.append(
                    latest_losses["alpha"]
                )

            episode_reward += reward
            fairness_values.append(fairness)
            energy_values.append(energy)

            previous_state = current_state.copy()
            current_state = next_current_state.copy()
            combined_state = next_combined_state.copy()
            global_step += 1

        episode_rewards.append(float(episode_reward))
        episode_fairness_last.append(
            float(fairness_values[-1])
        )
        episode_energy_last.append(
            float(energy_values[-1])
        )
        episode_fairness_mean.append(
            float(np.mean(fairness_values))
        )
        episode_energy_mean.append(
            float(np.mean(energy_values))
        )

        if latest_losses is None:
            loss_text = "warming up"
        else:
            loss_text = (
                f"actor={latest_losses['actor_loss']:.4f}, "
                f"critic={latest_losses['critic_loss']:.4f}, "
                f"alpha={latest_losses['alpha']:.4f}"
            )

        print(
            f"Episode {episode + 1:4d}/{max_episodes} | "
            f"Reward={episode_reward:10.4f} | "
            f"Fairness(last)={fairness_values[-1]:.4f} | "
            f"Energy(last)={energy_values[-1]:.4f} | "
            f"{loss_text}"
        )

    running_time = time.time() - start_time
    print(f"Running time: {running_time:.2f} seconds")

    os.makedirs("data", exist_ok=True)

    result_path = os.path.join(
        "data",
        f"data_sac_keras3_seed{CFG.seed}.npz",
    )
    np.savez_compressed(
        result_path,
        algorithm=np.asarray("SAC-Keras3"),
        seed=np.int32(CFG.seed),
        tensorflow_version=np.asarray(tf.__version__),
        keras_version=np.asarray(tf.keras.__version__),
        max_episodes=np.int32(max_episodes),
        max_episode_steps=np.int32(max_episode_steps),

        # Names compatible with the current DDPG result file.
        ep_rewardall=np.asarray(
            episode_rewards,
            dtype=np.float32,
        ),
        ep_fairness=np.asarray(
            episode_fairness_last,
            dtype=np.float32,
        ),
        ep_energy=np.asarray(
            episode_energy_last,
            dtype=np.float32,
        ),

        # Additional, more meaningful episode-level summaries.
        ep_fairness_mean=np.asarray(
            episode_fairness_mean,
            dtype=np.float32,
        ),
        ep_energy_mean=np.asarray(
            episode_energy_mean,
            dtype=np.float32,
        ),

        actor_losses=np.asarray(
            actor_losses,
            dtype=np.float32,
        ),
        critic_losses=np.asarray(
            critic_losses,
            dtype=np.float32,
        ),
        alpha_losses=np.asarray(
            alpha_losses,
            dtype=np.float32,
        ),
        alpha_values=np.asarray(
            alpha_values,
            dtype=np.float32,
        ),
        running_time_seconds=np.float32(running_time),
    )
    print("Saved results to:", result_path)

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        episode_rewards,
        label="SAC reward",
    )
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        episode_fairness_mean,
        label="Mean fairness",
    )
    plt.xlabel("Episode")
    plt.ylabel("Jain Fairness Index")
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(
        episode_energy_mean,
        label="Mean energy consumption",
    )
    plt.xlabel("Episode")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()