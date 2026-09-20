
"""
DDPG-GA hybrid baseline
=======================

Environment:
- Same Keras 3 / TensorFlow 2.x runtime as the current SAC and MADDPG scripts.
- Same experimental scale: K=10, 2000 episodes, 100 steps per episode.
- Same scalar action range [0, 1].
- Same state construction: current state + previous state.
- Same reward form:
      reward = channel_reward * fairness - 0.1 * energy

Hybrid mechanism:
1. DDPG actor proposes a seed action.
2. A bounded genetic algorithm searches around that seed.
3. The DDPG critic Q(s, a) is used as the GA fitness.
4. The best critic-scored candidate is executed.
5. The selected transition is stored and used to update DDPG.

Compatibility:
- TensorFlow 2.16+ / Keras 3
- Python 3.10+
- No tf.compat.v1 / tf.Session / tf.layers
- Matplotlib uses the Agg backend, so PyCharm GUI-backend errors are avoided.

Outputs:
- data/data_ddpg_ga_keras3_seed0.npz
- data/ddpg_ga_checkpoint_seed0_ep100.npz
- ...
- data/ddpg_ga_checkpoint_seed0_ep2000.npz
- data/ddpg_ga_models_seed0/actor.weights.h5
- data/ddpg_ga_models_seed0/critic.weights.h5
- data/ddpg_ga_training_seed0.png
"""

import os
import time
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class Config:
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
    hidden_dim: int = 128

    action_dim: int = 1
    action_bound: float = 1.0

    exploration_std_start: float = 0.20
    exploration_std_end: float = 0.05

    # GA parameters
    ga_population: int = 16
    ga_generations: int = 5
    ga_elites: int = 2
    ga_tournament_size: int = 3
    ga_crossover_prob: float = 0.80
    ga_mutation_prob: float = 0.15
    ga_mutation_std_start: float = 0.10
    ga_mutation_std_end: float = 0.03

    # Let the critic warm up before using it as the GA fitness.
    ga_start_steps: int = 2_000

    energy_weight: float = 0.1
    checkpoint_interval: int = 100

    master_seed: int = 0
    environment_seed: int = 0
    exploration_seed: int = 1_000
    replay_seed: int = 2_000
    ga_seed: int = 5_000

    # Set to True before the first run.
    quick_test: bool = False


CFG = Config()
EPS = 1e-6


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================
# Fixed trajectories for fair comparison
# ============================================================
class FixedTrajectoryBank:
    """
    Generates algorithm-independent user-motion and common-state trajectories.

    If the matching trajectory file already exists, it is loaded instead of
    regenerated. This makes repeated runs with the same environment seed see
    exactly the same state transitions.
    """

    def __init__(
        self,
        path: str,
        episodes: int,
        steps: int,
        num_users: int,
        seed: int,
    ) -> None:
        self.path = Path(path)
        self.episodes = episodes
        self.steps = steps
        self.num_users = num_users
        self.seed = seed

        expected_shapes = {
            "initial_user_positions": (episodes, num_users, 2),
            "initial_other_states": (episodes, 4),
            "user_position_deltas": (episodes, steps, num_users, 2),
            "other_state_deltas": (episodes, steps, 4),
        }

        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.initial_user_positions = data["initial_user_positions"]
            self.initial_other_states = data["initial_other_states"]
            self.user_position_deltas = data["user_position_deltas"]
            self.other_state_deltas = data["other_state_deltas"]

            actual_shapes = {
                "initial_user_positions": self.initial_user_positions.shape,
                "initial_other_states": self.initial_other_states.shape,
                "user_position_deltas": self.user_position_deltas.shape,
                "other_state_deltas": self.other_state_deltas.shape,
            }

            if actual_shapes != expected_shapes:
                raise ValueError(
                    "Existing trajectory file dimensions do not match.\n"
                    f"Expected: {expected_shapes}\n"
                    f"Actual: {actual_shapes}\n"
                    f"Delete this file and run again: {self.path}"
                )

            print("Loaded fixed trajectories:", self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.RandomState(seed)

            self.initial_user_positions = rng.randn(
                episodes, num_users, 2
            ).astype(np.float32)

            self.initial_other_states = rng.randn(
                episodes, 4
            ).astype(np.float32)

            self.user_position_deltas = (
                rng.randn(episodes, steps, num_users, 2) * 0.1
            ).astype(np.float32)

            self.other_state_deltas = (
                rng.randn(episodes, steps, 4) * 0.1
            ).astype(np.float32)

            np.savez_compressed(
                self.path,
                initial_user_positions=self.initial_user_positions,
                initial_other_states=self.initial_other_states,
                user_position_deltas=self.user_position_deltas,
                other_state_deltas=self.other_state_deltas,
                environment_seed=np.int32(seed),
            )
            print("Created fixed trajectories:", self.path)

    def reset(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.initial_user_positions[episode].copy(),
            self.initial_other_states[episode].copy(),
        )

    def step_delta(
        self,
        episode: int,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.user_position_deltas[episode, step],
            self.other_state_deltas[episode, step],
        )


# ============================================================
# Replay buffer
# ============================================================
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        seed: int,
    ) -> None:
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        self.rng = np.random.RandomState(seed)

        self.states = np.zeros(
            (capacity, state_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (capacity, action_dim), dtype=np.float32
        )
        self.rewards = np.zeros(
            (capacity, 1), dtype=np.float32
        )
        self.next_states = np.zeros(
            (capacity, state_dim), dtype=np.float32
        )
        self.dones = np.zeros(
            (capacity, 1), dtype=np.float32
        )

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.pointer % self.capacity

        self.states[idx] = np.asarray(
            state, dtype=np.float32
        ).reshape(-1)
        self.actions[idx] = np.asarray(
            action, dtype=np.float32
        ).reshape(-1)
        self.rewards[idx, 0] = float(reward)
        self.next_states[idx] = np.asarray(
            next_state, dtype=np.float32
        ).reshape(-1)
        self.dones[idx, 0] = float(done)

        self.pointer += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        if self.size < batch_size:
            raise RuntimeError(
                f"Replay buffer has {self.size} samples, "
                f"but batch_size={batch_size}."
            )

        idx = self.rng.choice(
            self.size,
            size=batch_size,
            replace=False,
        )

        return (
            tf.convert_to_tensor(self.states[idx]),
            tf.convert_to_tensor(self.actions[idx]),
            tf.convert_to_tensor(self.rewards[idx]),
            tf.convert_to_tensor(self.next_states[idx]),
            tf.convert_to_tensor(self.dones[idx]),
        )


# ============================================================
# Actor and critic
# ============================================================
class Actor(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_bound: float,
        name: str,
    ) -> None:
        super().__init__(name=name)
        self.action_bound = float(action_bound)

        self.fc1 = tf.keras.layers.Dense(
            hidden_dim, activation="relu"
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden_dim, activation="relu"
        )
        self.fc3 = tf.keras.layers.Dense(
            hidden_dim, activation="relu"
        )
        self.output_layer = tf.keras.layers.Dense(
            action_dim, activation="tanh"
        )

    def call(
        self,
        states: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        del training

        x = self.fc1(states)
        x = self.fc2(x)
        x = self.fc3(x)
        raw_action = self.output_layer(x)

        # [-1, 1] -> [0, action_bound]
        return (
            0.5
            * (raw_action + 1.0)
            * self.action_bound
        )


class Critic(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int,
        name: str,
    ) -> None:
        super().__init__(name=name)

        self.fc1 = tf.keras.layers.Dense(
            hidden_dim, activation="relu"
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden_dim, activation="relu"
        )
        self.fc3 = tf.keras.layers.Dense(
            hidden_dim, activation="relu"
        )
        self.q_layer = tf.keras.layers.Dense(
            1, activation=None
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
# DDPG agent
# ============================================================
class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: Config,
    ) -> None:
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(
            cfg.hidden_dim,
            action_dim,
            cfg.action_bound,
            name="Actor",
        )
        self.target_actor = Actor(
            cfg.hidden_dim,
            action_dim,
            cfg.action_bound,
            name="TargetActor",
        )
        self.critic = Critic(
            cfg.hidden_dim,
            name="Critic",
        )
        self.target_critic = Critic(
            cfg.hidden_dim,
            name="TargetCritic",
        )

        dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, action_dim), dtype=tf.float32)

        self.actor(dummy_state)
        self.target_actor(dummy_state)
        self.critic((dummy_state, dummy_action))
        self.target_critic((dummy_state, dummy_action))

        self.target_actor.set_weights(
            self.actor.get_weights()
        )
        self.target_critic.set_weights(
            self.critic.get_weights()
        )

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.actor_lr
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.critic_lr
        )

        if hasattr(self.actor_optimizer, "build"):
            self.actor_optimizer.build(
                self.actor.trainable_variables
            )
        if hasattr(self.critic_optimizer, "build"):
            self.critic_optimizer.build(
                self.critic.trainable_variables
            )

        self.memory = ReplayBuffer(
            cfg.replay_capacity,
            state_dim,
            action_dim,
            cfg.replay_seed,
        )

    def choose_action(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        state_tensor = tf.convert_to_tensor(
            np.asarray(
                state, dtype=np.float32
            ).reshape(1, -1)
        )
        action = self.actor(
            state_tensor, training=False
        ).numpy()[0]

        return np.clip(
            action,
            0.0,
            self.cfg.action_bound,
        )

    def critic_values(
        self,
        state: np.ndarray,
        candidate_actions: np.ndarray,
    ) -> np.ndarray:
        candidates = np.asarray(
            candidate_actions,
            dtype=np.float32,
        ).reshape(-1, self.action_dim)

        repeated_states = np.repeat(
            np.asarray(
                state, dtype=np.float32
            ).reshape(1, -1),
            len(candidates),
            axis=0,
        )

        q = self.critic(
            (
                tf.convert_to_tensor(repeated_states),
                tf.convert_to_tensor(candidates),
            ),
            training=False,
        )
        return q.numpy().reshape(-1)

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
    ) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as critic_tape:
            next_actions = self.target_actor(
                next_states, training=False
            )
            target_q = self.target_critic(
                (next_states, next_actions),
                training=False,
            )

            q_target = (
                rewards
                + self.cfg.gamma
                * (1.0 - dones)
                * target_q
            )
            q_target = tf.stop_gradient(q_target)

            current_q = self.critic(
                (states, actions),
                training=True,
            )
            critic_loss = tf.reduce_mean(
                tf.square(current_q - q_target)
            )

        critic_gradients = critic_tape.gradient(
            critic_loss,
            self.critic.trainable_variables,
        )
        critic_pairs = [
            (g, v)
            for g, v in zip(
                critic_gradients,
                self.critic.trainable_variables,
            )
            if g is not None
        ]
        self.critic_optimizer.apply_gradients(
            critic_pairs
        )

        with tf.GradientTape() as actor_tape:
            policy_actions = self.actor(
                states, training=True
            )
            actor_q = self.critic(
                (states, policy_actions),
                training=False,
            )
            actor_loss = -tf.reduce_mean(actor_q)

        actor_gradients = actor_tape.gradient(
            actor_loss,
            self.actor.trainable_variables,
        )
        actor_pairs = [
            (g, v)
            for g, v in zip(
                actor_gradients,
                self.actor.trainable_variables,
            )
            if g is not None
        ]
        self.actor_optimizer.apply_gradients(
            actor_pairs
        )

        # Soft update targets
        for source, target in zip(
            self.actor.weights,
            self.target_actor.weights,
        ):
            target.assign(
                (1.0 - self.cfg.tau) * target
                + self.cfg.tau * source
            )

        for source, target in zip(
            self.critic.weights,
            self.target_critic.weights,
        ):
            target.assign(
                (1.0 - self.cfg.tau) * target
                + self.cfg.tau * source
            )

        return actor_loss, critic_loss

    def learn(self) -> dict[str, float]:
        actor_loss, critic_loss = self._train_step(
            *self.memory.sample(self.cfg.batch_size)
        )

        return {
            "actor_loss": float(actor_loss.numpy()),
            "critic_loss": float(critic_loss.numpy()),
        }


# ============================================================
# Critic-guided genetic algorithm
# ============================================================
class CriticGuidedGA:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.RandomState(
            cfg.ga_seed
        )

    @staticmethod
    def _linear_schedule(
        start: float,
        end: float,
        step: int,
        total_steps: int,
    ) -> float:
        ratio = min(
            max(step / max(total_steps, 1), 0.0),
            1.0,
        )
        return start + ratio * (end - start)

    def _tournament(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        ids = self.rng.choice(
            len(population),
            size=self.cfg.ga_tournament_size,
            replace=False,
        )
        winner_id = ids[np.argmax(fitness[ids])]
        return population[winner_id].copy()

    def refine(
        self,
        agent: DDPGAgent,
        state: np.ndarray,
        seed_action: np.ndarray,
        global_step: int,
        total_steps: int,
    ) -> tuple[np.ndarray, float, float]:
        cfg = self.cfg

        seed_action = np.asarray(
            seed_action,
            dtype=np.float32,
        ).reshape(1, cfg.action_dim)

        mutation_std = self._linear_schedule(
            cfg.ga_mutation_std_start,
            cfg.ga_mutation_std_end,
            global_step,
            total_steps,
        )

        population = np.zeros(
            (cfg.ga_population, cfg.action_dim),
            dtype=np.float32,
        )

        # First individual is exactly the DDPG action.
        population[0] = seed_action[0]

        # 70% local search around the DDPG seed.
        local_count = max(
            int(round(0.70 * (cfg.ga_population - 1))),
            1,
        )
        population[1:1 + local_count] = (
            seed_action
            + self.rng.normal(
                loc=0.0,
                scale=mutation_std,
                size=(local_count, cfg.action_dim),
            ).astype(np.float32)
        )

        # Remaining individuals are global random candidates.
        random_start = 1 + local_count
        if random_start < cfg.ga_population:
            population[random_start:] = self.rng.uniform(
                low=0.0,
                high=cfg.action_bound,
                size=(
                    cfg.ga_population - random_start,
                    cfg.action_dim,
                ),
            ).astype(np.float32)

        population = np.clip(
            population,
            0.0,
            cfg.action_bound,
        )

        seed_q = float(
            agent.critic_values(
                state,
                seed_action,
            )[0]
        )

        for _ in range(cfg.ga_generations):
            fitness = agent.critic_values(
                state,
                population,
            )

            elite_ids = np.argsort(
                fitness
            )[-cfg.ga_elites:]

            next_population = [
                population[idx].copy()
                for idx in elite_ids
            ]

            while len(next_population) < cfg.ga_population:
                parent1 = self._tournament(
                    population,
                    fitness,
                )
                parent2 = self._tournament(
                    population,
                    fitness,
                )

                child1 = parent1.copy()
                child2 = parent2.copy()

                if (
                    self.rng.rand()
                    < cfg.ga_crossover_prob
                ):
                    alpha = self.rng.rand(
                        cfg.action_dim
                    ).astype(np.float32)

                    child1 = (
                        alpha * parent1
                        + (1.0 - alpha) * parent2
                    )
                    child2 = (
                        alpha * parent2
                        + (1.0 - alpha) * parent1
                    )

                for child in (child1, child2):
                    mutation_mask = (
                        self.rng.rand(cfg.action_dim)
                        < cfg.ga_mutation_prob
                    )

                    if np.any(mutation_mask):
                        child[mutation_mask] += self.rng.normal(
                            loc=0.0,
                            scale=mutation_std,
                            size=int(np.sum(mutation_mask)),
                        ).astype(np.float32)

                    child = np.clip(
                        child,
                        0.0,
                        cfg.action_bound,
                    )

                    next_population.append(
                        child.astype(np.float32)
                    )

                    if len(next_population) >= cfg.ga_population:
                        break

            population = np.asarray(
                next_population[:cfg.ga_population],
                dtype=np.float32,
            )

        final_fitness = agent.critic_values(
            state,
            population,
        )
        best_id = int(np.argmax(final_fitness))
        best_action = population[best_id].copy()
        best_q = float(final_fitness[best_id])

        # Safety fallback
        if best_q < seed_q:
            return seed_action[0].copy(), seed_q, seed_q

        return best_action, seed_q, best_q


# ============================================================
# Same reward/statistics as the current comparison environment
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
    action = np.asarray(
        action,
        dtype=np.float32,
    )
    return float(np.sum(action ** 2))


def reward_function(
    channel_gains: np.ndarray,
    action: np.ndarray,
    energy_weight: float,
) -> float:
    scalar_action = float(
        np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)[0]
    )

    channel_reward = abs(
        np.sum(channel_gains * scalar_action)
    )
    fairness = jain_fairness_index(
        channel_gains
    )
    energy_penalty = energy_consumption(
        action
    )

    return float(
        channel_reward * fairness
        - energy_weight * energy_penalty
    )


# ============================================================
# Save results
# ============================================================
def save_results(
    path: str,
    *,
    completed_episodes: int,
    episode_rewards: list[float],
    fairness_last: list[float],
    energy_last: list[float],
    fairness_mean: list[float],
    energy_mean: list[float],
    actor_action_mean: list[float],
    ga_action_mean: list[float],
    ga_q_gain_mean: list[float],
    ga_usage_rate: list[float],
    actor_losses: list[float],
    critic_losses: list[float],
    runtime_seconds: float,
    trajectory_path: str,
) -> None:
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        algorithm=np.asarray("DDPG-GA"),
        master_seed=np.int32(CFG.master_seed),
        environment_seed=np.int32(
            CFG.environment_seed
        ),
        exploration_seed=np.int32(
            CFG.exploration_seed
        ),
        replay_seed=np.int32(
            CFG.replay_seed
        ),
        ga_seed=np.int32(CFG.ga_seed),

        completed_episodes=np.int32(
            completed_episodes
        ),
        max_episode_steps=np.int32(
            CFG.max_episode_steps
        ),

        tensorflow_version=np.asarray(
            tf.__version__
        ),
        keras_version=np.asarray(
            getattr(
                tf.keras,
                "__version__",
                "unknown",
            )
        ),
        fixed_trajectory_path=np.asarray(
            trajectory_path
        ),

        ga_population=np.int32(
            CFG.ga_population
        ),
        ga_generations=np.int32(
            CFG.ga_generations
        ),
        ga_elites=np.int32(
            CFG.ga_elites
        ),
        ga_start_steps=np.int32(
            CFG.ga_start_steps
        ),

        # Standard comparison fields
        ep_rewardall=np.asarray(
            episode_rewards,
            dtype=np.float32,
        ),
        ep_fairness=np.asarray(
            fairness_last,
            dtype=np.float32,
        ),
        ep_energy=np.asarray(
            energy_last,
            dtype=np.float32,
        ),
        ep_fairness_mean=np.asarray(
            fairness_mean,
            dtype=np.float32,
        ),
        ep_energy_mean=np.asarray(
            energy_mean,
            dtype=np.float32,
        ),

        # Hybrid-specific fields
        ep_actor_action_mean=np.asarray(
            actor_action_mean,
            dtype=np.float32,
        ),
        ep_ga_action_mean=np.asarray(
            ga_action_mean,
            dtype=np.float32,
        ),
        ep_ga_q_gain_mean=np.asarray(
            ga_q_gain_mean,
            dtype=np.float32,
        ),
        ep_ga_usage_rate=np.asarray(
            ga_usage_rate,
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
        running_time_seconds=np.float32(
            runtime_seconds
        ),
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    set_global_seed(CFG.master_seed)

    print("TensorFlow:", tf.__version__)
    print(
        "Keras:",
        getattr(
            tf.keras,
            "__version__",
            "unknown",
        ),
    )
    print(
        "Eager execution:",
        tf.executing_eagerly(),
    )

    max_episodes = (
        20
        if CFG.quick_test
        else CFG.max_episodes
    )
    max_episode_steps = (
        20
        if CFG.quick_test
        else CFG.max_episode_steps
    )
    warmup_steps = (
        100
        if CFG.quick_test
        else CFG.warmup_steps
    )
    ga_start_steps = (
        150
        if CFG.quick_test
        else CFG.ga_start_steps
    )

    original_state_dim = (
        4 + 2 * CFG.num_users
    )
    combined_state_dim = (
        2 * original_state_dim
    )

    data_dir = Path("data")
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_path = (
        data_dir
        / (
            "fixed_env_trajectory_"
            f"seed{CFG.environment_seed}_"
            f"ep{max_episodes}_"
            f"steps{max_episode_steps}_"
            f"K{CFG.num_users}.npz"
        )
    )

    trajectories = FixedTrajectoryBank(
        path=str(trajectory_path),
        episodes=max_episodes,
        steps=max_episode_steps,
        num_users=CFG.num_users,
        seed=CFG.environment_seed,
    )

    agent = DDPGAgent(
        state_dim=combined_state_dim,
        action_dim=CFG.action_dim,
        cfg=CFG,
    )
    ga = CriticGuidedGA(CFG)

    exploration_rng = np.random.RandomState(
        CFG.exploration_seed
    )

    total_steps = (
        max_episodes * max_episode_steps
    )
    global_step = 0
    start_time = time.time()

    episode_rewards = []
    fairness_last = []
    energy_last = []
    fairness_mean = []
    energy_mean = []

    actor_action_mean = []
    ga_action_mean = []
    ga_q_gain_mean = []
    ga_usage_rate = []

    actor_losses = []
    critic_losses = []

    for episode in range(max_episodes):
        user_positions, other_state = (
            trajectories.reset(episode)
        )

        current_state = np.concatenate(
            [
                other_state.flatten(),
                user_positions.flatten(),
            ]
        ).astype(np.float32)

        previous_state = np.zeros_like(
            current_state
        )

        combined_state = np.concatenate(
            [current_state, previous_state]
        ).astype(np.float32)

        episode_reward = 0.0
        fairness_values = []
        energy_values = []
        actor_action_values = []
        ga_action_values = []
        q_gain_values = []
        ga_used_values = []

        latest_losses = None

        for step in range(max_episode_steps):
            if global_step < warmup_steps:
                actor_action = exploration_rng.uniform(
                    low=0.0,
                    high=CFG.action_bound,
                    size=(CFG.action_dim,),
                ).astype(np.float32)

                executed_action = actor_action.copy()
                q_gain = 0.0
                ga_used = 0.0

            else:
                deterministic_action = agent.choose_action(
                    combined_state
                )

                progress = min(
                    global_step / max(total_steps, 1),
                    1.0,
                )
                exploration_std = (
                    CFG.exploration_std_start
                    + progress
                    * (
                        CFG.exploration_std_end
                        - CFG.exploration_std_start
                    )
                )

                actor_action = np.clip(
                    deterministic_action
                    + exploration_rng.normal(
                        loc=0.0,
                        scale=exploration_std,
                        size=(CFG.action_dim,),
                    ).astype(np.float32),
                    0.0,
                    CFG.action_bound,
                )

                if global_step >= ga_start_steps:
                    (
                        executed_action,
                        seed_q,
                        best_q,
                    ) = ga.refine(
                        agent=agent,
                        state=combined_state,
                        seed_action=actor_action,
                        global_step=global_step,
                        total_steps=total_steps,
                    )
                    q_gain = best_q - seed_q
                    ga_used = 1.0
                else:
                    executed_action = actor_action.copy()
                    q_gain = 0.0
                    ga_used = 0.0

            user_delta, other_delta = (
                trajectories.step_delta(
                    episode,
                    step,
                )
            )

            user_positions = (
                user_positions + user_delta
            )
            other_state = (
                other_state + other_delta
            )

            next_current_state = np.concatenate(
                [
                    other_state.flatten(),
                    user_positions.flatten(),
                ]
            ).astype(np.float32)

            # Correct temporal stacking:
            # next input = [s_(t+1), s_t]
            next_combined_state = np.concatenate(
                [
                    next_current_state,
                    current_state,
                ]
            ).astype(np.float32)

            distances = np.sqrt(
                np.sum(
                    user_positions ** 2,
                    axis=1,
                )
            )
            channel_gains = (
                1.0 / (1.0 + distances)
            )

            reward = reward_function(
                channel_gains,
                executed_action,
                CFG.energy_weight,
            )
            fairness = jain_fairness_index(
                channel_gains
            )
            energy = energy_consumption(
                executed_action
            )

            done = (
                step == max_episode_steps - 1
            )

            agent.store_transition(
                combined_state,
                executed_action,
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

            episode_reward += reward
            fairness_values.append(fairness)
            energy_values.append(energy)
            actor_action_values.append(
                float(np.mean(actor_action))
            )
            ga_action_values.append(
                float(np.mean(executed_action))
            )
            q_gain_values.append(float(q_gain))
            ga_used_values.append(float(ga_used))

            current_state = (
                next_current_state.copy()
            )
            combined_state = (
                next_combined_state.copy()
            )
            global_step += 1

        episode_rewards.append(
            float(episode_reward)
        )
        fairness_last.append(
            float(fairness_values[-1])
        )
        energy_last.append(
            float(energy_values[-1])
        )
        fairness_mean.append(
            float(np.mean(fairness_values))
        )
        energy_mean.append(
            float(np.mean(energy_values))
        )

        actor_action_mean.append(
            float(np.mean(actor_action_values))
        )
        ga_action_mean.append(
            float(np.mean(ga_action_values))
        )
        ga_q_gain_mean.append(
            float(np.mean(q_gain_values))
        )
        ga_usage_rate.append(
            float(np.mean(ga_used_values))
        )

        if latest_losses is None:
            loss_text = "warming up"
        else:
            loss_text = (
                f"actor={latest_losses['actor_loss']:.4f}, "
                f"critic={latest_losses['critic_loss']:.4f}"
            )

        print(
            f"Episode {episode + 1:4d}/{max_episodes} | "
            f"Reward={episode_reward:10.4f} | "
            f"Fairness(last)={fairness_values[-1]:.4f} | "
            f"Energy(last)={energy_values[-1]:.4f} | "
            f"ActorAction(mean)={np.mean(actor_action_values):.4f} | "
            f"GAAction(mean)={np.mean(ga_action_values):.4f} | "
            f"GA_Q_Gain(mean)={np.mean(q_gain_values):.4f} | "
            f"{loss_text}"
        )

        if (
            (episode + 1)
            % CFG.checkpoint_interval
            == 0
        ):
            checkpoint_path = (
                data_dir
                / (
                    "ddpg_ga_checkpoint_"
                    f"seed{CFG.master_seed}_"
                    f"ep{episode + 1}.npz"
                )
            )

            save_results(
                str(checkpoint_path),
                completed_episodes=episode + 1,
                episode_rewards=episode_rewards,
                fairness_last=fairness_last,
                energy_last=energy_last,
                fairness_mean=fairness_mean,
                energy_mean=energy_mean,
                actor_action_mean=actor_action_mean,
                ga_action_mean=ga_action_mean,
                ga_q_gain_mean=ga_q_gain_mean,
                ga_usage_rate=ga_usage_rate,
                actor_losses=actor_losses,
                critic_losses=critic_losses,
                runtime_seconds=(
                    time.time() - start_time
                ),
                trajectory_path=str(
                    trajectory_path
                ),
            )

            print(
                "Checkpoint saved:",
                checkpoint_path,
            )

    runtime_seconds = (
        time.time() - start_time
    )

    final_path = (
        data_dir
        / (
            "data_ddpg_ga_keras3_"
            f"seed{CFG.master_seed}.npz"
        )
    )

    save_results(
        str(final_path),
        completed_episodes=max_episodes,
        episode_rewards=episode_rewards,
        fairness_last=fairness_last,
        energy_last=energy_last,
        fairness_mean=fairness_mean,
        energy_mean=energy_mean,
        actor_action_mean=actor_action_mean,
        ga_action_mean=ga_action_mean,
        ga_q_gain_mean=ga_q_gain_mean,
        ga_usage_rate=ga_usage_rate,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        runtime_seconds=runtime_seconds,
        trajectory_path=str(
            trajectory_path
        ),
    )

    print(
        f"Running time: "
        f"{runtime_seconds:.2f} seconds"
    )
    print(
        "Saved final results to:",
        final_path,
    )

    # Save weights
    model_dir = (
        data_dir
        / (
            "ddpg_ga_models_"
            f"seed{CFG.master_seed}"
        )
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    actor_weight_path = (
        model_dir / "actor.weights.h5"
    )
    critic_weight_path = (
        model_dir / "critic.weights.h5"
    )

    agent.actor.save_weights(
        actor_weight_path
    )
    agent.critic.save_weights(
        critic_weight_path
    )

    print(
        "Saved actor weights:",
        actor_weight_path,
    )
    print(
        "Saved critic weights:",
        critic_weight_path,
    )

    # Non-interactive plot
    figure_path = (
        data_dir
        / (
            "ddpg_ga_training_"
            f"seed{CFG.master_seed}.png"
        )
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    axes[0, 0].plot(
        episode_rewards
    )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel(
        "Episodic Reward"
    )
    axes[0, 0].set_title(
        "DDPG-GA Reward"
    )

    axes[0, 1].plot(
        energy_mean
    )
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel(
        "Mean Energy"
    )
    axes[0, 1].set_title(
        "Energy Consumption"
    )

    axes[1, 0].plot(
        ga_q_gain_mean
    )
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel(
        "Mean Critic-Q Gain"
    )
    axes[1, 0].set_title(
        "GA Improvement over DDPG"
    )

    axes[1, 1].plot(
        ga_usage_rate
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel(
        "GA Usage Rate"
    )
    axes[1, 1].set_title(
        "GA Activation"
    )

    fig.tight_layout()
    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        "Saved training figure:",
        figure_path,
    )


if __name__ == "__main__":
    main()
