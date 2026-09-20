import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

#####################  Hyper Parameters  ####################
K = 10  # number of users
s_dim = 4 + 2 * K  # state dimension (original state + user locations)
a_dim = 1

gamma = 0.9  # discount factor
clip_epsilon = 0.2  # PPO clip parameter
lr_actor = 0.0003
lr_critic = 0.001
update_steps = 10
batch_size = 64

epochs = 2000
steps_per_epoch = 100


class PPO:
    def __init__(self):
        self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32, [None, s_dim], "state")
        self.a = tf.placeholder(tf.float32, [None, a_dim], "action")
        self.r = tf.placeholder(tf.float32, [None, 1], "reward")
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")

        self.actor_old = self._build_actor('oldpi')
        self.actor = self._build_actor('pi')
        self.critic = self._build_critic()

        with tf.variable_scope('loss_actor'):
            ratio = tf.exp(self.actor.log_prob - self.actor_old.log_prob)
            surr = ratio * self.advantage
            self.loss_actor = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * self.advantage
            ))
        with tf.variable_scope('train_actor'):
            self.train_actor_op = tf.train.AdamOptimizer(lr_actor).minimize(self.loss_actor, var_list=self.actor.params)

        with tf.variable_scope('loss_critic'):
            self.v = self.critic.value
            self.loss_critic = tf.reduce_mean(tf.square(self.r - self.v))
        with tf.variable_scope('train_critic'):
            self.train_critic_op = tf.train.AdamOptimizer(lr_critic).minimize(self.loss_critic)

        self.sess.run(tf.global_variables_initializer())

    def _build_actor(self, name):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.s, 128, tf.nn.relu)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu)
            mu = tf.layers.dense(l2, a_dim, tf.nn.tanh)
            sigma = tf.layers.dense(l2, a_dim, tf.nn.softplus)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        actor = lambda: None
        actor.sample = tf.clip_by_value(norm_dist.sample(1), 0, 1)
        actor.log_prob = norm_dist.log_prob(self.a)
        actor.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return actor

    def _build_critic(self):
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.s, 128, tf.nn.relu)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu)
            v = tf.layers.dense(l2, 1)
        critic = lambda: None
        critic.value = v
        return critic

    def update(self, s, a, r):
        v = self.sess.run(self.v, {self.s: s})
        adv = r - v
        old_params = self.sess.run(self.actor.params)
        for old, new in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='oldpi'), old_params):
            self.sess.run(tf.assign(old, new))

        for _ in range(update_steps):
            self.sess.run(self.train_actor_op, {self.s: s, self.a: a, self.advantage: adv})
            self.sess.run(self.train_critic_op, {self.s: s, self.r: r})

    def choose_action(self, s):
        return self.sess.run(self.actor.sample, {self.s: s})[0]


def jain_fairness_index(channel_gains):
    return np.sum(channel_gains) ** 2 / (len(channel_gains) * np.sum(channel_gains ** 2))

def energy_consumption(action):
    return np.sum(action ** 2)

def reward_function(channel_gains, action):
    fairness = jain_fairness_index(channel_gains)
    energy_penalty = energy_consumption(action)
    return np.abs(np.sum(channel_gains * action)) * fairness - 0.1 * energy_penalty

ppo = PPO()
all_rewards = []

for epoch in range(epochs):
    ep_reward = 0
    buffer_s, buffer_a, buffer_r = [], [], []
    user_pos = np.random.randn(K, 2)
    other_state = np.random.randn(1, 4)
    s = np.concatenate([other_state.flatten(), user_pos.flatten()]).reshape(1, -1)

    for t in range(steps_per_epoch):
        a = ppo.choose_action(s)
        user_pos += np.random.randn(K, 2) * 0.1
        other_state += np.random.randn(1, 4) * 0.1
        s_ = np.concatenate([other_state.flatten(), user_pos.flatten()]).reshape(1, -1)

        distance = np.sqrt(np.sum(user_pos ** 2, axis=1))
        channel_gain = 1 / (1 + distance)
        r = reward_function(channel_gain, a)

        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append([r])

        ep_reward += r
        s = s_

    ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_r))
    all_rewards.append(ep_reward)
    print(f"Epoch {epoch+1}, Reward: {ep_reward:.2f}")

plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO Reward Trend')
plt.grid(True)
plt.show()
