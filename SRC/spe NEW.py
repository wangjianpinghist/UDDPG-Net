import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from enviroment import Env_cellular as env
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
Pn = 1
K = 10  # the number of grant-based users

MAX_EPISODES = 400
MAX_EP_STEPS = 100
LR_A = 0.002    # learning rate for actor
LR_C = 0.004    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S)
        q = self._build_c(self.S, self.a)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})[0]

    def learn(self):
        indices = np.random.choice(min(MEMORY_CAPACITY, self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        r = np.reshape(r, (1, 1))
        a = np.reshape(a, (1, 1))
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 128, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net, 128, activation=tf.nn.relu, name='l2', trainable=trainable)
            net3 = tf.layers.dense(net2, 128, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 128
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net, 128, activation=tf.nn.relu, name='lx2', trainable=trainable)
            net3 = tf.layers.dense(net2, 128, activation=tf.nn.relu, name='lx3', trainable=trainable)
            return tf.layers.dense(net3, 1, trainable=trainable)

###############################  training  ####################################

s_dim = 4
a_dim = 1
a_bound = 1
state_am = 10000

locationspace = np.linspace(1, 1000, num=K)
location_vector = np.zeros((K, 2))
location_vector[:, 1] = locationspace

location_GF = np.array([[1, 1]])

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 1  # control exploration
t1 = time.time()
ep_rewardall = []
ep_rewardall_greedy = []
ep_rewardall_random = []

for i in range(MAX_EPISODES):
    s = np.random.randn(1, s_dim)  # Initialize state
    s_random = s.copy()
    s_greedy = s.copy()
    ep_reward = 0
    ep_reward_greedy = 0
    ep_reward_random = 0

    for j in range(MAX_EP_STEPS):
        # DDPG Policy
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), 0, 1)
        s_ = np.random.randn(1, s_dim)
        r = np.random.rand(1)
        ddpg.store_transition(s, a, r, s_)
        ddpg.learn()
        s = s_

        # Greedy Policy
        a_greedy = ddpg.choose_action(s_greedy)
        s_greedy = np.random.randn(1, s_dim)
        r_greedy = np.random.rand(1)

        # Random Policy
        a_random = np.random.uniform(0, 1, size=(1,))
        s_random = np.random.randn(1, s_dim)
        r_random = np.random.rand(1)

        ep_reward += r
        ep_reward_greedy += r_greedy
        ep_reward_random += r_random

    print(f'Episode: {i+1}, Reward: {int(ep_reward)}, Greedy Reward: {int(ep_reward_greedy)}, Random Reward: {int(ep_reward_random)}')

    ep_rewardall.append(ep_reward)
    ep_rewardall_greedy.append(ep_reward_greedy)
    ep_rewardall_random.append(ep_reward_random)

print('Running time:', time.time() - t1)

# Plot results
plt.plot(ep_rewardall, label='DDPG')
plt.plot(ep_rewardall_greedy, label='Greedy')
plt.plot(ep_rewardall_random, label='Random')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()
