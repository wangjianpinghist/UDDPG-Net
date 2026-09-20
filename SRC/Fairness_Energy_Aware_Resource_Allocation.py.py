import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
Pn = 1
K = 10  # the number of grant-based users

MAX_EPISODES = 2000
MAX_EP_STEPS = 100
LR_A = 0.002  # learning rate for actor
LR_C = 0.004  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 4 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim * 2], 's')  # 当前状态和前一状态
        self.S_ = tf.placeholder(tf.float32, [None, s_dim * 2], 's_')  # 下一状态和当前状态
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # 构建 Actor 和 Critic 网络
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

    def choose_action(self, s, s_prev):
        state_input = np.hstack((s, s_prev))
        state_input = np.reshape(state_input, (1, -1))
        return self.sess.run(self.a, {self.S: state_input})[0]

    def learn(self):
        indices = np.random.choice(min(MEMORY_CAPACITY, self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim * 2]
        ba = bt[:, self.s_dim * 2: self.s_dim * 2 + self.a_dim]
        br = bt[:, self.s_dim * 2 + self.a_dim: self.s_dim * 2 + self.a_dim + 1]
        bs_ = bt[:, -self.s_dim * 2:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, s_prev, a, r, s_, s_prev_):
        state_combined = np.hstack((s.flatten(), s_prev.flatten()))
        next_state_combined = np.hstack((s_.flatten(), s_prev_.flatten()))

        r = np.reshape(r, (1, 1))
        a = np.reshape(a, (1, 1))
        transition = np.hstack((state_combined, a.flatten(), r.flatten(), next_state_combined))

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
            w1_s = tf.get_variable('w1_s', [self.s_dim * 2, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net, 128, activation=tf.nn.relu, name='lx2', trainable=trainable)
            net3 = tf.layers.dense(net2, 128, activation=tf.nn.relu, name='lx3', trainable=trainable)
            return tf.layers.dense(net3, 1, name='q', trainable=trainable)


###############################  training  ####################################

s_dim = 4 + 2 * K  # 原始状态维度 + 用户位置信息（每个用户2维坐标）
a_dim = 1
a_bound = 1

ddpg = DDPG(a_dim, s_dim, a_bound)


# 定义公平性和能耗函数
def jain_fairness_index(channel_gains):
    return np.sum(channel_gains) ** 2 / (len(channel_gains) * np.sum(channel_gains ** 2))


def energy_consumption(action):
    return np.sum(action ** 2)  # 假设能耗与动作的平方成正比


def reward_function(channel_gains, action, user_positions):
    # 信道增益奖励
    channel_reward = np.abs(np.sum(channel_gains * action))

    # 公平性奖励
    fairness = jain_fairness_index(channel_gains)

    # 能耗惩罚
    energy_penalty = energy_consumption(action)

    # 综合奖励
    reward = channel_reward * fairness - 0.1 * energy_penalty  # 0.1 是能耗惩罚的权重

    return reward


var = 1  # control exploration
import time

t1 = time.time()
ep_rewardall = []
ep_rewardall_greedy = []
ep_rewardall_random = []
ep_fairness = []  # 记录公平性
ep_energy = []  # 记录能耗

for i in range(MAX_EPISODES):
    # 初始化用户位置和其他状态
    user_pos = np.random.randn(K, 2)  # 每个用户有2维坐标
    other_state = np.random.randn(1, 4)  # 原始状态
    s = np.concatenate([other_state.flatten(), user_pos.flatten()]).reshape(1, -1)
    s_prev = np.zeros_like(s)  # 初始化前一状态为全零
    ep_reward = 0
    r_greedy = 0
    r_random = 0

    for j in range(MAX_EP_STEPS):
        # DDPG 策略
        a = ddpg.choose_action(s, s_prev)
        a = np.clip(np.random.normal(a, var), 0, 1)

        # 更新用户位置和其他状态
        user_pos += np.random.randn(K, 2) * 0.1  # 用户位置随机游走
        other_state += np.random.randn(1, 4) * 0.1  # 更新原始状态
        s_ = np.concatenate([other_state.flatten(), user_pos.flatten()]).reshape(1, -1)

        # 计算奖励（基于新的奖励函数）
        distance = np.sqrt(np.sum(user_pos ** 2, axis=1))  # 用户到原点的距离
        channel_gain = 1 / (1 + distance)  # 信道增益与距离相关
        r = reward_function(channel_gain, a, user_pos)  # 使用新的奖励函数
        ep_reward += r

        # 计算公平性和能耗
        fairness = jain_fairness_index(channel_gain)
        energy = energy_consumption(a)

        # 贪心策略
        r_greedy = np.abs(np.sum(s * 0.5))
        # 随机策略
        random_action = np.random.uniform(0, 1, size=(1, s_dim))  # 调整 random_action 的形状
        r_random = np.abs(np.sum(s * random_action))

        # 更新状态
        ddpg.store_transition(s, s_prev, a, r, s_, s_prev)
        ddpg.learn()
        s_prev = s.copy()
        s = s_.copy()

    # 记录每个策略的总奖励、公平性和能耗
    ep_rewardall.append(ep_reward)
    ep_rewardall_greedy.append(r_greedy)
    ep_rewardall_random.append(r_random)
    ep_fairness.append(fairness)
    ep_energy.append(energy)

    # 输出每个回合的结果
    print(
        f"Episode {i + 1}, DDPG Reward: {ep_reward:.2f}, Greedy Reward: {r_greedy:.2f}, Random Reward: {r_random:.2f}, Fairness: {fairness:.2f}, Energy: {energy:.2f}")

print('Running time: ', time.time() - t1)

# 绘制奖励曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ep_rewardall, "^-", label='DDPG: rewards')
plt.plot(ep_rewardall_greedy, "+:", label='Greedy: rewards')
plt.plot(ep_rewardall_random, "o--", label='Random: rewards')
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.legend()

# 绘制公平性和能耗曲线
plt.subplot(1, 2, 2)
plt.plot(ep_fairness, label='Fairness')
plt.xlabel("Episode")
plt.ylabel("Fairness Index")
plt.legend()

plt.tight_layout()
plt.show()

# 绘制能耗曲线
plt.plot(ep_energy, label='Energy Consumption')
plt.xlabel("Episode")
plt.ylabel("Energy")
plt.legend()
plt.show()

# 保存结果
np.savez_compressed('data/data_snapshot1',
                    ep_rewardall=ep_rewardall,
                    ep_rewardall_greedy=ep_rewardall_greedy,
                    ep_rewardall_random=ep_rewardall_random,
                    ep_fairness=ep_fairness,
                    ep_energy=ep_energy)
