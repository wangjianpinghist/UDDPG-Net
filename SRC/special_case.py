"""
Note: This is based on Mofan's codes from: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
This code is used to generate the figures for random fading without average (a snapshot).
Simply change K
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from enviroment import Env_cellular as env
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
Pn = 1
K=10 # the number of grant based users 表示授予基用户的数量。这可能与强化学习任务中的环境设置有关，例如在某些网络环境下，任务的目标是优化多个用户的连接或带宽分配。

MAX_EPISODES = 400 #表示训练过程中最大迭代的次数（即最大回合数）。每个回合会训练一段时间，并根据奖励反馈调整策略
MAX_EP_STEPS = 100 #表示每个回合（episode）中最大步数。每个回合会有多个步骤，直到达到最大步数或者任务完成。
LR_A = 0.002    # learning rate for actor  LR_A 是演员网络（Actor）使用的学习率。演员网络负责选择动作，因此其学习率控制着动作选择策略的更新速度。较低的学习率有助于训练过程的稳定，但可能会导致收敛较慢。
LR_C = 0.004    # learning rate for critic LR_C 是评论员网络（Critic）使用的学习率。评论员网络评估由演员选择的动作的价值，即通过计算 Q 值来评估动作的好坏。评论员网络的学习率也影响更新 Q 值的速度。
GAMMA = 0.9     # reward discount GAMMA 是奖励折扣因子，也称为折扣因子。它决定了未来奖励对当前决策的影响。较低的 GAMMA 值使得模型更关注短期奖励，而较高的 GAMMA 值则让模型更加关注长期奖励
TAU = 0.01      # soft replacement TAU 用于软目标更新。在DDPG中，演员和评论员的目标网络会逐渐更新（通过软更新机制），而不是完全替换。这有助于提高训练过程的稳定性。TAU 的值越小，目标网络的更新越平滑。
MEMORY_CAPACITY = 100000
# MEMORY_CAPACITY 表示经验回放池的容量。经验回放池用于存储智能体在与环境交互过程中收集的状态、动作、奖励和下一个状态的转移。训练过程中，智能体从回放池中随机抽取样本来更新策略。较大的池容量可以存储更多的经验，有助于提高训练效率和稳定性。
BATCH_SIZE = 32
# BATCH_SIZE 表示每次训练时使用的样本数量。在训练过程中，智能体从经验回放池中随机抽取一个批次的样本进行训练。批次大小决定了每次更新时使用的经验数量。较大的批次有助于训练的稳定性，但会增加计算成本。
# MEMORY_CAPACITY 增加到了 100000，相比之前的 10000，这意味着经验回放池可以存储更多的转移数据。这对更长时间的训练或者更复杂的环境有帮助，因为可以使用更多的历史经验来训练模型。


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # self.memory是经验回放池
        self.pointer = 0           # self.pointer 是指针，用于指示下一个经验存储的位置。当回放池满时，pointer 会覆盖最旧的经验。
        self.sess = tf.Session()


        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')   # self.S 是当前状态的占位符，表示在每次训练或预测时，智能体会传入当前的环境状态
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_') # self.S_ 是下一状态的占位符，表示在每次训练时，智能体会传入执行某个动作后得到的下一状态
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')       # self.R 是奖励的占位符，表示在每次训练时，智能体会传入当前状态-动作对对应的即时奖励值
        # a_dim是动作维度 s_dim是状态维度 a_bound是动作范围
        self.a = self._build_a(self.S,)      # 生成演员网络输出（即动作）
        q = self._build_c(self.S, self.a, )  # 生成评论员网络输出（即Q值）
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')  # 用于获取演员网络（Actor）的可训练参数
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic') # 用于获取评论员网络（Critic）的可训练参数
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement 软更新

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation 软更新
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)      # 更新演员网络

        with tf.control_dependencies(target_update):    # 更新评论员网络
            q_target = self.R + GAMMA * q_              # Q值的计算和优化
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s })[0]

    def learn(self):
        indices = np.random.choice(min(MEMORY_CAPACITY,self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]                     # 从经验回放池中采样出来的一批数据
        bs = bt[:, :self.s_dim]                          # 这批数据中的状态 s
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]  # 这批数据中的动作 a
        br = bt[:, -self.s_dim - 1: -self.s_dim]         # 这批数据中的奖励 r
        bs_ = bt[:, -self.s_dim:]                        # 这批数据中的下一状态 s_

        self.sess.run(self.atrain, {self.S: bs})         # 这两行代码相当于一次训练循环，在每一轮中，演员和评论员网络都会根据从经验回放池中抽样的经验来更新其参数，以便逐渐提高策略的质量
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        # learn这一部分，该方法是 DDPG 算法中的核心部分，用于训练演员和评论员网络
    def store_transition(self, s, a, r, s_):     # 定义了一个存储经验的函数
        r = np.reshape(r,(1,1))
        a = np.reshape(a,(1,1))  # 将当前状态和当前动作的数据转化为二维数组
        #print(f"state is {s}, action is {a}, reward is {r}, next state is {s_}")
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # 用新记忆替换旧记忆
        self.memory[index, :] = transition
        self.pointer += 1
        # 这一段是将智能体与环境生成的（状态，动作，奖励，下一目标状态）存储在经验回放池中
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            a2 = tf.layers.dense(net, 64, activation=tf.nn.tanh, name='l2', trainable=trainable)
            #a3 = tf.layers.dense(a2, 30, activation=tf.nn.tanh, name='l3', trainable=trainable)

            a = tf.layers.dense(a2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
            # 一个具有 64 个神经元的隐层，使用 ReLU 激活函数
            # 一个具有 64 个神经元的隐层，使用 tanh 激活函数
            # 输出层，生成最终的动作，使用 tanh 激活函数，并通过与动作空间上界 a_bound 相乘进行缩放，确保输出的动作值在合理的范围内

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)  # 状态输入的权重矩阵
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)  # 动作输入的权重矩阵
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)  # 偏置项
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # 根据状态 s 和动作 a 计算加权和，经过 ReLU 激活函数得到第一个隐藏层的输出
            net2 = tf.layers.dense(net, 64, activation=tf.nn.relu, name='lx2', trainable=trainable)
            # 将第一个隐藏层的输出传入第二个全连接层，继续提取特征
            #net3 = tf.layers.dense(net2, 30, activation=tf.nn.relu, name='lx3', trainable=trainable)

            #not sure about this part
            return tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
            # 通过最后的全连接层，得到一个标量值，表示 Q 值

###############################  training  ####################################



s_dim = 4# 维度：状态空间大小
a_dim = 1# 维度: 动作空间大小
a_bound = 1 # 动作空间的上界
state_am = 10000 # 状态放大系数，用于调整状态的范围

locationspace = np.linspace(1,1000, num=K)
location_vector = np.zeros((K, 2))
location_vector[:,1] = locationspace


location_GF = np.array([[1,1]])# np.ones((1, 2))


ddpg = DDPG(a_dim, s_dim, a_bound)

var = 1  # control exploration
t1 = time.time()
ep_rewardall = []
ep_rewardall_greedy = []
ep_rewardall_random = []
for i in range(MAX_EPISODES):
    ##### fading for GB user
    hnx1 = np.random.randn(K, 2)
    hnx2 = np.random.randn(K, 2)
    fading_n = hnx1 ** 2 + hnx2 ** 2
    #### fading for GF user
    h0x1 = np.random.randn(1, 1)
    h0x2 = np.random.randn(1, 1)
    fading_0 = h0x1[0, 0] ** 2 + h0x2[0, 0] ** 2
    #if fading_0<0.01:
    #    print(fading_0)

    myenv = env(MAX_EP_STEPS, s_dim, location_vector, location_GF, K, Pn, fading_n, fading_0)

    batter_ini = myenv.reset()
    s = myenv.channel_sequence[i%myenv.K,:].tolist()
    s.append(myenv.h0)
    s.append(batter_ini)
    s = np.reshape(s,(1,s_dim))
    s = s*state_am #amplify the state
    s_greedy = s
    s_random = s
    #print(s[0,0:2])
    ep_reward = 0
    ep_reward_random = 0
    ep_reward_greedy = 0
    for j in range(MAX_EP_STEPS):

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), 0, 1)    # 为动作选择添加随机性以进行探索 生成一个围绕a（即Actor网络预测的动作）为中心的高斯噪声
        r, s_, done = myenv.step(a,s/state_am,j)
        s_ = s_ * state_am
        ddpg.store_transition(s, a, r, s_)
        if var >0.1:        # 在训练过程中探索噪声的幅度是逐渐衰减的
            var *= .9999    # decay the action randomness

        ddpg.learn()        # 更新智能体的策略
        s = s_              # 更新当前状态
        ep_reward += r      # 累积回合奖励

        ##### greedy
        r_greedy, s_next_greedy, done = myenv.step_greedy(s_greedy/state_am, j)
        s_greedy = s_next_greedy*state_am
        ep_reward_greedy += r_greedy

        ##### random
        r_random, s_next_random, done = myenv.step_random(s_random/state_am, j)
        s_random = s_next_random*state_am
        ep_reward_random += r_random


        if j == MAX_EP_STEPS-1:
            #print(f"Episode: {i}, reward is {ep_reward}, and Explore is {var}")
            print('Episode:', i, ' Reward: %i' % int(ep_reward),'fading', fading_0, 'Reward Greedy: %i' % int(ep_reward_greedy),' Reward random: %i' % int(ep_reward_random), 'Explore: %.2f' % var )
            #print(myenv.location)
            # if ep_reward > -300:RENDER = True
            break
    ep_reward = np.reshape(ep_reward/MAX_EP_STEPS, (1,))
    ep_rewardall.append(ep_reward)

    ep_reward_greedy = np.reshape(ep_reward_greedy/MAX_EP_STEPS, (1,))
    ep_rewardall_greedy.append(ep_reward_greedy)

    ep_reward_random = np.reshape(ep_reward_random/MAX_EP_STEPS, (1,))
    ep_rewardall_random.append(ep_reward_random)

#print(s_)
print('Running time: ', time.time() - t1)

print(f"{ep_reward}  ")
print(ep_rewardall)
plt.plot(ep_rewardall, "^-", label='DDPG: rewards')
plt.plot(ep_rewardall_greedy, "+:", label='Greedy: rewards')
plt.plot(ep_rewardall_random, "o--", label='Random: rewards')
plt.xlabel("Episode")
plt.ylabel(" Epsiodic Reward - Average Data Rate (NPCU)")
plt.legend( )
plt.show()

''' Save final results'''
np.savez_compressed('data/data_snapshot1', ep_rewardall=ep_rewardall, ep_rewardall_greedy=ep_rewardall_greedy, ep_rewardall_random=ep_rewardall_random)