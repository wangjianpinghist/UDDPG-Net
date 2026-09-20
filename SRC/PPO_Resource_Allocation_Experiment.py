import gym
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from gym.vector.utils import spaces
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

# Disable v2 behavior for tensorflow
tf.disable_v2_behavior()

# 自定义环境
class CustomEnv(gym.Env):
    def __init__(self, K=10):
        super(CustomEnv, self).__init__()
        self.K = K
        self.s_dim = 4 + 2 * self.K  # 状态维度，原始状态 + 用户位置信息
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # 动作空间，假设动作是一个标量
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.s_dim,), dtype=np.float32)  # 观察空间

    def reset(self):
        # 初始化用户位置和其他状态
        user_pos = np.random.randn(self.K, 2)  # 用户位置（K个用户，每个2维坐标）
        other_state = np.random.randn(1, 4)  # 原始状态（4维）
        s = np.concatenate([other_state.flatten(), user_pos.flatten()]).reshape(1, -1)
        return s.flatten()  # 返回一维状态

    def step(self, action):
        # 执行动作并返回下一个状态和奖励
        user_pos = np.random.randn(self.K, 2)
        other_state = np.random.randn(1, 4)
        s = np.concatenate([other_state.flatten(), user_pos.flatten()]).reshape(1, -1)

        # 假设信道增益与用户到原点的距离相关
        distance = np.sqrt(np.sum(user_pos ** 2, axis=1))  # 用户到原点的距离
        channel_gain = 1 / (1 + distance)  # 信道增益（简单模型）

        # 奖励函数：信道增益与动作（功率分配）相关
        reward = np.abs(np.sum(channel_gain * action))

        done = False  # 设置是否终止
        info = {}  # 额外信息
        return s.flatten(), reward, done, info

# 设置训练参数
MAX_EPISODES = 400
MAX_EP_STEPS = 100
K = 10  # 用户数量
ep_rewardall = []  # 记录每个回合的奖励

# 创建自定义环境
def make_env():
    return CustomEnv(K=K)

# 使用 DummyVecEnv 进行环境封装，PPO2 模型需要环境是向量化的
env = DummyVecEnv([make_env])

# 初始化PPO2模型，使用MlpPolicy作为网络结构
ppo_model = PPO2('MlpPolicy', env, verbose=1)

# 训练PPO模型并记录奖励
for i in range(MAX_EPISODES):
    s = env.reset()  # 获取初始状态
    ep_reward = 0

    for j in range(MAX_EP_STEPS):
        action, _ = ppo_model.predict(s)  # 使用PPO模型预测动作
        s_, reward, done, _ = env.step(action)  # 执行动作并得到新的状态和奖励

        # 记录奖励
        ep_reward += reward

        # 更新状态
        s = s_

    # 记录每个回合的奖励
    ep_rewardall.append(ep_reward[0])  # 注意此处是 ep_reward[0]，因为它是一个封装在列表中的值
    print(f"Episode {i + 1}, PPO Reward: {ep_reward[0]:.2f}")

# 绘制奖励曲线
plt.figure(figsize=(12, 5))
plt.plot(ep_rewardall, "^-", label='PPO: rewards')
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.legend()
plt.title("PPO Training Rewards")
plt.show()
