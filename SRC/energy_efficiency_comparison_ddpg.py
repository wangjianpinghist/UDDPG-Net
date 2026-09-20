import matplotlib.pyplot as plt
import numpy as np

# 模拟结果数据
pairs = ['1st pair', '2nd pair', '3rd pair', '4th pair', '5th pair']
ddpg_rate = np.array([238341.87, 240756.33, 242729.09, 244397.03, 245841.87])  # bps
ddpg_energy = np.array([0.0050, 0.0063, 0.0075, 0.0088, 0.0100])  # J
ref_rate = np.array([240314.63, 242729.09, 244701.85, 246369.79, 247814.63])  # bps
ref_energy = np.array([0.0060, 0.0075, 0.0090, 0.0105, 0.0120])  # J

# 能量效率 = bps / J = B/J
ddpg_efficiency = ddpg_rate / ddpg_energy
ref_efficiency = ref_rate / ref_energy

# 可视化柱状图
x = np.arange(len(pairs))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, ddpg_efficiency, width, label='DDPG (Proposed)', color='navy')
bars2 = ax.bar(x + width/2, ref_efficiency, width, label='Reference', color='gold')

# 图形美化
ax.set_ylabel('Energy Efficiency (B/J)', fontsize=12)
ax.set_xlabel('Transmitter-receiver pair', fontsize=12)
ax.set_title('Energy Efficiency Comparison (Underwater 20m, 15kHz Bandwidth)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(pairs)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
