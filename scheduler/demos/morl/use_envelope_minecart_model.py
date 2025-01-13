import numpy as np
import torch as th
import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.envelope.envelope import Envelope

def main():
    # 创建推理环境
    eval_env = mo_gym.make("minecart-v0")
    eval_env = MORecordEpisodeStatistics(eval_env, gamma=0.98)

    # 初始化模型
    agent = Envelope(
        eval_env,
        max_grad_norm=0.1,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=64,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=10000,
        learning_starts=100,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,
        tau=1,
        log=False,
    )

    # 加载训练好的模型权重
    agent.load(path="/home/wds/zhitai/graduate/idn/scheduler/demos/morl/saved_models/envelope_minecart_model.tar")

    # 推理过程
    obs, _ = eval_env.reset()
    terminated, truncated = False, False

    # 设置评估权重向量
    w = np.array([0.1, 0.4, 0.2])
    tensor_w = th.tensor(w).float().to(agent.device)

    total_reward = np.zeros_like(w)
    path = [obs]

    while not terminated and not truncated:
        # 使用模型推理选择动作
        action = agent.eval(obs, tensor_w)
        
        # 添加随机动作以探索
        if np.random.random() < 0.1:  # 10% 概率随机选择动作
            action = eval_env.action_space.sample()
        
        print(f"Action: {action}, Obs: {obs}, Weight: {w}")

        prev_obs = obs
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        path.append(obs)

        print(f"Prev Obs: {prev_obs}, Current Obs: {obs}, Reward: {reward}")

    print(f"Total Reward: {total_reward}")

    # 可视化路径
    path = np.array(path)
    positions = path[:, :2]  # 提取 x, y 坐标

    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 1], marker="o", label="Path")
    plt.scatter(0, 0, c="red", label="Base")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Minecart Path")
    plt.legend()
    plt.savefig("/home/wds/zhitai/graduate/idn/scheduler/demos/morl/minecart_path.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
