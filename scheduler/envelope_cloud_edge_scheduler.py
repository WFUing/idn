import numpy as np
import gymnasium
from core.isn import InferenceServiceNet  
from core.request import IsnRequest  
from morl_baselines.multi_policy.envelope.envelope import Envelope
from gymnasium.envs.registration import register
from morl_baselines.common.weights import equally_spaced_weights
import json
   
# 注册环境
register(
    id="CLoudEdgeEnv-v0",  # 给环境分配唯一 ID
    entry_point="core.cloud_edge_env:CLoudEdgeEnv",  # 模块路径
)

if __name__ == "__main__":
    def make_env():
        # 创建 SchedulingEnv 实例
        isn = InferenceServiceNet()  # 初始化你的 ISN 对象
        # env = SchedulingEnv(isn, request, lambda_accuracy, lambda_latency)
        env = gymnasium.make("CLoudEdgeEnv-v0", isn=isn)  # 使用注册的 ID 创建环境
        # env.spec.id = 'SchedulingEnv-v0'
        # env = MORecordEpisodeStatistics(env, gamma=0.98)
        return env
 
    # 创建环境和评估环境
    env = make_env()
    eval_env = make_env()

    agent = Envelope(
        env=env,
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
        # log=True,
        # project_name="SchedulingEnv-MORL",
        # experiment_name="Envelope-Scheduling",
        log=False,  # 禁用 wandb
        project_name=None,  # 确保项目名称为空
        experiment_name=None,  # 确保实验名称为空
    )

    results = agent.train(
        total_timesteps=100,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=np.array([0, 0]),
        known_pareto_front=None,  # 如果有已知的 Pareto 前沿，可以传递进来
        num_eval_weights_for_front=10,  # 减少用于评估的权重数量
        num_eval_episodes_for_front=5,  # 每个权重的评估 Episode 数
        eval_freq=50,  # 每 50 步评估一次
        reset_num_timesteps=False,
        reset_learning_starts=False,
        verbose=True,
    )

    agent.save(save_dir="./scheduler/saved_models", filename="envelope_model.pt", save_replay_buffer=True)

    print("Training Results:", results)

    

