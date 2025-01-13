# __init__.py

from gymnasium.envs.registration import register

register(
    id="SchedulingEnv-v0",  # 环境的唯一 ID
    entry_point="scheduler_env.SchedulingEnv",  # 替换为实际模块路径
)