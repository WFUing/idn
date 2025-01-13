from gym.envs.registration import register

register(
    id="SchedulingEnv-v0",
    entry_point="..core.scheduler_env/:SchedulingEnv"
)