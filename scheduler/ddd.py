from core.isn import InferenceServiceNet


isn = InferenceServiceNet()
print(isn.nodes)
print(isn.models)
total_models = sum(len(models) for models in isn.models.values())
print(total_models)

model_counts_list = [len(models) for models in isn.models.values()]
print(model_counts_list)

# print(isn.find_optimal_path('edge_1', 'cloud_1'))

# from core.scheduler_env import SchedulingEnv
# from gymnasium.envs.registration import register

# register(
#     id="SchedulingEnv-v0",  # 环境的唯一 ID
#     entry_point="core.scheduler_env.SchedulingEnv",  # 替换为实际模块路径
# )

# from gym import envs

# # Get all environments' specifications
# envids = [spec.id for spec in envs.registry.values()]  # Access values of the registry

# # Print each environment ID
# for envid in envids:
#     print(envid)


