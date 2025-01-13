import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# 创建优化问题类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # 多目标最小化
creator.create("Individual", list, fitness=creator.FitnessMin)

# 定义目标函数
def evaluate(individual):
    f1 = individual[0] ** 2 + individual[1] ** 2
    f2 = (individual[0] - 1) ** 2 + (individual[1] - 1) ** 2
    return f1, f2  # 返回多个目标

# 初始化种群
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5, 5)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册评估函数
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建种群
population = toolbox.population(n=50)

# 运行遗传算法
final_population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=True)

# 提取最终种群中的个体及其目标函数值
print("Final Population and Fitness:")
for individual in final_population:
    fitness = individual.fitness.values  # 获取目标函数值 (f1, f2)
    print(f"Individual: {individual}, Fitness: {fitness}")

# 提取个体的决策变量和适应度值
x1_values = [ind[0] for ind in final_population]  # 决策变量 x1
x2_values = [ind[1] for ind in final_population]  # 决策变量 x2
f1_values = [ind.fitness.values[0] for ind in final_population]  # 目标函数 f1
f2_values = [ind.fitness.values[1] for ind in final_population]  # 目标函数 f2

# 绘制三维 Pareto 前沿
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 散点图：x1, x2, f1 或 f2
ax.scatter(x1_values, x2_values, f1_values, c='blue', label='f1', marker='o', alpha=0.7)
ax.scatter(x1_values, x2_values, f2_values, c='red', label='f2', marker='^', alpha=0.7)

# 设置轴标签
ax.set_xlabel("x1 (Decision Variable 1)")
ax.set_ylabel("x2 (Decision Variable 2)")
ax.set_zlabel("Fitness (f1 / f2)")

# 图例和标题
ax.legend()
ax.set_title("3D Pareto Front Visualization")

# 保存图片
output_file = "3d_pareto_front.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"3D Pareto front image saved as '{output_file}'")
