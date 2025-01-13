import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

from .enum.request_type import get_headers_by_type
from .node import Capacity


class Nsga2Scheduler:
    def __init__(self, isn, request):
        """
        初始化 NSGA-II 调度器，用于解决模型选择与节点分配的多目标优化问题。
        :param isn
        :param request
        """
        self.isn = isn
        self.request = request

        # 定义优化问题类型，权重 (-1.0, 1.0) 表示第一个目标最小化，第二个目标最大化
        creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.Fitness)

        # 获取模型和节点的数量，用于初始化决策变量范围
        models_count = len(self.isn.models[f"type:{self.request.req_type}"].keys())
        nodes_count = len(self.isn.nodes)

        # 初始化工具箱
        self.toolbox = base.Toolbox()
        # 注册随机整数生成器，用于模型索引和节点索引
        self.toolbox.register("models_size", random.randint, 0, models_count - 1)
        self.toolbox.register("nodes_count", random.randint, 0, nodes_count - 1)

        # 注册个体生成器，每个个体由一个模型索引和一个节点索引组成
        self.toolbox.register(
            "individual", tools.initCycle, creator.Individual, 
            (self.toolbox.models_size, self.toolbox.nodes_count), n=1
        )

        # 注册种群生成器
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 注册遗传算法的操作，包括交叉、变异和选择
        self.toolbox.register("mate", tools.cxTwoPoint)  # 使用单点交叉（或可以用交换交叉）
        self.toolbox.register("mutate", self.mutate_discrete)  # 离散变异操作
        # self.toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
        # self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)  # 变异操作
        self.toolbox.register("select", tools.selNSGA2)  # NSGA-II 选择操作

        # 注册适应度评估函数
        self.toolbox.register("evaluate", self.evaluate)

    def evaluate(self, individual):
        """
        适应度评估函数，计算个体的两个目标值：总延迟（最小化）和模型准确率（最大化）。
        :param individual: 单个个体，包含模型索引和节点索引。
        :return: (latency, model_accuracy) 两个目标值。
        """
        # model_index = max(0, min(len(self.isn.models[f"type:{self.request.req_type}"]) - 1, int(individual[0])))
        # node_index = max(0, min(len(self.isn.nodes) - 1, int(individual[1])))
        model_index = individual[0]
        node_index = individual[1]
        # print(individual[0], individual[1])
        # 获取选择的节点
        node = self.isn.nodes[node_index]

        # 计算传输时延
        transport = self.isn.find_optimal_path(self.request.hostname, node.hostname)  
        transport_latency = transport['total_latency'] * 2 + self.request.datasize / transport['min_bandwidth']
        
        # 获取选择的模型信息
        model_type = f"type:{self.request.req_type}"
        selected_model_name = self.isn.get_model_name(model_type, model_index)
        selected_model_info = self.isn.get_model(model_type, selected_model_name)
        request_capacity = Capacity(self.request.cpu, self.request.memory, self.request.gpu, self.request.vpu)

        # 校验请求资源是否符合节点资源，并调整到适配范围
        if not node.resource.is_larger_than(request_capacity):
            request_capacity = node.resource.scale_down_to_fit(request_capacity)

        # 计算推理时延（基于不同的计算资源类型）
        if request_capacity.vpu != 0:
            inference_latency = (
                self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (node.cpu_gflops * request_capacity.cpu) + 
                self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (node.vpu_gflops * request_capacity.vpu)
            )
        elif request_capacity.gpu != 0:
            inference_latency = (
                self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (node.cpu_gflops * request_capacity.cpu) + 
                self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (node.gpu_gflops * request_capacity.gpu)
            )
        else:
            inference_latency = self.isn.models[model_type][selected_model_name]['flops']  / (node.cpu_gflops * request_capacity.cpu) 

        # 模型精度
        model_accuracy = selected_model_info[get_headers_by_type(self.request.req_type)[0]]  

        # 总延迟包括传输时延和推理时延
        latency = inference_latency + transport_latency

        return latency, model_accuracy

    def mutate_discrete(self, individual):
        """
        离散变异操作，对模型索引和节点索引进行变异。
        :param individual: 个体
        :return: 变异后的个体
        """
        # 对模型索引进行变异
        if random.random() < 0.5:  # 50%的概率对模型索引变异
            individual[0] = random.randint(0, len(self.isn.models[f"type:{self.request.req_type}"]) - 1)

        # 对节点索引进行变异
        if random.random() < 0.5:  # 50%的概率对节点索引变异
            individual[1] = random.randint(0, len(self.isn.nodes) - 1)

        return individual,

    def run(self, population_size=100, ngen=50, cxpb=0.7, mutpb=0.2):
        """
        运行 NSGA-II 遗传算法，优化模型选择与节点分配。
        :param population_size: 种群大小
        :param ngen: 进化代数
        :param cxpb: 交叉概率
        :param mutpb: 变异概率
        :return: 最终种群和帕累托前沿解
        """
        # 初始化种群
        population = self.toolbox.population(n=population_size)

        # 使用 NSGA-II 算法进行优化
        algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=population_size,  # 父代大小
            lambda_=population_size * 2,  # 子代大小
            cxpb=cxpb,  # 交叉概率
            mutpb=mutpb,  # 变异概率
            ngen=ngen,  # 进化代数
            verbose=False
        )

        # 提取 Pareto 前沿（非支配解集合）
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        return population, pareto_front

    def visualize(self, pareto_front):
        """
        可视化帕累托前沿，展示延迟和模型准确率的权衡关系，并在每个点上显示其目标值。
        :param pareto_front: 帕累托前沿解
        """
        f1_values = [ind.fitness.values[0] for ind in pareto_front]  # 延迟（最小化目标）
        f2_values = [ind.fitness.values[1] for ind in pareto_front]  # 模型准确率（最大化目标）

        # 绘制帕累托前沿
        plt.figure(figsize=(10, 6))
        plt.scatter(f1_values, f2_values, color="red", label="Pareto Front", s=20)
        plt.xlabel("Latency (Minimize)")
        plt.ylabel("Model Accuracy (Maximize)")
        plt.title("NSGA-II: Pareto Front")
        plt.grid(True)
        plt.legend()

        # 为每个点添加数值标签
        for i, ind in enumerate(pareto_front):
            latency = ind.fitness.values[0]
            accuracy = ind.fitness.values[1]
            # 使用 annotate 在每个点旁边显示数值
            plt.annotate(f'({latency:.2f}, {accuracy:.2f})', 
                        (f1_values[i], f2_values[i]), 
                        textcoords="offset points", 
                        xytext=(5, 5),  # 设置偏移量
                        ha='center', fontsize=8)

        plt.savefig("nsga2_pareto_front_maxmin.png", dpi=300, bbox_inches="tight")


    def visualize_vars_vs_objectives(self, pareto_front):
        """
        可视化决策变量 (Vars) 与优化目标 (f1_values 和 f2_values) 的关系。
        :param pareto_front: 帕累托前沿解
        """
        # 提取目标值和变量值
        f1_values = [ind.fitness.values[0] for ind in pareto_front]  # 延迟（最小化目标）
        f2_values = [ind.fitness.values[1] for ind in pareto_front]  # 模型准确率（最大化目标）
        vars_model = [ind[0] for ind in pareto_front]  # 模型索引
        vars_node = [ind[1] for ind in pareto_front]  # 节点索引

        # 绘制模型索引 vs 延迟
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # 第1个子图
        plt.scatter(vars_model, f1_values, color="blue", label="Latency vs Model Index")
        for i in range(len(pareto_front)):
            plt.annotate(f'{f1_values[i]:.2f}', (vars_model[i], f1_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')
        plt.xlabel("Model Index")
        plt.ylabel("Latency (Minimize)")
        plt.title("Latency vs Model Index")
        plt.grid(True)
        plt.legend()

        # 绘制节点索引 vs 准确率
        plt.subplot(1, 2, 2)  # 第2个子图
        plt.scatter(vars_node, f2_values, color="green", label="Accuracy vs Node Index")
        for i in range(len(pareto_front)):
            plt.annotate(f'{f2_values[i]:.2f}', (vars_node[i], f2_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')
        plt.xlabel("Node Index")
        plt.ylabel("Model Accuracy (Maximize)")
        plt.title("Accuracy vs Node Index")
        plt.grid(True)
        plt.legend()

        # 显示和保存图像
        plt.tight_layout()
        plt.savefig("vars_vs_objectives.png", dpi=300, bbox_inches="tight")