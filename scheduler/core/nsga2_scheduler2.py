import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

from .enum.request_type import get_headers_by_type
from .node import Capacity


class Nsga2Scheduler2:
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
        self.models_count = len(self.isn.models[f"type:{self.request.req_type}"].keys())
        self.nodes_count = len(self.isn.nodes)

        # 初始化工具箱
        self.toolbox = base.Toolbox()
        # 注册随机整数生成器，用于模型索引和节点索引
        self.toolbox.register("request_size", random.randint, 0, request.datasize)

        # 注册个体生成器，每个个体由一个模型索引和一个节点索引组成
        self.toolbox.register(
            "individual", self.custom_individual_init, creator.Individual,
            request.datasize, self.models_count * self.nodes_count
        )

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 注册遗传算法的操作，包括交叉、变异和选择
        self.toolbox.register("mate", tools.cxTwoPoint)  # 使用单点交叉（或可以用交换交叉）
        self.toolbox.register("mutate", self.mutate_discrete)  # 离散变异操作
        # self.toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
        # self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)  # 变异操作
        self.toolbox.register("select", tools.selNSGA2)  # NSGA-II 选择操作

        # 注册适应度评估函数
        self.toolbox.register("evaluate", self.evaluate)

    # 定义自定义初始化函数
    def custom_individual_init(self, container, num_tasks, num_positions):
        """
        自定义初始化函数，用于生成种群，按照伪代码逻辑初始化个体。

        :param container: 用于存储种群的容器（如 list）。
        :param num_tasks: 剩余需要完成的子任务数量。
        :param num_positions: 可选位置的数量 (m + 1)。
        :return: 初始化的种群。
        """
        individual = [0] * (self.models_count * self.nodes_count)

        # 初始化个体中的基因点
        for _ in range(num_tasks):
            k = random.randint(0, num_positions - 1)  # 随机选择位置 k
            individual[k] += 1  # 对选定位置的基因点初始化

        return container(individual)

    # 定义自定义交叉操作
    def custom_crossover(self, parents, num_positions, segment_length):
        """
        自定义交叉操作，用于交换父代和母代的基因片段。

        :param parents: 父代和母代个体列表，长度为 2。
        :param num_positions: 可选位置数量 (m + 1)。
        :param segment_length: 要交换的基因片段长度。
        :return: 子代个体列表。
        """
        cross_point = random.randint(0, num_positions - segment_length)  # 随机产生交叉点

        # 交换父母代的片段
        for i in range(segment_length):
            parents[0][cross_point + i], parents[1][cross_point + i] = (
                parents[1][cross_point + i],
                parents[0][cross_point + i],
            )

        children = []

        def strengthen(individual):
            """将弱合法个体转化为强合法。"""
            while sum(individual) > num_positions:
                idx = random.randint(0, len(individual) - 1)  # 随机选择一个位置
                if individual[idx] > 0:
                    individual[idx] -= 1  # 减少该位置的值，确保值不小于 0
            return individual

        for child in parents:
            if sum(child) < num_positions:
                # 丢弃非法子代
                continue
            elif sum(child) > num_positions:
                # 强化弱合法子代
                children.append(strengthen(child))
            else:
                # 保留合法子代
                children.append(child)

        return children

    def evaluate(self, individual):

        # print(individual)
        # print("len:",len(individual))
        # print(self.models_count * self.nodes_count)
        # 计算总长度验证是否符合模型结构
        assert len(
            individual) == self.models_count * self.nodes_count, "Individual size does not match model and node dimensions"

        model_accuracy = 0.0

        latency = 0.0

        # 遍历 individual，计算 model_id 和 node_id
        for i, value in enumerate(individual):
            model_id = i // self.nodes_count  # 计算 model_id，行号，从1开始
            node_id = i % self.nodes_count  # 计算 node_id，列号，从1开始
            a, b = self.evaluate_model_node(model_id, node_id, value)
            model_accuracy += a
            latency += b

        return model_accuracy, latency

    def evaluate_model_node(self, model_index, node_index, request_size):
        """
        适应度评估函数，计算个体的两个目标值：总延迟（最小化）和模型准确率（最大化）。
        :param individual: 单个个体，包含模型索引和节点索引。
        :return: (latency, model_accuracy) 两个目标值。
        """
        # 获取选择的节点
        node = self.isn.nodes[node_index]

        # 计算传输时延
        transport = self.isn.find_optimal_path(self.request.hostname, node.hostname)
        transport_latency = transport['total_latency'] * 2 + request_size * 512 * 512 / transport['min_bandwidth']

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
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (
                    node.cpu_gflops * request_capacity.cpu) +
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (
                            node.vpu_gflops * request_capacity.vpu)
            )
        elif request_capacity.gpu != 0:
            inference_latency = (
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (
                    node.cpu_gflops * request_capacity.cpu) +
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (
                            node.gpu_gflops * request_capacity.gpu)
            )
        else:
            inference_latency = self.isn.models[model_type][selected_model_name]['flops'] / (
                    node.cpu_gflops * request_capacity.cpu)

            # 模型精度
        model_accuracy = selected_model_info[get_headers_by_type(self.request.req_type)[0]]

        # 总延迟包括传输时延和推理时延
        latency = request_size * inference_latency + transport_latency

        return latency, model_accuracy

    def mutate_discrete(self, individual):
        """
        对染色体 individual 进行离散变异操作。

        :param individual: list[int] 表示染色体，每个位置的值表示任务分布
        :return: new_pop_i 新的变异后染色体
        """
        pop_i = individual.copy()  # 复制个体避免直接修改原始数据
        l = len(pop_i)  # 染色体长度

        # k 定义为一个控制变异次数的变量，可以根据需要设置，比如固定次数或动态确定
        k = random.randint(1, l // 2)  # 示例随机变异次数，范围为 [1, l//2]

        while k > 0:
            # 随机引入负任务位置
            p = random.randint(0, l - 1)
            while pop_i[p] <= 0:  # 确保选中非负任务的位置
                p = random.randint(0, l - 1)
            pop_i[p] -= 1  # 引入负任务

            # 随机引入正任务位置
            p = random.randint(0, l - 1)
            pop_i[p] += 1  # 引入正任务

            k -= 1  # 递减变异次数

        mutated_individual = type(individual)(pop_i)  # 保留 Individual 类型
        return mutated_individual,

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

        # 打开文件记录迭代过程
        with open("iteration_results.txt", "w") as file:
            file.write("Generation,Best_Fitness_1,Best_Fitness_2\n")

            # 使用 NSGA-II 算法进行优化
            for gen in range(ngen):
                algorithms.eaMuPlusLambda(
                    population,
                    self.toolbox,
                    mu=population_size,  # 父代大小
                    lambda_=population_size * 2,  # 子代大小
                    cxpb=cxpb,  # 交叉概率
                    mutpb=mutpb,  # 变异概率
                    ngen=1,  # 每次执行一代
                    verbose=True
                )

                # 提取当前代的最佳个体
                best_ind = tools.selBest(population, 1)[0]
                best_fitness = best_ind.fitness.values

                # 记录当前代结果
                file.write(f"{gen},{best_fitness[0]},{best_fitness[1]}\n")

        file.close()
        # 提取 Pareto 前沿（非支配解集合）
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        return population, pareto_front

