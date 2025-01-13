import time

import gymnasium
from gymnasium import spaces
import numpy as np
from .enum.request_type import get_headers_by_type
from .node import Capacity
from .request import IsnRequest


class CLoudEdgeEnv(gymnasium.Env):

    def __init__(self, isn):
        """
        初始化 CLoudEdgeEnv 环境。
        :param isn
        """
        super(CLoudEdgeEnv, self).__init__()

        # 将基础设施网络赋值给实例变量
        self.observation_matrix = None
        self.current_request = None
        self.current_request_index = None
        self.requests = None
        self.isn = isn
        self.latency = None
        self.accuracy = None

        # 初始化动作维度（模型数量和节点数量）
        self.model_counts_list = [len(models) for models in isn.models.values()]  # 获取每种类型模型的数量
        models_count = sum(self.model_counts_list)  # 计算所有模型的总数量
        nodes_count = len(self.isn.nodes)  # 获取基础设施中的节点数量

        # 确保环境中至少有一个模型和一个节点
        assert models_count > 0, "没有可用的模型，请检查请求类型是否正确。"
        assert nodes_count > 0, "没有可用的节点，请检查基础设施网络。"

        # 初始化索引映射器，用于在 1D 和 2D 索引之间转换
        self.mapper = IndexMapper([models_count, nodes_count])

        # 动作空间：选择一个模型和一个节点
        # 将动作空间定义为离散空间，大小为模型数量 × 节点数量
        # self.action_space = spaces.MultiDiscrete([models_count, nodes_count])  # 多离散空间（备用方式）
        self.action_space = spaces.Discrete(self.mapper.total_size)  # 单一离散空间

        # 观察空间：所有节点的资源状态
        self.observation_space = spaces.Box(
            low=0,  # 最小值为 0（资源不能为负）
            high=np.inf,  # 最大值为无穷大（假设资源上限未预定义）
            shape=(len(self.isn.nodes) * 4, ),  # 每个节点包含 7 个资源指标（CPU、内存、GPU、VPU 及其 FLOPS）
            dtype=np.float32  # 数据类型为浮点数
        )

        # 奖励空间：定义准确率和时延的范围
        self.reward_space = spaces.Box(
            low=np.array([[0], [0]]),  # 修改为 (2, 1)
            high=np.array([[1], [np.inf]]),  # 修改为 (2, 1)
            shape=(2, 1),  # 奖励空间的形状
            dtype=np.float32  # 数据类型
        )

    import random
    import time

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态，包括初始化 observation_space 和请求列表。
        :param seed: 可选的随机种子，用于保证结果的可重复性。
        :param options: 额外的选项参数（此处可传入 requests）。
        :return: 初始化后的 observation_space 矩阵，表示环境的初始状态。
        """
        # 如果提供了随机种子，则设置随机数生成器的种子，保证结果可重复
        if seed is not None:
            np.random.seed(seed)

        # 如果 options 中没有 requests，则随机生成请求
        if options and "requests" in options:
            requests = options["requests"]
        else:
            num_requests = np.random.randint(1, 10)  # 随机生成 1 到 10 个请求
            current_time = int(time.time() * 1000)  # 当前时间戳（毫秒）
            requests = [
                IsnRequest(
                    req_type=np.random.randint(1, len(self.isn.models)),  # 随机请求类型
                    arrivetime=current_time,  # 当前时间作为到达时间
                    deadline=current_time + np.random.randint(1000, 5000),  # 随机生成截止时间（1 到 5 秒后）
                    accuracy=np.random.uniform(0.5, 1.0),  # 随机生成期望准确率（70% 到 100%）
                    hostname=f"Host_{i}",  # 主机名以 "Host_i" 命名
                    datasize=np.random.uniform(1, 10),  # 随机生成数据大小
                    cpu=np.random.randint(1, 4),  # 随机生成 CPU 需求（1 到 4 核）
                    gpu=np.random.randint(0, 2),  # 随机生成 GPU 需求（0 到 2 核）
                    vpu=np.random.randint(0, 2),  # 随机生成 VPU 需求（0 到 2 核）
                    memory=np.random.randint(1, 16)  # 随机生成内存需求（1 到 16 GB）
                ) for i in range(num_requests)
            ]

        # 将请求存储到环境变量中
        self.requests = requests

        # 当前请求索引
        self.current_request_index = 0

        # 初始化 observation_space 矩阵
        observation_matrix = []
        for node in self.isn.nodes:  # 遍历节点列表，假设每个节点是一个 Node 对象
            # 为每个节点生成随机资源使用值，确保不超过节点容量
            observation_matrix.extend([
                np.random.uniform(0, node.capacity.cpu),  # 随机生成 CPU 使用量
                np.random.uniform(0, node.capacity.memory),  # 随机生成内存使用量
                np.random.uniform(0, node.capacity.gpu),  # 随机生成 GPU 使用量
                np.random.uniform(0, node.capacity.vpu)  # 随机生成 VPU 使用量
            ])

        # 将观察矩阵转换为 NumPy 数组
        self.observation_matrix = np.array(observation_matrix, dtype=np.float32)

        self.latency = 0.0
        self.accuracy = 0.0
        # 返回观察矩阵，作为环境的初始状态
        return self.observation_matrix, {}

    def step(self, action):
        """
        执行一步动作，更新环境状态并计算奖励。
        :param action: 动作，表示选择的模型和节点。
        :return: 新的状态、奖励、是否结束、是否截断和额外信息。
        """
        # 将 1D 动作解码为 2D 索引 (model_index, node_index)
        model_index, node_index = self.mapper.to_2d(action)
        model_type, model_index = self.get_model_type(model_index)

        if self.current_request_index >= len(self.requests):
            return self.observation_matrix, [self.accuracy, self.latency], True, False, {}

        # 当前的请求
        current_request = self.requests[self.current_request_index]
        self.current_request_index += 1
        if current_request.req_type != model_type:
            return self.observation_matrix, [self.accuracy, self.latency], False, False, {}

        # 选择的节点
        node = self.isn.nodes[node_index]
        # 计算传输时延
        transport = self.isn.find_optimal_path(current_request.hostname, node.hostname)  # 查找最优路径
        transport_latency = transport['total_latency'] * 2 + current_request.datasize / transport[
            'min_bandwidth']  # 计算总传输时延

        # 选择的模型
        model_type = f"type:{current_request.req_type}"  # 根据请求类型获取模型类别
        selected_model_name = self.isn.get_model_name(model_type, model_index)  # 获取选定模型的名称
        selected_model_info = self.isn.get_model(model_type, selected_model_name)  # 获取选定模型的详细信息

        # 请求的资源需求
        request_capacity = Capacity(current_request.cpu, current_request.memory, current_request.gpu, current_request.vpu)

        # 检查节点资源是否足够，若不足则调整请求的资源需求
        if not node.resource.is_larger_than(request_capacity):
            request_capacity = node.resource.scale_down_to_fit(request_capacity)

        # 推理时延计算
        if request_capacity.vpu != 0:
            # 如果请求使用 VPU，则计算 CPU 和 VPU 的混合推理时延
            inference_latency = (
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (
                        node.cpu_gflops * request_capacity.cpu) +
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (
                                node.vpu_gflops * request_capacity.vpu)
            )
        elif request_capacity.gpu != 0:
            # 如果请求使用 GPU，则计算 CPU 和 GPU 的混合推理时延
            inference_latency = (
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (
                        node.cpu_gflops * request_capacity.cpu) +
                    self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (
                                node.gpu_gflops * request_capacity.gpu)
            )
        else:
            # 如果只使用 CPU，则直接计算 CPU 的推理时延
            inference_latency = self.isn.models[model_type][selected_model_name]['flops'] / (
                        node.cpu_gflops * request_capacity.cpu)

        # 模型的准确率
        model_accuracy = selected_model_info[get_headers_by_type(current_request.req_type)[0]]  # 根据请求类型获取模型准确率
        latency = inference_latency + transport_latency  # 总时延 = 推理时延 + 传输时延

        self.accuracy = self.accuracy + model_accuracy
        self.latency = self.latency + latency
        # 计算奖励
        reward = [self.accuracy, self.latency]

        if model_accuracy >= current_request.accuracy and latency <= (current_request.deadline - current_request.arrivetime):
            self.accuracy = self.accuracy + model_accuracy
            self.latency = self.latency + latency
            # 计算奖励
            reward = [self.accuracy, self.latency]

        # 一步决策后直接截断
        truncated = False
        # 判断是否终止
        terminated = False

        # 更新观察空间
        new_node_data = [request_capacity.cpu, request_capacity.memory, request_capacity.gpu, request_capacity.vpu]
        start_index = node_index * 4
        end_index = start_index + 4
        self.observation_matrix[start_index:end_index] = new_node_data

        # 附加信息
        info = {
            "selected_node": node.hostname,  # 选择的节点名称
            "deployed_model": selected_model_name,  # 部署的模型名称
            "accuracy": model_accuracy,  # 模型的准确率
            "latency": latency,  # 总时延
        }

        # 返回状态、奖励、是否终止、是否截断和附加信息
        return self.observation_matrix, reward, terminated, truncated, info

    def get_model_type(self, model_id):
        """
        根据模型编号获取模型的类型。

        :param model_id: int，模型编号，从 0 开始编号。
        :return: tuple，包含模型类型编号（从 1 开始计数）和模型在该类型中的偏移索引。
        """
        # 检查模型编号是否为非负数
        assert model_id >= 0, "模型编号必须为非负数"

        # 累积计数，用于确定模型属于哪个类型
        cumulative_count = 0

        # 遍历每种模型类型及其数量
        for model_type, count in enumerate(self.model_counts_list, start=1):
            cumulative_count += count  # 更新累积模型数量
            if model_id < cumulative_count:  # 如果模型编号小于累积数量
                return model_type, model_id - (cumulative_count - count)  # 返回模型类型和偏移索引

        # 如果模型编号超过所有模型总数，则抛出异常
        raise ValueError("模型编号超出了模型总数")


class IndexMapper:
    def __init__(self, dimensions):
        """
        初始化映射器。

        :param dimensions: 一个列表，例如 [models_count, nodes_count]
        """
        self.dimensions = dimensions
        self.total_size = np.prod(dimensions)  # 总的一维索引数量

    def to_1d(self, indices):
        """
        将二维索引映射到一维索引。

        :param indices: 一个列表 [i, j] 表示在二维空间中的索引
        :return: 对应的一维索引
        """
        assert len(indices) == len(self.dimensions), "Indices dimensions must match"
        flat_index = 0
        for i, dim in zip(indices, self.dimensions):
            assert 0 <= i < dim, f"Index {i} is out of bounds for dimension {dim}"
            flat_index = flat_index * dim + i
        return flat_index

    def to_2d(self, flat_index):
        """
        将一维索引解码回二维索引。

        :param flat_index: 一维索引
        :return: 对应的二维索引 [i, j]
        """
        assert 0 <= flat_index < self.total_size, "Flat index is out of bounds"
        indices = []
        for dim in reversed(self.dimensions):
            indices.append(flat_index % dim)
            flat_index //= dim
        return list(reversed(indices))