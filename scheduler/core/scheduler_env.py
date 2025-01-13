import gymnasium
from gymnasium import spaces
import numpy as np

from .enum.request_type import get_headers_by_type

from .node import Capacity


class SchedulingEnv(gymnasium.Env):

    def __init__(self, isn, request, lambda_accuracy=1, lambda_latency=-1):
        super(SchedulingEnv, self).__init__()

        self.isn = isn
        self.request = request
        self.lambda_accuracy = lambda_accuracy
        self.lambda_latency = lambda_latency

        # Initialize dimensions for actions
        models_count = len(self.isn.models[f"type:{self.request.req_type}"].keys())
        nodes_count = len(self.isn.nodes)
        assert models_count > 0, "No models available for the given request type."
        assert nodes_count > 0, "No nodes available in the infrastructure."

        # Initialize the mapper for converting between 2D and 1D
        self.mapper = IndexMapper([models_count, nodes_count])

        # Action space: choose a model and a node
        # self.action_space = spaces.MultiDiscrete([models_count, nodes_count])
        self.action_space = spaces.Discrete(self.mapper.total_size)

        # Observation space: resource state of all nodes
        self.observation_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(len(self.isn.nodes) * 4,),  # CPU, memory, GPU, VPU for each node
            dtype=np.float32
        )

        # Define reward space (e.g., [accuracy, latency])
        self.reward_space = spaces.Box(
            low=np.array([0, 0]),  # Min values for accuracy and latency
            high=np.array([1, np.inf]),  # Max values for accuracy (1) and latency (unbounded)
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # 处理随机种子
        super().reset(seed=seed)
        self.np_random, _ = gymnasium.utils.seeding.np_random(seed)

        # 初始化环境状态
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Observation: Concatenation of node resources (CPU, memory, GPU, VPU)
        resources = []

        for node in self.isn.nodes:
            resources.extend([node.capacity.cpu, node.capacity.memory, node.capacity.gpu, node.capacity.vpu])

        self.state = np.array(resources, dtype=np.float32)

        # 如果需要根据 options 修改初始化逻辑，可以在这里处理
        return self.state, {}

    def step(self, action):
        # Decode 1D action into 2D indices (model_index, node_index)
        model_index, node_index = self.mapper.to_2d(action)

        # model_index, node_index = action

        # 选择的 node
        node = self.isn.nodes[node_index]

        # 传输时延
        transport = self.isn.find_optimal_path(self.request.hostname, node.hostname)  
        transport_latency = transport['total_latency'] * 2 + self.request.datasize / transport['min_bandwidth']
        
        # 选择的 model
        model_type = f"type:{self.request.req_type}"
        selected_model_name = self.isn.get_model_name(model_type, model_index)
        selected_model_info = self.isn.get_model(model_type, selected_model_name)
        request_capacity = Capacity(self.request.cpu, self.request.memory, self.request.gpu, self.request.vpu)

        if not node.resource.is_larger_than(request_capacity):
            request_capacity = node.resource.scale_down_to_fit(request_capacity)

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

        # 模型的准确率
        model_accuracy = selected_model_info[get_headers_by_type(self.request.req_type)[0]]  
        latency = inference_latency + transport_latency

        # 奖励
        reward = self.lambda_accuracy * model_accuracy + self.lambda_latency * latency

        terminated = False
        if model_accuracy < self.request.accuracy or latency > (self.request.deadline - self.request.arrivetime):
            terminated = True

        # 一步就行
        truncated = True

        resources = []
        # 更新 observation
        for i, node in enumerate(self.isn.nodes):  # 遍历时获取索引和节点
            if i == node_index:  # 检查是否是目标节点
                resources.extend([
                    max(0, node.resource.cpu - request_capacity.cpu), 
                    max(0, node.resource.memory - request_capacity.memory), 
                    max(0, node.resource.gpu - request_capacity.gpu), 
                    max(0, node.resource.vpu - request_capacity.vpu)
                ])
            else:
                resources.extend([node.resource.cpu, node.resource.memory, node.resource.gpu, node.resource.vpu])


        # Additional info
        info = {
            "selected_node": node.hostname,
            "deployed_model": selected_model_name,
            "accuracy": model_accuracy,
            "latency": latency,
        }

        return self.state, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"State: {self.state}")

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