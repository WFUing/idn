from idn.utils.local_utils import LocalUtil


class InferNode:
    def __init__(self, node_id, models, task_types, max_capacity, network_latency, network_bandwidth, accuracies, delays):
        """初始化局部节点"""
        self.node_id = node_id
        self.models = models  # 节点上的模型
        self.task_types = task_types  # 支持的任务类型
        self.max_capacity = max_capacity  # 每种任务的最大服务容量
        self.current_load = [0] * len(max_capacity)  # 当前负载
        self.available_capacity = max_capacity.copy()  # 初始化可用容量
        self.network_latency = network_latency  # 网络延迟
        self.network_bandwidth = network_bandwidth  # 节点带宽
        self.accuracies = accuracies  # 每个模型的准确性
        self.delays = delays  # 每个模型的推理延迟
        self.cpu_usage = 0.0  # 当前CPU使用率
        self.gpu_usage = 0.0  # 当前GPU使用率
        self.memory_usage = 0.0  # 当前内存使用情况

    def assign_model(self, model_id):
        """尝试将模型分配到节点，并检查资源约束"""
        if self.models[model_id] == 1:
            print(f"Model {self.models[model_id]} is already assigned to Node {self.node_id}.")
            return False

        # 计算当前已使用的资源
        current_usage = sum(self.model_assignment[i] * self.model_sizes[i] for i in range(len(self.models)))

        # 如果新模型的资源需求不会超过预算，则分配模型
        if current_usage + self.model_sizes[model_id] <= self.resource_budget:
            self.model_assignment[model_id] = 1  # 分配模型
            print(f"Model {self.models[model_id]} assigned to Node {self.node_id}.")
            return True
        else:
            print(f"Cannot assign model {self.models[model_id]} to Node {self.node_id}. Not enough resources.")
            return False

    def get_available_capacity(self, task_type):
        """获取某种任务类型的可用容量"""
        if task_type in self.task_types:
            index = self.task_types.index(task_type)
            return self.available_capacity[index]
        return 0

    def update_load(self, load):
        """更新节点的负载和可用容量"""
        self.current_load = load
        self.available_capacity = [
            max_cap - cur_load for max_cap, cur_load in zip(self.max_capacity, self.current_load)
        ]
        # 更新节点状态
        if all(cap <= 0 for cap in self.available_capacity):
            self.status = "Overloaded"
        else:
            self.status = "Normal"

    def get_inference_delay(self, task_type):
        """获取模型的推理延迟"""
        if task_type in self.task_types:
            index = self.task_types.index(task_type)
            return self.delays[index]
        return float('inf')

    def get_accuracy(self, task_type):
        """获取模型的推理准确性"""
        if task_type in self.task_types:
            index = self.task_types.index(task_type)
            return self.accuracies[index]
        return 0.0

    def update_load_for_task(self, task_type, load):
        """更新节点的负载和可用容量"""
        if task_type in self.task_types:
            index = self.task_types.index(task_type)
            self.current_load[index] += load
            self.available_capacity[index] = self.max_capacity[index] - self.current_load[index]
            # 更新资源使用情况

    def update_resource_usage(self):
        """更新节点的资源使用情况"""
        self.cpu_usage = LocalUtil.get_cpu_usage()
        self.memory_usage = LocalUtil.get_memory_usage()
        self.gpu_usage = LocalUtil.get_gpu_usage()

        # 网络延时监测
        self.network_latency = LocalUtil.ping_latency('8.8.8.8')  # 使用 Google 的 DNS 服务器来测试延迟

        # 网络带宽监测
        self.network_bandwidth = LocalUtil.get_network_bandwidth()

    def get_status(self):
        """返回节点的当前状态"""
        return {
            "Node ID": self.node_id,
            "Models": self.models,
            "Task Types": self.task_types,
            "Max Capacity": self.max_capacity,
            "Current Load": self.current_load,
            "Available Capacity": self.available_capacity,
            "Latency": self.network_latency,
            "Bandwidth": self.network_bandwidth,
            "Accuracies": self.accuracies,
            "Delays": self.delays
        }

    def print_status(self):
        """打印节点的当前状态"""
        status = self.get_status()
        for key, value in status.items():
            print(f"{key}: {value}")
        print("-" * 40)