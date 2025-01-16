from .capacity import Capacity

class Node:
    def __init__(self, hostname, capacity, allocated_capacity=None, allocated_models=None, is_cloud=False, extra=0, cpu_gflops = 0, gpu_gflops = 0):
        """
        初始化 Node 实例，表示一个计算节点。
        
        :param hostname: 节点名称
        :param capacity: 节点的总容量（一个 Capacity 对象）
        :param allocated_capacity: 已分配的容量（一个 Capacity 对象，默认值为 None，若为 None 则初始化为 Capacity(0, 0)）
        :param allocated_models: 已分配的模型配置（字典，默认值为 None，若为 None 则初始化为空字典）
        :param is_cloud: 是否为云端节点，默认值为 False
        :param extra: 额外运行的时间，默认为 0
        :param cpu_flops: 每核 CPU 的算力，默认为 0
        :param gpu_flops: 每核 GPU 的算力，默认为 0
        """
        self.hostname = hostname        # 节点名称
        self.capacity = capacity        # 节点总容量（Capacity 对象）
        self.is_cloud = is_cloud        # 是否为云端节点
        self.cpu_gflops = cpu_gflops    # 每核 CPU 的算力
        self.gpu_gflops = gpu_gflops    # 每核 GPU 的算力
        self.extra = extra              # 额外运行的时间

        # 如果没有传递 allocated_models，则初始化为空字典
        if allocated_models is None:
            allocated_models = {}
        
        self.allocated_models = allocated_models

        # 如果没有传递 allocated_capacity，则初始化为 Capacity(0, 0)
        if allocated_capacity is None:
            allocated_capacity = Capacity(0, 0)
        
        self.allocated_capacity = allocated_capacity  # 已分配的资源
        self.remain_capacity: Capacity = capacity - allocated_capacity  # 计算剩余资源（节点总容量 - 已分配容量）

    def add_model(self, model_name, model_capacity):
        """
        向节点中添加一个新的模型配置。
        
        :param model_name: 模型的名称
        """
        # 添加模型配置到已分配模型字典中
        if self.allocated_models.get(model_name) is None:
            self.allocated_models[model_name] = 1
        else:
            self.allocated_models[model_name] += 1

        self.add_allocated_capacity(model_capacity)

        # print(f"Added model {model_name}, now allocated models: {self.allocated_models}")

    def remove_model(self, model_name):
        """
        从节点中移除一个模型配置。
        
        :param model_name: 模型的名称
        """
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        
        # 从已分配模型字典中移除指定模型
        if self.allocated_models.get(model_name) is not None:
            self.allocated_models[model_name] -= 1
            if self.allocated_models[model_name] == 0:
                del self.allocated_models[model_name]

        print(f"Removed model {model_name}, now allocated models: {self.allocated_models}")

    def modify_flops(self, cpu_gflops = 0, gpu_gflops = 0, vpu_gflops=0):
        """
        修改节点的算力

        :param cpu_flops: 修改后每核 CPU 的算力
        :param gpu_flops: 修改后每核 GPU 的算力
        :param vpu_flops: 修改后每核 VPU 的算力
        """
        if cpu_gflops != 0:
            self.cpu_gflops = cpu_gflops
        if gpu_gflops != 0:
            self.gpu_flops = gpu_gflops

    def add_allocated_capacity(self, new_allocated_capacity):
        """
        添加节点的已分配容量，并更新分配情况（即剩余资源）。
        
        :param new_allocated_capacity: 新的已分配容量（一个 Capacity 对象）
        :raise TypeError: 如果 new_allocated_capacity 不是 Capacity 实例
        :raise ValueError: 如果新分配的容量超过了节点的总容量
        """
        if not isinstance(new_allocated_capacity, Capacity):
            raise TypeError("new_allocated_capacity must be an instance of the 'Capacity' class.")
        
        # 检查新的 allocated_capacity 是否超过了节点的总容量
        if (new_allocated_capacity.cpu > self.remain_capacity.cpu or
            new_allocated_capacity.memory > self.remain_capacity.memory or
            new_allocated_capacity.gpu > self.remain_capacity.gpu):
            raise ValueError("New allocated capacity exceeds the total capacity of the node.")

        # 更新 allocated_capacity 和分配情况
        self.allocated_capacity: Capacity = self.allocated_capacity + new_allocated_capacity
        self.remain_capacity: Capacity = self.remain_capacity - new_allocated_capacity

        # print(f"Updated allocated capacity to: {new_allocated_capacity}")
        # print(f"Remaining resource: {self.remain_capacity}")

    def __repr__(self):
        """
        定义 Node 对象的字符串表示，用于打印时显示节点的基本信息和已分配的模型。
        
        :return: Node 对象的字符串表示
        """
        return f"Node(hostname={self.hostname}, allocated_models={self.allocated_models}, cpu_flops={self.cpu_gflops}, gpu_flops={self.gpu_gflops})"
