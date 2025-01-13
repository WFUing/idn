from .model import ModelAllocatedConfig


class Capacity:
    def __init__(self, cpu, memory, gpu=0, vpu=0):
        """
        初始化 Capacity 实例，表示一个节点的资源容量。
        
        :param cpu: CPU 核数
        :param memory: 内存大小
        :param gpu: GPU 核数，默认为 0
        :param vpu: VPU 核数，默认为 0
        """
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
        self.vpu = vpu

    def __sub__(self, other):
        """
        实现减法运算符重载，用于计算两个 Capacity 对象的差异。
        
        :param other: 另一个 Capacity 对象
        :return: 返回两个 Capacity 对象相减后的新 Capacity 对象
        """
        if not isinstance(other, Capacity):
            raise TypeError("Operands must be instances of the 'Capacity' class.")
        
        # 执行减法操作，确保每个属性都能正确相减
        return Capacity(
            self.cpu - other.cpu,
            self.memory - other.memory,
            self.gpu - other.gpu,
            self.vpu - other.vpu
        )

    def scale_down_to_fit(self, other):
        """
        按比例缩小当前 Capacity 对象以适应目标 Capacity 对象。
        
        :param other: 目标 Capacity 对象
        :return: 按比例缩小后的新 Capacity 对象
        """
        if not isinstance(other, Capacity):
            raise TypeError("Operands must be instances of the 'Capacity' class.")

        # 计算每种资源的缩小比例
        ratios = []
        if other.cpu > 0 and self.cpu != 0:
            ratios.append(self.cpu / other.cpu)
        if other.memory > 0 and self.memory != 0:
            ratios.append(self.memory / other.memory)
        if other.gpu > 0 and self.gpu != 0:
            ratios.append(self.gpu / other.gpu)
        if other.vpu > 0 and self.vpu != 0:
            ratios.append(self.vpu / other.vpu)

        # 取最小比例作为缩小系数
        scale_factor = min(ratios) if ratios else 1

        # 按比例缩小
        return Capacity(
            self.cpu * scale_factor,
            self.memory * scale_factor,
            self.gpu * scale_factor,
            self.vpu * scale_factor
        )

    def is_larger_than(self, other):
        """
        比较当前 Capacity 是否大于目标 Capacity 对象。
        
        :param other: 目标 Capacity 对象
        :return: 如果当前 Capacity 大于或等于目标 Capacity 则返回 True，否则返回 False
        """
        if not isinstance(other, Capacity):
            raise TypeError("Operands must be instances of the 'Capacity' class.")

        return (
            self.cpu >= other.cpu and
            self.memory >= other.memory and
            self.gpu >= other.gpu and
            self.vpu >= other.vpu
        )

    def __repr__(self):
        """
        定义 Capacity 对象的字符串表示，用于打印时显示资源的分配情况。
        
        :return: Capacity 对象的字符串表示
        """
        return f"Capacity(cpu={self.cpu}, memory={self.memory}, gpu={self.gpu}, vpu={self.vpu})"


class Node:
    def __init__(self, hostname, capacity, allocated_capacity=None, allocated_models=None, is_cloud=False, extra=0, cpu_gflops = 0, gpu_gflops = 0, vpu_gflops=0):
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
        :param vpu_flops: 每核 VPU 的算力，默认为 0
        """
        self.hostname = hostname     # 节点名称
        self.capacity = capacity     # 节点总容量（Capacity 对象）
        self.is_cloud = is_cloud     # 是否为云端节点
        self.cpu_gflops = cpu_gflops   # 每核 CPU 的算力
        self.gpu_gflops = gpu_gflops   # 每核 GPU 的算力
        self.vpu_gflops = vpu_gflops   # 每核 VPU 的算力
        self.extra = extra           # 额外运行的时间

        # 如果没有传递 allocated_models，则初始化为空字典
        if allocated_models is None:
            allocated_models = {}
        
        self.allocated_models = allocated_models

        # 如果没有传递 allocated_capacity，则初始化为 Capacity(0, 0)
        if allocated_capacity is None:
            allocated_capacity = Capacity(0, 0)
        
        self.allocated_capacity = allocated_capacity  # 已分配的资源
        self.resource = capacity - allocated_capacity  # 计算剩余资源（节点总容量 - 已分配容量）

    def add_model(self, model_name, model_config):
        """
        向节点中添加一个新的模型配置。
        
        :param model_name: 模型的名称
        :param model_config: ModelConfig 对象，包含模型的配置
        :raise TypeError: 如果 model_config 不是 ModelConfig 实例
        """
        if not isinstance(model_config, ModelAllocatedConfig):
            raise TypeError("model_config must be an instance of ModelConfig.")
        
        # 添加模型配置到已分配模型字典中
        self.allocated_models[model_name] = model_config
        print(f"Added model {model_name} with config: {model_config}")

    def modify_model(self, model_name, pod_name, cpu=None, memory=None, gpu=None, vpu=None):
        """
        修改已存在的模型配置。
        
        :param model_name: 要修改的模型名称
        :param pod_name: pod 名称
        :param cpu: 修改后的 CPU 数量（可选）
        :param memory: 修改后的内存大小（可选）
        :param gpu: 修改后的 GPU 数量（可选）
        :param vpu: 修改后的 VPU 数量（可选）
        :raise KeyError: 如果指定的模型名称不存在
        """
        if model_name not in self.allocated_models:
            raise KeyError(f"Model {model_name} not found in the node's allocated models.")
        
        model_config = self.allocated_models[model_name]
        
        # 调用 ModelConfig 中的 modify_config 方法修改模型配置
        model_config.modify_config(pod_name, cpu, memory, gpu, vpu)


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
        if vpu_gflops != 0:
            self.vpu_gflops = vpu_gflops

    def modify_allocated_capacity(self, new_allocated_capacity):
        """
        修改节点的已分配容量，并更新分配情况（即剩余资源）。
        
        :param new_allocated_capacity: 新的已分配容量（一个 Capacity 对象）
        :raise TypeError: 如果 new_allocated_capacity 不是 Capacity 实例
        :raise ValueError: 如果新分配的容量超过了节点的总容量
        """
        if not isinstance(new_allocated_capacity, Capacity):
            raise TypeError("new_allocated_capacity must be an instance of the 'Capacity' class.")
        
        # 检查新的 allocated_capacity 是否超过了节点的总容量
        if (new_allocated_capacity.cpu > self.capacity.cpu or
            new_allocated_capacity.memory > self.capacity.memory or
            new_allocated_capacity.gpu > self.capacity.gpu or
            new_allocated_capacity.vpu > self.capacity.vpu):
            raise ValueError("New allocated capacity exceeds the total capacity of the node.")

        # 更新 allocated_capacity 和分配情况
        self.allocated_capacity = new_allocated_capacity
        self.resource = self.capacity - self.allocated_capacity

        print(f"Updated allocated capacity to: {new_allocated_capacity}")
        print(f"Remaining resource: {self.resource}")

    def __repr__(self):
        """
        定义 Node 对象的字符串表示，用于打印时显示节点的基本信息和已分配的模型。
        
        :return: Node 对象的字符串表示
        """
        return f"Node(hostname={self.hostname}, allocated_models={self.allocated_models}, cpu_flops={self.cpu_gflops}, gpu_flops={self.gpu_gflops}, vpu_flops={self.vpu_gflops})"
