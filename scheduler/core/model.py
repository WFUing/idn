class Model:
    def __init__(self, type, model_name, model_accuracy, model_gflops):
        """
        初始化 Model 对象，表示一个机器学习模型。
        
        :param type: 模型类型（可以是分类、回归等）
        :param model_name: 模型名称
        :param model_accuracy: 模型的准确率
        :param model_gflops: 模型的算量
        """
        self.type = type                # 模型类型（例如：分类、回归）
        self.name = model_name          # 模型名称
        self.accuracy = model_accuracy  # 模型准确率
        self.gflops = model_gflops        # 模型算量


class ModelConfig:
    def __init__(self, pod_name, cpu, memory, gpu=0, vpu=0):
        """
        初始化 ResourceConfig 对象，表示一个 Pod 的资源配置。
        
        :param pod_name: Pod 名称，用于区分不同的配置
        :param cpu: Pod 配置的 CPU 数量
        :param memory: Pod 配置的内存大小
        :param gpu: Pod 配置的 GPU 数量，默认为 0
        :param vpu: Pod 配置的 VPU 数量，默认为 0
        """
        self.pod_name = pod_name  # Pod 名称，用于区分不同 Pod 的资源配置
        self.cpu = cpu            # CPU 配置（例如：4）
        self.memory = memory      # 内存配置（例如：8GB）
        self.gpu = gpu            # GPU 配置（例如：2）
        self.vpu = vpu            # VPU 配置（例如：0）
        self.latency = 0

    def predict_latency(self):
        return 0
    
    def __repr__(self):
        """
        自定义字符串表示，方便打印 ResourceConfig 对象内容
        
        :return: 返回 ResourceConfig 对象的字符串表示
        """
        return f"ResourceConfig(pod_name={self.pod_name}, cpu={self.cpu}, memory={self.memory}, gpu={self.gpu}, vpu={self.vpu})"


class ModelAllocatedConfig:
    def __init__(self, model_name, config=None, flops=0):
        """
        初始化 ModelConfig 对象，表示与模型相关的多个 Pod 配置。
        
        :param model_name: 模型名称，用于区分不同的模型配置
        :param config: 一个字典，存储不同 Pod 的配置（默认值为 None，初始化为空字典）
        """
        self.model_name = model_name  # 模型名称（例如："ResNet50"）
        self.flops = flops

        if config is None:
            config = {}  # 如果没有提供配置，则初始化为空字典
        self.config = config  # 存储与模型相关的多个 Pod 配置（字典形式，key 为 pod_name，value 为 ResourceConfig）

    def add_config(self, pod_name, resource_config):
        """
        向 ModelConfig 中添加新的 Pod 配置。
        
        :param pod_name: Pod 的名称（例如："pod1"）
        :param resource_config: 对应的 ResourceConfig 对象，包含 Pod 的资源配置
        :raise TypeError: 如果 resource_config 不是 ResourceConfig 实例
        """
        # 确保传入的 resource_config 是 RecourceConfig 类型
        if not isinstance(resource_config, ModelConfig):
            raise TypeError("The resource_config must be an instance of RecourceConfig.")
        
        # 将资源配置添加到字典中，key 为 pod_name，value 为 ResourceConfig 对象
        self.config[pod_name] = resource_config
        print(f"Added new ResourceConfig for pod {pod_name}: {resource_config}")

    def modify_config(self, pod_name, cpu=None, memory=None, gpu=None, vpu=None):
        """
        修改已存在的 Pod 配置，更新其资源设置。
        
        :param pod_name: 要修改的 Pod 名称
        :param cpu: 修改后的 CPU 数量（如果不传则不修改）
        :param memory: 修改后的内存大小（如果不传则不修改）
        :param gpu: 修改后的 GPU 数量（如果不传则不修改）
        :param vpu: 修改后的 VPU 数量（如果不传则不修改）
        :raise KeyError: 如果 pod_name 不存在
        """
        # 检查 pod_name 是否存在于配置中
        if pod_name not in self.config:
            raise KeyError(f"Pod with name {pod_name} does not exist.")
        
        # 获取现有的配置对象
        existing_config = self.config[pod_name]
        
        # 如果传入了新的参数，则更新现有的配置
        if cpu is not None:
            existing_config.cpu = cpu
        if memory is not None:
            existing_config.memory = memory
        if gpu is not None:
            existing_config.gpu = gpu
        if vpu is not None:
            existing_config.vpu = vpu
        
        # 打印更新后的配置
        print(f"Updated ResourceConfig for {pod_name}: {existing_config}")

    def __repr__(self):
        """
        自定义字符串表示，方便打印 ModelConfig 对象内容
        
        :return: 返回 ModelConfig 对象的字符串表示
        """
        return f"ModelConfig(model_name={self.model_name}, config={self.config})"
