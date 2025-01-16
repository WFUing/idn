from .capacity import Capacity

class IsnModel:
    def __init__(self, model_name, model_accuracy, model_gflops, cpu, memory, gpu=0):
        """
        初始化 Model 对象，表示一个机器学习模型。
        
        :param type: 模型类型（可以是分类、回归等）
        :param model_name: 模型名称
        :param model_accuracy: 模型的准确率
        :param model_gflops: 模型的算量
        """                         
        self.name = model_name                      # 模型名称
        self.accuracy = model_accuracy              # 模型准确率
        self.gflops = model_gflops                  # 模型算量
        self.capacity = Capacity(cpu, memory, gpu)  # 模型所需资源配置


