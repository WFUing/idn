import os

import torch


class ModelRegistrar:
    def __init__(self, global_node, deploy_base_path='deployable_models'):
        """
        初始化模型注册器。

        :param global_node: 全局协调节点实例。
        :param deploy_base_path: 存储转换后模型的基础路径。
        """
        self.global_node = global_node
        self.deploy_base_path = deploy_base_path
        if not os.path.exists(self.deploy_base_path):
            os.makedirs(self.deploy_base_path)

        self.models_registry = {}  # 存储模型元数据

    def register_model(self, model_id, model_code=None, model_file=None, task_type=None, accuracy=None, delay=None, model_size=None):
        """
        注册一个AI模型。

        :param model_id: 模型的唯一标识符。
        :param model_code: 模型的代码形式（Python类或函数）。
        :param model_file: 模型的预训练文件路径。
        :param task_type: 模型支持的任务类型。
        :param accuracy: 模型的推理准确性。
        :param delay: 模型的推理延迟。
        :param model_size: 模型的资源消耗（如 GPU 内存）。
        """
        if model_id in self.models_registry:
            print(f"Model {model_id} is already registered.")
            return False

        if model_code:
            # 处理代码形式的模型
            # 这里假设 model_code 是一个可执行的 Python 类定义
            exec(model_code, globals())
            model_class = globals()[model_id]
            model_instance = model_class()
            torch_model = model_instance
            # 转换为TorchScript
            scripted_model = torch.jit.script(torch_model)
            deployable_path = os.path.join(self.deploy_base_path, f"{model_id}.pt")
            scripted_model.save(deployable_path)
            print(f"Model {model_id} registered and converted to TorchScript at {deployable_path}.")
        elif model_file:
            # 处理预训练模型文件
            if not os.path.exists(model_file):
                print(f"Model file {model_file} does not exist.")
                return False
            # 假设模型文件是一个PyTorch模型
            model = torch.load(model_file)
            scripted_model = torch.jit.script(model)
            deployable_path = os.path.join(self.deploy_base_path, f"{model_id}.pt")
            scripted_model.save(deployable_path)
            print(f"Model {model_id} registered and converted to TorchScript at {deployable_path}.")
        else:
            print("Either model_code or model_file must be provided.")
            return False

        # 记录模型元数据
        self.models_registry[model_id] = {
            "task_type": task_type,
            "accuracy": accuracy,
            "delay": delay,
            "model_size": model_size,
            "deployable_path": deployable_path
        }

        return True

    def deploy_model(self, model_id, infer_nodes):
        """
        将模型部署到指定的边缘节点。

        :param model_id: 要部署的模型ID。
        :param infer_nodes: 边缘节点实例的列表。
        """
        if model_id not in self.models_registry:
            print(f"Model {model_id} is not registered.")
            return False

        deployable_path = self.models_registry[model_id]["deployable_path"]

        for node in infer_nodes:
            node.receive_model(model_id, deployable_path)

        print(f"Model {model_id} deployed to all specified InferNodes.")
        return True

    def get_model_info(self, model_id):
        """
        获取模型的元数据信息。

        :param model_id: 模型的唯一标识符。
        :return: 模型元数据字典。
        """
        return self.models_registry.get(model_id, None)
