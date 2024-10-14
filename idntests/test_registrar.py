from idn.node.coord_node import CoordNode
from idn.node.model_registrar import ModelRegistrar

if __name__ == "__main__":
    coord_node = CoordNode()

    # 添加节点
    coord_node.add_node("Node_1", "Edge_1", ["ResNet50", "MobileNet"], ["Classification", "Detection"], [100, 50], 10, 100)
    coord_node.add_node("Node_2", "Cloud", ["BERT", "GPT-2"], ["NLP"], [200, 150], 50, 1000)

    # 更新节点负载
    coord_node.update_node("Node_1", [70, 30])  # 更新 Node_1 的负载

    # 获取最优节点
    coord_node.get_optimal_node("Classification")

    # 打印所有节点信息
    coord_node.print_node_info()

    model_registrar = ModelRegistrar(global_node=coord_node)

    # 定义一个简单的PyTorch模型类的代码形式
    model_code = """
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1)
            self.fc1 = nn.Linear(32*6*6, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, 32*6*6)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    """

    # 注册模型
    model_registrar.register_model(
        model_id="SimpleCNN",
        model_code=model_code,
        task_type="Classification",
        accuracy=0.85,
        delay=20,
        model_size=250  # 假设模型消耗 250 MB GPU 内存
    )
