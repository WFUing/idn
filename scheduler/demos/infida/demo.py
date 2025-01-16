import numpy as np

class INFIDA:
    def __init__(self, num_nodes, num_models, learning_rate, budgets, model_sizes):
        """
        初始化 INFIDA 算法

        参数:
        - num_nodes: 节点数量
        - num_models: 模型数量
        - learning_rate: 学习率 \(\eta\)
        - budgets: 每个节点的资源预算 (list)
        - model_sizes: 每个模型在节点的资源占用 (2D array)
        """
        self.num_nodes = num_nodes
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.budgets = budgets
        self.model_sizes = model_sizes

        # 初始化分数分配 (fractional allocation)
        self.fractional_allocation = np.random.rand(num_nodes, num_models)
        self.fractional_allocation /= np.sum(self.fractional_allocation, axis=1, keepdims=True)

    def compute_gradient(self, requests, costs):
        """
        计算梯度

        参数:
        - requests: 请求矩阵，每行对应任务请求 (num_tasks, num_nodes)
        - costs: 模型的服务成本矩阵 (num_nodes, num_models)

        返回:
        - 梯度 (2D array)
        """
        gradients = np.zeros_like(self.fractional_allocation)
        for node in range(self.num_nodes):
            for model in range(self.num_models):
                grad = 0
                for request in requests:
                    latency, accuracy = costs[node, model]
                    marginal_cost = latency - accuracy
                    grad += request[node] * marginal_cost
                gradients[node, model] = grad
        return gradients

    def mirror_update(self, gradients):
        """
        执行镜像梯度更新

        参数:
        - gradients: 当前梯度 (2D array)

        返回:
        - 更新后的分数分配 (2D array)
        """
        dual_state = np.log(self.fractional_allocation)
        dual_state += self.learning_rate * gradients
        new_allocation = np.exp(dual_state)

        # 投影到可行域 (确保满足资源预算)
        for node in range(self.num_nodes):
            scale = self.budgets[node] / np.sum(self.model_sizes[node] * new_allocation[node])
            new_allocation[node] *= scale
        return new_allocation

    def dep_round(self):
        """
        随机取整 (Dependent Rounding)

        返回:
        - 离散分配 (2D array)
        """
        discrete_allocation = np.zeros_like(self.fractional_allocation)
        for node in range(self.num_nodes):
            probs = self.fractional_allocation[node]
            for model in range(self.num_models):
                if np.random.rand() < probs[model]:
                    discrete_allocation[node, model] = 1
        return discrete_allocation

    def update_allocation(self, requests, costs):
        """
        更新分配

        参数:
        - requests: 请求矩阵
        - costs: 服务成本矩阵

        返回:
        - 离散分配 (2D array)
        """
        gradients = self.compute_gradient(requests, costs)
        self.fractional_allocation = self.mirror_update(gradients)
        return self.dep_round()

# 示例用法
if __name__ == "__main__":
    num_nodes = 5
    num_models = 3
    learning_rate = 0.1
    budgets = [100, 80, 90, 110, 120]
    model_sizes = np.array([
        [20, 30, 50],
        [25, 35, 45],
        [30, 40, 20],
        [50, 20, 30],
        [40, 50, 60]
    ])

    # 模拟请求和成本矩阵
    requests = np.random.randint(1, 10, size=(num_nodes, num_nodes))
    costs = np.random.rand(num_nodes, num_models, 2)  # [latency, accuracy]

    infida = INFIDA(num_nodes, num_models, learning_rate, budgets, model_sizes)

    for t in range(10):  # 运行 10 个时间步
        allocation = infida.update_allocation(requests, costs)
        print(f"第 {t+1} 步的离散分配:\n", allocation)
