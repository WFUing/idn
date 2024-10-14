class INFIDA:
    def __init__(self, nodes, models, net, alpha, learning_rate, max_iterations=1000):
        """
        INFIDA 初始化.

        nodes: 所有节点列表 (Node 类的实例)
        models: 所有模型列表 (Model 类的实例)
        net: 网络拓扑结构 (Net 类的实例)
        alpha: 控制延迟与准确性之间权衡的参数
        learning_rate: 梯度更新的学习率
        max_iterations: 最大迭代次数
        """
        self.nodes = nodes
        self.models = models
        self.net = net  # 网络拓扑
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def infida_algorithm(self, requests):
        """
        主函数：使用 INFIDA 算法进行模型的分配.

        requests: 请求列表 (Request 类的实例)
        """
        # 更新每个请求的路径
        # for request in requests:
        #     path, _ = self.net.select_path(request.source_node.node_id, strategy="shortest")
        #     request.path_nodes = [self.net.nodes[node_id] for node_id in path]

        for iteration in range(self.max_iterations):
            # 输出调试信息
            print(f"Iteration {iteration-1}:")
            for node in self.nodes:
                print(f"Node {node.node_id} Allocation: {node.allocation}")

            # 1. 计算子梯度
            subgradients = self.compute_subgradients(requests)

            # 2. OMA 更新分配
            for node in self.nodes:
                if node.is_cloud:  # 跳过云端节点的更新
                    continue
                fractional_allocation = self.update_fractional_allocation(node, subgradients[node.node_id])
                node.allocation = self.dependent_rounding(fractional_allocation)

    def compute_subgradients(self, requests):
        """
        计算每个节点和模型的子梯度.

        requests: 请求列表 (Request 类的实例)
        """
        subgradients = {}
        for node in self.nodes:
            node_gradients = {}
            for model in self.models:
                node_gradients[model.model_id] = self.compute_model_gradient(node, model, requests)
            subgradients[node.node_id] = node_gradients
        return subgradients

    def compute_model_gradient(self, node, model, requests):
        """
        计算模型在节点上的子梯度.

        node: 当前节点 (Node 类实例)
        model: 当前模型 (Model 类实例)
        requests: 请求列表 (Request 类实例)
        """
        gradient = 0
        for request in requests:
            if request.task_id == model.task_id:
                # 1. 计算当前模型的可用容量 l^t_{v\rho,m}
                l_t_v_rho_m = self.compute_available_capacity(node, model, request)
                # 如果可用容量为 0，则跳过计算
                if l_t_v_rho_m <= 0:
                    continue
                l_t_v_rho_m = l_t_v_rho_m / max(1, node.capacity)  # 确保归一化

                # 2. 计算当前模型的服务成本 C_{v\rho,m}
                gamma_k = model.compute_service_cost(request, node, self.net, self.alpha)

                # 3. 计算服务路径上最优模型的服务成本 γ_{K_{\rho}^*(y^t)\rho}
                gamma_k_optimal = self.compute_optimal_service_cost(request)

                # 4. 使用公式 (18) 计算子梯度
                if gamma_k_optimal > gamma_k:
                    # 只有在最优服务成本优于当前节点时才计算梯度
                    gradient += l_t_v_rho_m * (gamma_k_optimal - gamma_k)

        return gradient

    def compute_available_capacity(self, node, model, request):
        """
        计算模型的潜在可用容量.

        node: 当前节点 (Node 类实例)
        model: 当前模型 (Model 类实例)
        request: 请求 (Request 类实例)
        """
        # 计算节点剩余容量
        node_available_capacity = node.capacity - node.allocated_capacity

        # 模型的资源需求
        model_resource_demand = model.size

        # 返回节点剩余容量与模型资源需求之间的最小值
        return min(node_available_capacity, model_resource_demand)

    def compute_optimal_service_cost(self, request):
        """
        计算请求在所有节点中可能遇到的最优服务成本.

        request: 请求 (Request 类实例)
        """
        optimal_cost = float('inf')  # 初始化为无穷大

        # 遍历所有节点，寻找全局最优的模型
        for node in request.path_nodes:
            for model in node.models:
                if model.task_id == request.task_id:
                    # 计算模型的服务成本
                    current_cost = model.compute_service_cost(request, node, self.net, self.alpha)
                    # 选择最优的模型成本
                    if current_cost < optimal_cost:
                        optimal_cost = current_cost

        # 处理云端节点的服务成本
        cloud_node = request.path_nodes[-1]
        if cloud_node.can_handle_task(request.task_id):
            cloud_model = next((model for model in cloud_node.models if model.task_id == request.task_id), None)
            if cloud_model:
                cloud_cost = cloud_model.compute_service_cost(request, cloud_node, self.net, self.alpha)
                optimal_cost = min(optimal_cost, cloud_cost)

        # 返回全局最优的服务成本（节点中的最优模型和云端模型中较小的）
        return optimal_cost

    def update_fractional_allocation(self, node, gradients):
        """
        使用在线镜像上升法（MOA）更新模型的分配.

        node: 当前节点
        gradients: 当前节点的梯度字典 {model_id: gradient_value}
        """
        updated_allocation = {}
        total_gradient = sum(gradients.values())  # 计算总的梯度值，确保有更新

        if total_gradient == 0:
            return node.allocation  # 没有梯度更新，返回原来的分配

        for model_id, gradient in gradients.items():
            # 使用子梯度和学习率更新分配
            updated_allocation[model_id] = node.allocation.get(model_id, 0) + self.learning_rate * gradient
            # 确保更新后的分配不为负数
            updated_allocation[model_id] = max(updated_allocation[model_id], 0)
            # 确保更新后的分配不会超过节点的总容量
            updated_allocation[model_id] = min(updated_allocation[model_id], node.capacity)

        return self.project_allocation_to_constraints(updated_allocation, node.capacity)

    def project_allocation_to_constraints(self, allocation, capacity):
        """
        投影操作：确保分配满足容量约束.

        allocation: 当前节点的分配字典 {model_id: fractional_value}
        capacity: 节点容量
        """
        total_allocated = sum(allocation.values())
        if total_allocated <= capacity:
            return allocation
        # 如果超过了容量，按比例缩小分配
        scaling_factor = capacity / total_allocated
        return {model_id: value * scaling_factor for model_id, value in allocation.items()}

    def dependent_rounding(self, fractional_allocation):
        """
        依赖四舍五入操作，将连续分配转化为离散分配.

        fractional_allocation: 连续分配向量 {model_id: fractional_value}
        """
        rounded_allocation = {}
        total = sum(fractional_allocation.values())

        # 修正 total 为负数的情况，如果 total 小于等于 0，使用一个小正数 epsilon
        epsilon = 1e-6
        if total <= 0:
            total = epsilon

        for model_id, value in fractional_allocation.items():
            # 根据修正后的 total 进行四舍五入
            if value / total > 0.5:
                rounded_allocation[model_id] = 1
            else:
                rounded_allocation[model_id] = 0

        return rounded_allocation
