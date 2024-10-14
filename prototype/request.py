class Request:
    def __init__(self, request_id, task_id, source_node, path_nodes):
        """
        Request 初始化.

        request_id: 请求的唯一ID
        task_id: 请求对应的任务ID（指定要处理的任务）
        source_node: 请求发起的源节点
        path_nodes: 请求在服务路径中经过的节点列表
        """
        self.request_id = request_id  # 请求ID
        self.task_id = task_id  # 请求所对应的任务ID
        self.source_node = source_node  # 发起请求的源节点
        self.path_nodes = path_nodes  # 请求服务路径上经过的节点列表
        self.trace = []  # 跟踪已经经过的节点以及网络时延

    def add_to_trace(self, node, delay):
        """
        将经过的节点和对应的网络时延添加到trace中.

        node: 经过的节点 (Node 类实例)
        delay: 节点之间的网络时延
        """
        self.trace.append((node.node_id, delay))

    def total_network_latency(self):
        """
        计算已经过节点的总网络延迟.
        """
        return sum([delay for _, delay in self.trace])

    def update_path(self, all_nodes, max_path_length=4):
        """
        根据当前网络和节点状态更新请求的路径.

        all_nodes: 所有可用的节点列表 (Node 实例)
        max_path_length: 路径中允许的最大节点数量，默认为 3
        """
        # 过滤出可以处理该任务（task_id）的节点，且节点有可用的资源
        # available_nodes = [node for node in all_nodes if node.is_available() and node.can_handle_task(self.task_id)]
        # 过滤出节点有可用的资源
        available_nodes = [node for node in all_nodes if node.is_available()]

        if not available_nodes:
            print(f"Request {self.request_id}无法找到可用的节点处理任务 {self.task_id}")
            return

        # 根据网络延迟和负载对节点进行排序，选择负载低且延迟最小的节点
        # 使用浮点数计算负载比例: allocated_capacity / capacity
        sorted_nodes = sorted(available_nodes,
                              key=lambda node: (node.network_latency, node.allocated_capacity / node.capacity))

        # 确保路径节点数不超过 max_path_length
        self.path_nodes = sorted_nodes[:max_path_length]

        print(f"Request {self.request_id} updated path: {[node.node_id for node in self.path_nodes]}")

    def compute_transmission_cost(self, model_node, net):
        """
        计算从请求的源节点到模型节点的传输消耗.
        model_node: 模型所在的节点 (Node 类实例)
        net: 网络 (Net 类实例)

        返回总传输消耗（网络延迟）。
        """
        # 获取当前路径的节点ID
        if model_node not in self.path_nodes:
            # 如果模型节点不在路径中，则重新路由
            path, _ = net.select_path(self.source_node.node_id, strategy="shortest")
            self.path_nodes = [net.nodes[node_id] for node_id in path]

        # 计算从源节点到模型节点的传输延迟
        total_latency = 0
        for i in range(len(self.path_nodes) - 1):
            node1 = self.path_nodes[i]
            node2 = self.path_nodes[i + 1]
            if node2 == model_node:
                break  # 到达模型所在节点时停止计算
            total_latency += net.get_delay(node1.node_id, node2.node_id)

        return total_latency
