import heapq


class Net:
    def __init__(self):
        self.nodes = {}  # 存储所有节点
        self.edges = {}  # 存储节点之间的通信时延

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def add_edge(self, node1_id, node2_id, delay):
        """
        添加节点之间的通信链路及时延.
        """
        if node1_id not in self.edges:
            self.edges[node1_id] = {}
        if node2_id not in self.edges:
            self.edges[node2_id] = {}
        self.edges[node1_id][node2_id] = delay
        self.edges[node2_id][node1_id] = delay  # 双向边

    def get_delay(self, node1_id, node2_id):
        """
        获取两个节点之间的网络时延.
        """
        if node1_id in self.edges and node2_id in self.edges[node1_id]:
            return self.edges[node1_id][node2_id]
        return float('inf')  # 如果没有直接连接，返回无穷大表示不可达

    def route_request(self, request):
        """
        路由请求，并更新 request 的 trace.
        """
        for i in range(len(request.path_nodes) - 1):
            node1 = request.path_nodes[i]
            node2 = request.path_nodes[i + 1]
            delay = self.get_delay(node1.node_id, node2.node_id)
            request.add_to_trace(node2, delay)  # 添加到 trace

    def get_cloud_nodes(self):
        """
        获取所有云端节点.
        """
        return [node for node in self.nodes.values() if node.is_cloud]

    def shortest_path(self, start_node_id):
        """
        最短路径优先，返回从起始节点到达某个云端节点的最短路径.
        """
        queue = [(0, start_node_id, [])]  # (当前路径开销, 当前节点, 路径)
        visited = set()

        while queue:
            (cost, current_node_id, path) = heapq.heappop(queue)
            if current_node_id in visited:
                continue
            path = path + [current_node_id]
            visited.add(current_node_id)

            # 如果当前节点是云端节点，则返回该路径
            if self.nodes[current_node_id].is_cloud:
                return path, cost

            # 遍历当前节点的所有邻居
            for neighbor_id, delay in self.edges.get(current_node_id, {}).items():
                if neighbor_id not in visited:
                    heapq.heappush(queue, (cost + delay, neighbor_id, path))

        return None, float('inf')

    def load_balanced_path(self, start_node_id):
        """
        负载均衡路径选择，根据负载和网络延迟返回最佳路径.
        """
        queue = [(0, start_node_id, [])]  # (当前路径开销, 当前节点, 路径)
        visited = set()

        while queue:
            (cost, current_node_id, path) = heapq.heappop(queue)
            if current_node_id in visited:
                continue
            path = path + [current_node_id]
            visited.add(current_node_id)

            # 如果当前节点是云端节点，则返回该路径
            if self.nodes[current_node_id].is_cloud:
                return path, cost

            # 遍历当前节点的所有邻居
            for neighbor_id, delay in self.edges.get(current_node_id, {}).items():
                neighbor_node = self.nodes[neighbor_id]
                # 根据负载调整路径的代价
                load_penalty = neighbor_node.allocated_capacity / neighbor_node.capacity if not neighbor_node.is_cloud else 0
                adjusted_delay = delay * (1 + load_penalty)

                if neighbor_id not in visited:
                    heapq.heappush(queue, (cost + adjusted_delay, neighbor_id, path))

        return None, float('inf')

    def multipath_routing(self, start_node_id, k=3):
        """
        多路径路由，返回从起始节点到达云端节点的 k 条不同路径.
        """
        queue = [(0, start_node_id, [])]  # (当前路径开销, 当前节点, 路径)
        visited = set()
        paths = []

        while queue and len(paths) < k:
            (cost, current_node_id, path) = heapq.heappop(queue)
            if current_node_id in visited:
                continue
            path = path + [current_node_id]
            visited.add(current_node_id)

            # 如果当前节点是云端节点，则保存该路径
            if self.nodes[current_node_id].is_cloud:
                paths.append((path, cost))
                continue

            # 遍历当前节点的所有邻居
            for neighbor_id, delay in self.edges.get(current_node_id, {}).items():
                if neighbor_id not in visited:
                    heapq.heappush(queue, (cost + delay, neighbor_id, path))

        return paths if paths else None

    def select_path(self, start_node_id, strategy="shortest", k=3):
        """
        根据策略选择路径：最短路径、负载均衡路径或多路径路由.
        """
        if strategy == "shortest":
            return self.shortest_path(start_node_id)
        elif strategy == "load_balanced":
            return self.load_balanced_path(start_node_id)
        elif strategy == "multipath":
            return self.multipath_routing(start_node_id, k)
        else:
            raise ValueError("Unknown strategy")
