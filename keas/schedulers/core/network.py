import numpy as np
import heapq


class Network:
    def __init__(self, nodes=None):
        """
        初始化 Network 对象，用二维矩阵存储节点之间的时延和带宽。
        
        :param nodes: 节点名称的列表，初始化时为空时，表示可以动态添加节点
        """
        self.nodes = nodes if nodes else []  # 初始化节点列表，默认为空
        self.n = len(self.nodes)  # 节点数量
        
        # 创建 n x n 的矩阵，用于存储时延和带宽，初始化为 None 或默认值
        self.latency = np.full((self.n, self.n), None)    # 存储时延 ms
        self.bandwidth = np.full((self.n, self.n), None)  # 存储带宽 mbps
        
        # 创建节点名称与索引的映射关系
        self.node_to_index = {node: index for index, node in enumerate(self.nodes)}
        
    def add_node(self, node_name):
        """添加一个新节点到网络中，并相应扩展时延和带宽矩阵"""
        if node_name in self.node_to_index:
            print(f"Node {node_name} already exists.")
            return
        
        # 将新节点加入节点列表
        self.nodes.append(node_name)
        self.n += 1
        
        # 扩展 latency 和 bandwidth 矩阵，新增一行一列
        new_latency_row = np.full((1, self.n-1), None)
        new_bandwidth_row = np.full((1, self.n-1), None)
        
        self.latency = np.vstack([self.latency, new_latency_row])
        self.bandwidth = np.vstack([self.bandwidth, new_bandwidth_row])
        
        new_latency_column = np.full((self.n, 1), None)
        new_bandwidth_column = np.full((self.n, 1), None)
        
        self.latency = np.hstack([self.latency, new_latency_column])
        self.bandwidth = np.hstack([self.bandwidth, new_bandwidth_column])
        
        # 更新节点名称到索引的映射
        self.node_to_index[node_name] = self.n - 1
        
        print(f"Node {node_name} added to the network.")
    
    def remove_node(self, node_name):
        """删除一个节点，并相应删除时延和带宽矩阵中的行和列"""
        if node_name not in self.node_to_index:
            print(f"Node {node_name} does not exist.")
            return
        
        # 获取节点的索引
        index = self.node_to_index[node_name]
        
        # 删除该节点的时延和带宽行列
        self.latency = np.delete(self.latency, index, axis=0)
        self.latency = np.delete(self.latency, index, axis=1)
        self.bandwidth = np.delete(self.bandwidth, index, axis=0)
        self.bandwidth = np.delete(self.bandwidth, index, axis=1)
        
        # 从节点列表和索引映射中删除该节点
        self.nodes.remove(node_name)
        del self.node_to_index[node_name]
        
        # 更新所有节点的索引映射
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        
        print(f"Node {node_name} removed from the network.")
    
    def add_connection(self, node_a, node_b, latency, bandwidth):
        """添加或更新两个节点之间的连接信息（时延和带宽）"""
        index_a = self.node_to_index[node_a]
        index_b = self.node_to_index[node_b]
        
        # 更新时延和带宽矩阵
        self.latency[index_a][index_b] = latency
        self.bandwidth[index_a][index_b] = bandwidth
        
        # print(f"Connection added/updated between {node_a} and {node_b}: "f"Latency = {latency} ms, Bandwidth = {bandwidth} Mbps")
    
    def modify_connection(self, node_a, node_b, latency=None, bandwidth=None):
        """修改现有连接的时延或带宽"""
        index_a = self.node_to_index[node_a]
        index_b = self.node_to_index[node_b]
        
        if latency is not None:
            self.latency[index_a][index_b] = latency
        if bandwidth is not None:
            self.bandwidth[index_a][index_b] = bandwidth
        
        # print(f"Connection between {node_a} and {node_b} updated: "
        #       f"Latency = {latency if latency is not None else 'unchanged'} ms, "
        #       f"Bandwidth = {bandwidth if bandwidth is not None else 'unchanged'} Mbps")
    
    def set_default_network(self, nodes):
        """
        设置默认网络，将节点之间的时延和带宽按照规则初始化：
        - is_cloud 为 True 的节点之间: latency=5ms, bandwidth=100 Mbps
        - is_cloud 为 True 和 False 的节点之间: latency=100ms, bandwidth=1 Mbps

        :param nodes: 节点字典，key 为 hostname，value 为 Node 对象
        """
        hostnames = list(nodes.keys())  # 获取所有主机名列表

        for i, hostname_a in enumerate(hostnames):
            node_a = nodes[hostname_a]
            for j, hostname_b in enumerate(hostnames):
                if i == j:
                    continue  # 跳过自己与自己的连接

                node_b = nodes[hostname_b]

                if node_a.is_cloud or node_b.is_cloud:
                    # 云节点与非云节点之间
                    self.add_connection(hostname_a, hostname_b, latency=0.1, bandwidth=1)
                else:
                    # 云节点与非云节点之间
                    self.add_connection(hostname_a, hostname_b, latency=0.005, bandwidth=100)


    def get_connection_info(self, node_a, node_b):
        """获取两个节点之间的时延和带宽"""
        index_a = self.node_to_index[node_a]
        index_b = self.node_to_index[node_b]
        
        latency = self.latency[index_a][index_b]
        bandwidth = self.bandwidth[index_a][index_b]
        
        if latency is not None and bandwidth is not None:
            return {'latency': latency, 'bandwidth': bandwidth}
        else:
            return None
    
    def remove_connection(self, node_a, node_b):
        """删除两个节点之间的连接"""
        index_a = self.node_to_index[node_a]
        index_b = self.node_to_index[node_b]
        
        # 删除连接信息，设置为 None 或默认值
        self.latency[index_a][index_b] = None
        self.bandwidth[index_a][index_b] = None
        
        print(f"Connection between {node_a} and {node_b} removed.")
    
    def find_optimal_path(self, start_node, end_node):
        """
        找到节点之间的最短时延路径，同时保证路径上的最小带宽最大。
        
        :param start_node: 起始节点名称
        :param end_node: 目标节点名称
        :return: 包含路径、总时延、最小带宽的字典
        """
        if start_node not in self.node_to_index or end_node not in self.node_to_index:
            raise ValueError(f"Start node {start_node} or end node {end_node} not found in the network.")
        
        n = self.n
        start_idx = self.node_to_index[start_node]
        end_idx = self.node_to_index[end_node]
        
        # 初始化最短时延和路径信息
        min_latency = [float('inf')] * n
        max_bandwidth = [0] * n
        previous_node = [-1] * n
        visited = [False] * n
        
        # 优先队列，用于动态更新最短时延和最大带宽
        pq = []
        heapq.heappush(pq, (0, float('inf'), start_idx))  # (当前时延, 路径最小带宽, 当前节点)
        min_latency[start_idx] = 0
        max_bandwidth[start_idx] = float('inf')
        
        while pq:
            current_latency, current_bandwidth, current_idx = heapq.heappop(pq)
            
            if visited[current_idx]:
                continue
            visited[current_idx] = True
            
            # 遍历所有邻接节点
            for neighbor_idx in range(n):
                if self.latency[current_idx][neighbor_idx] is None or self.bandwidth[current_idx][neighbor_idx] is None:
                    continue  # 如果没有连接，跳过
                
                latency = self.latency[current_idx][neighbor_idx]
                bandwidth = self.bandwidth[current_idx][neighbor_idx]
                
                # 更新路径上的最小带宽为当前路径的瓶颈值
                new_bandwidth = min(current_bandwidth, bandwidth)
                new_latency = current_latency + latency
                
                # 如果找到更优路径，更新数据
                if (new_latency < min_latency[neighbor_idx] or
                    (new_latency == min_latency[neighbor_idx] and new_bandwidth > max_bandwidth[neighbor_idx])):
                    min_latency[neighbor_idx] = new_latency
                    max_bandwidth[neighbor_idx] = new_bandwidth
                    previous_node[neighbor_idx] = current_idx
                    heapq.heappush(pq, (new_latency, new_bandwidth, neighbor_idx))
        
        # 构造路径
        path = []
        current = end_idx
        while current != -1:
            path.append(self.nodes[current])
            current = previous_node[current]
        path.reverse()
        
        # 如果路径不可达
        if min_latency[end_idx] == float('inf'):
            return None
        
        return {
            'path': path,
            'total_latency': min_latency[end_idx],
            'min_bandwidth': max_bandwidth[end_idx]
        }
    
    def __str__(self):
        """打印网络的时延和带宽矩阵"""
        s = "Network Latency Matrix:\n"
        s += "Nodes: " + ", ".join(self.nodes) + "\n"
        s += "Latency (ms):\n"
        s += str(self.latency) + "\n"
        s += "Bandwidth (Mbps):\n"
        s += str(self.bandwidth) + "\n"
        return s