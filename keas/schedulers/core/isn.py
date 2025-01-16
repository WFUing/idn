from .node import Capacity, Node
from .network import Network
import copy


class InferenceServiceNet:
    def __init__(self, nodes=None, models=None):
        if nodes is None:
            nodes = {}
        if models is None:
            models = {}
        self.nodes = nodes                          # 节点, key: hostname, value: node
        self.models = models                        # 模型, key: model_name, value: model
        self.network = Network(list(nodes.keys()))  # 网络，初始化时传入所有 hostname

    def get_hostname_by_index(self, index):
        return list(self.nodes.keys())[index]

    def get_model_name(self, index):
        return list(self.models.keys())[index]

    def add_node(self, node):
        """向 InferenceServiceNet 中添加节点"""
        self.nodes[node.hostname] = node
        self.network.add_node(node.hostname)  # 同时在 Network 中添加节点
    
    def remove_node(self, node_name):
        """从 InferenceServiceNet 中删除节点"""
        if node_name in self.nodes:
            del self.nodes[node_name]
        self.network.remove_node(node_name)  # 同时在 Network 中删除节点
    
    def add_connection(self, node_a, node_b, latency=0, bandwidth=0):
        """在 Network 中为两个节点添加连接"""
        self.network.add_connection(node_a.hostname, node_b.hostname, latency, bandwidth)

    def find_optimal_path(self, start_node, end_node):
        return self.network.find_optimal_path(start_node, end_node)

    def deep_copy(self):
        """实现深拷贝"""
        return copy.deepcopy(self)

    def __repr__(self):
        return f"InferenceServiceNet(nodes={self.nodes}, network={self.network})"

