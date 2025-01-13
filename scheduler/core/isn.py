from .model import ModelAllocatedConfig
from .node import Capacity, Node
from .network import Network
import redis
from .request import RequestRecords


class InferenceServiceNet:
    def __init__(self):
        self.network = Network()  # 初始化 Network
        self.nodes = []           # 存储 Node 列表
        self.sync_nodes()
        self.models = {}
        self.sync_models()
        self.sync_network()
        self.requests = RequestRecords()
        # self.macfg = ModelConfig()

    def get_hostname_by_index(self, index):
        return self.nodes[index].hostname

    def get_model_name(self, type, index):
        # 检查类型是否存在
        if type not in self.models:
            raise KeyError(f"Type '{type}' not found in models.")

        # 检查类型下的索引是否存在
        if index >= len(self.models[type]):
            raise IndexError(f"Index '{index}' out of range for type '{type}'.")

        model_names = list(self.models[type].keys())
        return model_names[index]
    

    def get_model(self, type, name):
        # 检查类型是否存在
        if type not in self.models:
            raise KeyError(f"Type '{type}' not found in models.")

        if name not in self.models[type]:
            raise KeyError(f"name '{name}' not found in models {type}.")

        return self.models[type][name]

    def add_node(self, node):
        """向 InferenceServiceNet 中添加节点"""
        self.nodes.append(node)
        self.network.add_node(node.hostname)  # 同时在 Network 中添加节点
    
    def remove_node(self, node_name):
        """从 InferenceServiceNet 中删除节点"""
        self.nodes = [node for node in self.nodes if node.hostname != node_name]
        self.network.remove_node(node_name)  # 同时在 Network 中删除节点
    
    def add_connection(self, node_a, node_b, latency=0, bandwidth=0):
        """在 Network 中为两个节点添加连接"""
        self.network.add_connection(node_a.hostname, node_b.hostname, latency, bandwidth)
    
    def add_request(self, request):
        self.requests.add_request(request)

    def show_request(self):
        self.requests.get_stats()

    def find_optimal_path(self, start_node, end_node):
        return self.network.find_optimal_path(start_node, end_node)

    def sync_models(self):
        """从 Redis 同步节点数据"""
        r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # 遍历 type:1 到 type:7 获取每种类型的模型数据
        for i in range(1, 8):
            key = f"type:{i}"
            
            type = {}

            # 获取该键的所有字段和值
            fields = r.hgetall(key)
            
            # 格式化数据并保存到 models 字典
            for field, value in fields.items():
                # 获取模型名称与指标名（如 ResNet18:top_1）
                model_name, metric = field.split(":")
                
                # 将值转换为浮动数值
                # print(value)
                value = float(value)
                
                # 确保该模型在 models 字典中已存在
                if model_name not in type:
                    type[model_name] = {}
                
                # 保存该指标
                type[model_name][metric] = value

            self.models[key] = type

    def sync_nodes(self):
        """从 Redis 同步节点数据"""
        r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

        node_names = r.smembers('nodes')  # 获取所有节点名称
        
        self.nodes = []  # 重置本地节点列表
        for node_name in node_names:
            # 从 Redis 中获取节点的资源数据
            resources = r.hgetall(f'resources:{node_name}')
            if not resources:
                continue  # 如果节点没有资源信息，跳过

            # 创建 Capacity 对象
            capacity = Capacity(
                cpu=int(resources.get('cpu', 0)),
                memory=int(resources.get('memory', 0)),
                gpu=int(resources.get('gpu', 0)),
                vpu=int(resources.get('vpu', 0)),
            )

            # 初始化节点
            node = Node(
                hostname=node_name,
                capacity=capacity,
                is_cloud='cloud' in node_name,
                extra=float(resources.get('extra', 0)),
                cpu_gflops=float(resources.get('cpu_gflops', 0)),
                gpu_gflops=float(resources.get('gpu_gflops', 0)),
                vpu_gflops=float(resources.get('vpu_gflops', 0))
            )
            self.add_node(node)

    def sync_network(self):
        """从 Redis 同步网络（延迟和带宽）数据"""
        r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

        node_names = r.smembers('nodes')  # 获取所有节点名称

        for source_node in node_names:
            # 获取延迟和带宽数据
            latency_data = r.hgetall(f'latency:{source_node}')
            bandwidth_data = r.hgetall(f'bandwidth:{source_node}')
            
            for target_node in node_names:
                if source_node == target_node:
                    continue  # 跳过自身

                # 从 Redis 中获取延迟和带宽值
                latency = float(latency_data.get(target_node, 0))
                bandwidth = float(bandwidth_data.get(target_node, 0))

                # 添加到网络中
                self.network.add_connection(source_node, target_node, latency, bandwidth)


    def __repr__(self):
        return f"InferenceServiceNet(nodes={self.nodes}, network={self.network})"

