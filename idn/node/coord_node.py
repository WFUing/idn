
class CoordNode:
    def __init__(self):
        # 存储所有节点的字典，键为节点ID，值为节点的具体信息
        self.nodes = {}

    def add_node(self, node_id, location, models, task_types, max_capacity, latency, bandwidth, status="Normal"):
        """添加新节点到全局节点路由表"""
        if node_id in self.nodes:
            print(f"Node {node_id} already exists.")
            return
        self.nodes[node_id] = {
            "Location": location,
            "Models": models,  # 可用模型列表，如 ['ResNet50', 'MobileNet']
            "Task Types": task_types,  # 支持任务类型，如 ['Classification', 'Detection']
            "Max Capacity": max_capacity,  # 最大容量，如 [100, 50]
            "Current Load": [0] * len(max_capacity),  # 当前负载，初始化为0
            "Available Capacity": max_capacity.copy(),  # 可用容量，初始化与最大容量相同
            "Latency": latency,  # 节点延迟
            "Bandwidth": bandwidth,  # 节点带宽
            "Status": status  # 节点状态
        }

    def update_node(self, node_id, load):
        """更新节点的负载和可用容量"""
        if node_id not in self.nodes:
            print(f"Node {node_id} does not exist.")
            return
        node = self.nodes[node_id]
        node["Current Load"] = load
        node["Available Capacity"] = [
            max_cap - cur_load for max_cap, cur_load in zip(node["Max Capacity"], node["Current Load"])
        ]
        # 更新状态，如果可用容量为0，则标记为过载
        if all(cap <= 0 for cap in node["Available Capacity"]):
            node["Status"] = "Overloaded"
        else:
            node["Status"] = "Normal"

    def get_optimal_node(self, task_type):
        """根据任务类型找到最佳的节点"""
        optimal_node = None
        max_available_capacity = -1

        for node_id, node in self.nodes.items():
            if task_type in node["Task Types"] and node["Status"] == "Normal":
                index = node["Task Types"].index(task_type)
                available_capacity = node["Available Capacity"][index]

                # 选择可用容量最大的节点
                if available_capacity > max_available_capacity:
                    max_available_capacity = available_capacity
                    optimal_node = node_id

        if optimal_node:
            print(f"Optimal node for task {task_type} is {optimal_node}")
            return optimal_node
        else:
            print(f"No available nodes for task {task_type}")
            return None

    def print_node_info(self):
        """打印所有节点的信息"""
        for node_id, node in self.nodes.items():
            print(f"Node ID: {node_id}")
            for key, value in node.items():
                print(f"  {key}: {value}")
            print("-" * 40)


