class Node:
    def __init__(self, node_id, capacity, is_cloud=False):
        """
        Node 初始化.

        node_id: 节点唯一ID
        capacity: 节点存储容量
        allocated_capacity: 当前节点的已经存储的容量，默认为 0
        allocated_load: 节点的已经使用的负载容量
        is_cloud: 是否为云端节点
        """
        self.node_id = node_id  # 节点ID
        self.capacity = capacity  # 节点总容量
        self.allocated_capacity = 0  # 已分配的资源量
        self.is_cloud = is_cloud  # 判断是否是云端节点
        self.models = []  # 节点上可用的模型
        self.allocation = {}  # 模型的分配情况

    def is_available(self):
        """
        判断节点是否有足够的资源处理更多任务.
        节点的负载不能超过其容量。
        """
        return self.allocated_capacity < self.capacity

    def can_handle_task(self, task_id):
        """
        判断节点上是否有模型可以处理特定的任务.
        """
        for model in self.models:
            if model.task_id == task_id:
                return True
        return False

    def add_model(self, model):
        """
        在节点上添加模型.
        """
        if self.allocated_capacity + model.size <= self.capacity:
            self.models.append(model)
            if(model.model_id in self.allocation.keys()):
                self.allocation[model.model_id] += 1
            else:
                self.allocation[model.model_id] = 1
            self.allocated_capacity += model.size
            return True
        else:
            print(f"Node {self.node_id} cannot add Model {model.model_id} due to capacity constraints.")
            return False

    def release_model(self, model):
        """
        释放分配给模型的资源，并从节点中移除模型.
        """
        if self.allocation[model.model_id] > 0:
            self.allocation[model.model_id] -= 1
            self.allocated_capacity -= model.size
            if self.allocation[model.model_id] == 0:
                self.models.remove(model)
        else:
            print(f"Node {self.node_id} cannot release {model.model_id} due to capacity constraints.")

