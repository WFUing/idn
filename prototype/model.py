class Model:
    def __init__(self, model_id, task_id, size, accuracy, inference_latency):
        self.model_id = model_id
        self.task_id = task_id
        self.size = size
        self.accuracy = accuracy
        self.inference_latency = inference_latency

    def compute_cloud_service_cost(self, request, alpha):
        """
        计算模型在节点上的服务成本.
        服务成本 = 已经过的网络延迟 + 推理延迟 + (1 - 准确率) * alpha
        """
        total_network_latency = request.total_network_latency()
        return total_network_latency + self.inference_latency + alpha * (1 - self.accuracy)

    def compute_service_cost(self, request, node, net, alpha):
        """
        计算模型在节点上的服务成本.
        服务成本 = 已经过的网络延迟 + 推理延迟 + (1 - 准确率) * alpha
        """
        transmission_cost = request.compute_transmission_cost(node, net)
        return transmission_cost + self.inference_latency + alpha * (1 - self.accuracy)
