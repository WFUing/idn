from prototype.infida import INFIDA
from prototype.model import Model
from prototype.node import Node
from prototype.request import Request


from prototype.infida import INFIDA
from prototype.model import Model
from prototype.node import Node
from prototype.request import Request
from prototype.net import Net


def test_infida():
    # 创建网络实例
    net = Net()

    # 定义节点，区分边缘节点和云端节点
    node1 = Node(node_id=1, capacity=1000)  # 边缘节点
    node2 = Node(node_id=2, capacity=800)   # 边缘节点
    node3 = Node(node_id=3, capacity=600)   # 边缘节点
    cloud_node = Node(node_id=4, capacity=2000, is_cloud=True)  # 云端节点

    # 将节点添加到网络中
    net.add_node(node1)
    net.add_node(node2)
    net.add_node(node3)
    net.add_node(cloud_node)

    # 定义节点之间的通信时延
    net.add_edge(1, 2, delay=10)
    net.add_edge(2, 3, delay=20)
    net.add_edge(3, 4, delay=30)

    # 定义模型
    model1 = Model(model_id=1, task_id=1, size=300, accuracy=0.85, inference_latency=30)
    model2 = Model(model_id=2, task_id=1, size=400, accuracy=0.90, inference_latency=50)
    model3 = Model(model_id=3, task_id=2, size=500, accuracy=0.88, inference_latency=60)
    cloud_model1 = Model(model_id=4, task_id=1, size=1000, accuracy=0.95, inference_latency=100)
    cloud_model2 = Model(model_id=5, task_id=2, size=1000, accuracy=0.92, inference_latency=100)

    # 将模型分配到节点上
    node1.add_model(model1)
    node2.add_model(model2)
    node3.add_model(model3)
    cloud_node.add_model(cloud_model1)
    cloud_node.add_model(cloud_model2)

    # 定义请求，假设每个请求有不同的服务路径
    request1 = Request(request_id=1, task_id=1, source_node=node1, path_nodes=[node1, node2, node3, cloud_node])
    request2 = Request(request_id=2, task_id=1, source_node=node2, path_nodes=[node2, node3, cloud_node])
    request3 = Request(request_id=3, task_id=2, source_node=node3, path_nodes=[node3, cloud_node])

    requests = [request1, request2, request3]

    # 定义 INFIDA 算法的参数
    alpha = 0.7  # 控制延迟与准确性之间的权衡
    learning_rate = 0.1  # 梯度更新步长
    max_iterations = 10  # 迭代次数

    # 实例化 INFIDA 类
    infida = INFIDA(nodes=[node1, node2, node3, cloud_node], models=[model1, model2, model3], net=net,
                    alpha=alpha, learning_rate=learning_rate, max_iterations=max_iterations)

    # 执行 INFIDA 算法
    infida.infida_algorithm(requests)

    # 输出结果
    for node in [node1, node2, node3, cloud_node]:
        print(f"Node {node.node_id} Allocation: {node.allocation}")


if __name__ == "__main__":
    test_infida()

