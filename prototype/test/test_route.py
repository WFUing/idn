from prototype.net import Net
from prototype.node import Node


def test_net():
    # 创建网络
    net = Net()

    # 创建节点，node1 和 node2 是普通节点，cloud_node 是云端节点
    node1 = Node(node_id=1, capacity=1000)
    node2 = Node(node_id=2, capacity=800)
    cloud_node = Node(node_id=999, capacity=10000, is_cloud=True)

    # 添加节点到网络
    net.add_node(node1)
    net.add_node(node2)
    net.add_node(cloud_node)

    # 添加通信链路和时延
    net.add_edge(1, 2, 10)
    net.add_edge(2, 999, 20)

    # 测试最短路径
    print("Shortest Path:", net.select_path(1, strategy="shortest"))

    # 测试负载均衡路径
    print("Load Balanced Path:", net.select_path(1, strategy="load_balanced"))

    # 测试多路径路由
    print("Multipath Routing:", net.select_path(1, strategy="multipath", k=2))


if __name__ == "__main__":
    test_net()
