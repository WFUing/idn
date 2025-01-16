from schedulers.wch_scheduler import WCHScheduler
from schedulers.core import InferenceServiceNet, Node, Capacity, IsnModel, IsnRequest

def create_model(name, cpu, ram, gpu, accuracy, model_gflops=0):
    return IsnModel(model_name=name, model_accuracy=accuracy, model_gflops=model_gflops, cpu=cpu, memory=ram, gpu=gpu)

if __name__ == "__main__":
    
    edge_node_1 = Node(hostname="edge_1", capacity=Capacity(cpu=2, memory=4, gpu=1), cpu_gflops = 400)
    edge_node_2 = Node(hostname="edge_2", capacity=Capacity(cpu=4, memory=8, gpu=2), cpu_gflops = 400)
    edge_node_3 = Node(hostname="edge_3", capacity=Capacity(cpu=1, memory=2, gpu=0), cpu_gflops = 450)
    edge_node_4 = Node(hostname="edge_4", capacity=Capacity(cpu=4, memory=8, gpu=0), cpu_gflops = 350)
    edge_node_5 = Node(hostname="edge_5", capacity=Capacity(cpu=2, memory=4, gpu=0), cpu_gflops = 370)
    edge_node_6 = Node(hostname="edge_6", capacity=Capacity(cpu=3, memory=7, gpu=0), cpu_gflops = 390)
    edge_node_7 = Node(hostname="edge_7", capacity=Capacity(cpu=2, memory=9, gpu=0), cpu_gflops = 350)
    edge_node_8 = Node(hostname="edge_8", capacity=Capacity(cpu=1, memory=2, gpu=0), cpu_gflops = 300)

    cloud_node = Node(hostname="cloud_1", capacity=Capacity(cpu=float('inf'), memory=float('inf'), gpu=float('inf')), is_cloud=True, cpu_gflops=1000)


    models = {
        "m1": create_model("m1", 0.5, 0.2, 0, 0.625, model_gflops=500),
        "m2": create_model("m2", 0.5, 0.4, 0, 0.650, model_gflops=500),
        "m3": create_model("m3", 1, 0.6, 0, 0.675, model_gflops=740),
        "m4": create_model("m4", 1, 0.8, 0, 0.700, model_gflops=760),
        "m5": create_model("m5", 1, 1, 0, 0.725, model_gflops=800),
        "m6": create_model("m6", 1, 2, 0, 0.750, model_gflops=900),
        "m7": create_model("m7", 2, 4, 0, 0.775, model_gflops=1400),
        "m8": create_model("m8", 2, 4, 1, 0.800, model_gflops=1500),
        "m*": create_model("m*", 4, 8, 2, 0.900, model_gflops=2500)  
    }


    isn = InferenceServiceNet(models=models)

    isn.add_node(edge_node_1)
    isn.add_node(edge_node_2)
    isn.add_node(edge_node_3)
    # isn.add_node(edge_node_4)
    # isn.add_node(edge_node_5)
    # isn.add_node(edge_node_6)
    # isn.add_node(edge_node_7)
    # isn.add_node(edge_node_8)
    isn.add_node(cloud_node)

    isn.network.set_default_network(isn.nodes)
    request1 = IsnRequest(time_period=700, hostname="edge_1", data_count=100, data_persize=10)
    # request2 = IsnRequest(time_period=700, hostname="edge_2", data_count=100, data_persize=10)
    # request3 = IsnRequest(time_period=700, hostname="edge_4", data_count=100, data_persize=10)
    # request4 = IsnRequest(time_period=700, hostname="edge_5", data_count=100, data_persize=10)
    # request5 = IsnRequest(time_period=700, hostname="edge_7", data_count=100, data_persize=10)

    # requests = [request1, request2, request3, request4, request5]
    requests = [request1]

    scheduler = WCHScheduler(isn)  # 使用占位 isn 对象
    # scheduler.schedule(requests)
    print(scheduler.schedule(requests))

