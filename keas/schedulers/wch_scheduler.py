import numpy as np
from .core import InferenceServiceNet, IsnRequest

class WCHScheduler:
    def __init__(self, isn: InferenceServiceNet):
        self.isn = isn
        # 创建 nodes 和 models 的映射
        self.node_index_map = {node_name: index for index, node_name in enumerate(self.isn.nodes.keys())}
        self.index_node_map = {index: node_name for index, node_name in enumerate(self.isn.nodes.keys())}
        self.model_index_map = {model_name: index for index, model_name in enumerate(self.isn.models.keys())}
        self.index_model_map = {index: model_name for index, model_name in enumerate(self.isn.models.keys())}


    def schedule(self, requests: list[IsnRequest]):
        num_nodes = len(self.isn.nodes)
        num_models = len(self.isn.models)
        num_requests = len(requests)
        X = []

        weights = self.generate_weight_vectors(100, 2)

        for weight in weights:

            temp_isn = self.isn.deep_copy()
            x = np.zeros((num_requests, num_models, num_nodes), dtype=int)

            temp_total_success = 0
            temp_total_latency = 0
            temp_total_accuracy = 0
            for i, request in enumerate(requests):

                path_to_cloud = temp_isn.find_optimal_path(request.hostname, "cloud_1")
                # print(path_to_cloud)
                cloud_transfer_latency = (
                    path_to_cloud['total_latency'] + 
                    1.0 * request.data_count * request.data_per_size / path_to_cloud['min_bandwidth']
                )

                temp_ww = float('-inf')
                temp_model_index = -1
                temp_node_index = -1
                temp_latency = float('inf')
                temp_accuracy = 0

                for model_name, model in temp_isn.models.items():
                
                    cloud_inference_latency = (
                            model.gflops / 
                            temp_isn.nodes["cloud_1"].cpu_gflops / 
                            model.capacity.cpu
                        ) * request.data_count
                    cloud_latency = cloud_transfer_latency + cloud_inference_latency
                    
                    # print("cloud: ",cloud_latency, cloud_transfer_latency, cloud_inference_latency)
                    for node_name, node in temp_isn.nodes.items():
                        
                        if model.capacity.is_larger_than(node.remain_capacity):
                            continue
                        model_index = self.model_index_map[model_name]  # 获取模型的序号
                        node_index = self.node_index_map[node_name]  # 获取节点的序号
                        path_to_node = temp_isn.find_optimal_path(request.hostname, node_name)
                        # print(path_to_node)
                        node_transfer_latency = (
                            path_to_node['total_latency'] + 
                            1.0 * request.data_count * request.data_per_size / path_to_node['min_bandwidth']
                        )

                        node_inference_latency = (
                            model.gflops / 
                            node.cpu_gflops / 
                            model.capacity.cpu
                        ) * request.data_count
                        total_latency = node_transfer_latency + node_inference_latency
                        # print(total_latency,node_transfer_latency,node_inference_latency)
                        if(total_latency > request.time_period):
                            continue
                        
                        total_latency = total_latency / cloud_latency

                        ww = weight[0] * model.accuracy - weight[1] * total_latency
                        
                        if ww > temp_ww:
                            temp_ww = ww
                            temp_model_index = model_index
                            temp_node_index = node_index
                            temp_latency = total_latency
                            temp_accuracy = model.accuracy
                        
                if temp_model_index != -1:
                    temp_total_success += 1
                    print(temp_model_index, temp_node_index)
                    print(temp_isn.nodes[self.index_node_map[temp_node_index]].remain_capacity)
                    print(temp_isn.models[self.index_model_map[temp_model_index]].capacity)
                    temp_isn.nodes[self.index_node_map[temp_node_index]].add_model(self.index_model_map[temp_model_index], temp_isn.models[self.index_model_map[temp_model_index]].capacity)
                x[i, temp_model_index, temp_node_index] = 1
                temp_total_latency += temp_latency
                temp_total_accuracy += temp_accuracy

            X.append({
                "x": x,
                "weight": weight,
                "success": temp_total_success,
                "accuracy": temp_total_accuracy,
                "latency": temp_total_latency
            })

        pareto = self.pareto_frontier(X)
        
        return pareto

    def pareto_frontier(self, X):
        """
        生成帕累托前沿解集。

        参数：
            X: List[dict] 包含解的信息，每个解有以下字段：
                - 'x': 解的具体值（可以是向量、参数等）
                - 'weight': 权重值（可选，不影响帕累托排序）
                - 'success': 成功率，需要最大化
                - 'accuracy': 准确率，需要最大化
                - 'latency': 时延，需要最小化

        返回：
            List[dict]: 帕累托前沿解集。
        """
        def dominates(a, b):
            """
            检查解 a 是否支配解 b。
            """
            return (
                a['accuracy'] >= b['accuracy'] and
                a['success'] >= b['success'] and
                a['latency'] <= b['latency'] and
                (
                    a['accuracy'] > b['accuracy'] or
                    a['success'] > b['success'] or
                    a['latency'] < b['latency']
                )
            )

        # 帕累托前沿解集
        pareto_front = []

        for candidate in X:
            # 检查当前解是否被帕累托前沿解集中的解支配
            is_dominated = False
            non_dominated_solutions = []
            for solution in pareto_front:
                if dominates(solution, candidate):
                    is_dominated = True
                    break
                if not dominates(candidate, solution):
                    non_dominated_solutions.append(solution)

            if not is_dominated:
                # 将候选解加入帕累托前沿，并去除被候选解支配的解
                non_dominated_solutions.append(candidate)
            
            pareto_front = non_dominated_solutions

        return pareto_front

    
    def generate_weight_vectors(self, num_weights, num_objectives):
        """
        生成用于标量化的均匀分布权重向量。
        """
        weights = np.random.dirichlet(np.ones(num_objectives), size=num_weights)
        return weights

