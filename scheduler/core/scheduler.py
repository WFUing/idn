import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

from .enum.request_type import get_headers_by_type
from .node import Capacity


class Scheduler:
    def __init__(self, isn, request):
        self.isn = isn
        self.request = request

    def get_nodes_and_models_for_request(self):
        """
        根据请求筛选满足条件的节点和模型列表
        :param isn: InferenceServiceNet 对象
        :param request: IsnRequest 对象
        :return: 满足条件的节点和模型列表
        """
        results = []
        model_type = f"type:{self.request.req_type}"  # 根据请求类型获取模型类别

        for node in self.isn.nodes:
            # 检查节点资源是否满足请求资源需求
            request_capacity = Capacity(self.request.cpu, self.request.memory, self.request.gpu, self.request.vpu)
            if not node.resource.is_larger_than(request_capacity):
                request_capacity = node.resource.scale_down_to_fit(request_capacity)

            for model_index in range(len(self.isn.models[model_type])):
                selected_model_name = self.isn.get_model_name(model_type, model_index)  # 获取选定模型的名称
                selected_model_info = self.isn.get_model(model_type, selected_model_name)  # 获取选定模型的详细信息

                # 推理时延计算
                if request_capacity.vpu != 0:
                    # 如果请求使用 VPU，则计算 CPU 和 VPU 的混合推理时延
                    inference_latency = (
                            self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (
                            node.cpu_gflops * request_capacity.cpu) +
                            self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (
                                    node.vpu_gflops * request_capacity.vpu)
                    )
                elif request_capacity.gpu != 0:
                    # 如果请求使用 GPU，则计算 CPU 和 GPU 的混合推理时延
                    inference_latency = (
                            self.isn.models[model_type][selected_model_name]['flops'] * 0.2 / (
                            node.cpu_gflops * request_capacity.cpu) +
                            self.isn.models[model_type][selected_model_name]['flops'] * 0.8 / (
                                    node.gpu_gflops * request_capacity.gpu)
                    )
                else:
                    # 如果只使用 CPU，则直接计算 CPU 的推理时延
                    inference_latency = self.isn.models[model_type][selected_model_name]['flops'] / (
                            node.cpu_gflops * request_capacity.cpu)

                # 计算传输时延
                transport = self.isn.find_optimal_path(self.request.hostname, node.hostname)  # 查找最优路径
                transport_latency = transport['total_latency'] * 2 + self.request.datasize / transport['min_bandwidth']  # 计算总传输时延

                latency = inference_latency + transport_latency  # 总时延 = 推理时延 + 传输时延

                # 模型的准确率
                model_accuracy = selected_model_info[get_headers_by_type(self.request.req_type)[0]]  # 根据请求类型获取模型准确率

    
                # 如果存在满足条件的模型，记录该节点和模型
                if model_accuracy > self.request.accuracy and (self.request.deadline - self.request.arrivetime) > latency:
                    results.append({
                        "node": node.hostname,
                        "models": selected_model_name
                    })
    
        return results
