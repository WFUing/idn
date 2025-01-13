import grpc
from concurrent import futures
from mygrpc import scheduler_pb2_grpc
from mygrpc import scheduler_pb2 
from scheduler.core.request import IsnRequest, RequestRecords
from core.isn import InferenceServiceNet


class SchedulerServicer(scheduler_pb2_grpc.SchedulerServicer):

    def __init__(self, isn):
        self.isn = isn

    def Schedule(self, request, context):

        request = IsnRequest(request.type, request.arrivetime, request.deadline, request.accuracy, request.hostname, request.datasize)
        self.isn.add_request(request)

        # 创建 Response 实例
        response = scheduler_pb2.Response()



        # 添加第一个 Deployment
        deployment = response.deployments.add()
        deployment.hostname = "edge1"
        deployment.model = "resnet18"
        return response


def run_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    isn = InferenceServiceNet()
    scheduler_pb2_grpc.add_SchedulerServicer_to_server(SchedulerServicer(isn), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    run_server()