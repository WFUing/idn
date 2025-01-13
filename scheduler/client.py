# client.py

import grpc
from mygrpc import scheduler_pb2
from mygrpc import scheduler_pb2_grpc


def run_client():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = scheduler_pb2_grpc.SchedulerStub(channel)
        response = stub.Schedule(scheduler_pb2.Request(type=1,timestamp=100000,accuracy=99.5))
        # 遍历 deployments 列表
        for deployment in response.deployments:
            print(f"Hostname: {deployment.hostname}, Model: {deployment.model}")


if __name__ == '__main__':
    run_client()