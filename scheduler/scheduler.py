# 示例运行
from core.nsga2_scheduler import Nsga2Scheduler
from core.isn import InferenceServiceNet
from core.request import IsnRequest
from core.nsga2_scheduler2 import Nsga2Scheduler2
from core.scheduler import Scheduler

if __name__ == "__main__":
    isn = InferenceServiceNet()  # 初始化你的 ISN 对象
    request = IsnRequest(req_type="1", accuracy=0.3, deadline=1000000, arrivetime=0, datasize=100, cpu = 1, gpu = 1, vpu = 0, memory=1, hostname="edge_1")
    scheduler = Nsga2Scheduler2(isn=isn, request=request)
    population, pareto_front = scheduler.run(population_size=100, ngen=5)
    print(pareto_front)

    # scheduler = Scheduler(isn, request)
    # results = scheduler.get_nodes_and_models_for_request()
    # print(results)