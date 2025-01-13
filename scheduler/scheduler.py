# 示例运行
from core.nsga2_scheduler import Nsga2Scheduler
from core.isn import InferenceServiceNet
from core.request import IsnRequest


if __name__ == "__main__":
    isn = InferenceServiceNet()  # 初始化你的 ISN 对象
    request = IsnRequest(req_type="1", accuracy=0.3, deadline=1000000, arrivetime=0, datasize=1024, cpu = 1, gpu = 1, vpu = 0, memory=1, hostname="edge_1")
    scheduler = Nsga2Scheduler(isn=isn, request=request)
    population, pareto_front = scheduler.run(population_size=10000, ngen=500)
    print(pareto_front)
    scheduler.visualize(pareto_front)
    scheduler.visualize_vars_vs_objectives(pareto_front)