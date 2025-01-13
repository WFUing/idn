import time
from collections import defaultdict


class IsnRequest:
    def __init__(self, req_type, arrivetime, deadline, accuracy, hostname, datasize=0, cpu=0, gpu=0, vpu=0, memory=0):
        """
        初始化请求记录
        :param req_type: 请求类型 (int)
        :param arrivetime: 请求到达时间 (int64，时间戳，单位为毫秒)
        :param deadline: 请求的期望截止时间 (int64，时间戳，单位为毫秒)
        :param accuracy: 期望准确率 (float)
        :param hostname: 发送请求的主机名 (string)
        """
        self.req_type = int(req_type) # 请求类型
        self.arrivetime = arrivetime  # 到达时间
        self.deadline = deadline      # 期望截止时间
        self.accuracy = accuracy      # 期望准确率
        self.hostname = hostname      # 发送请求的主机名
        self.datasize = datasize      # 发送的数据大小
        self.cpu = cpu                # 预期的cpu核数
        self.gpu = gpu                # 预期的gpu核数 
        self.vpu = vpu                # 预期的vpu核数 
        self.memory = memory          # 预期的memory大小

    def __repr__(self):
        return f"RequestRecord(type={self.req_type}, arrivetime={self.arrivetime}, " \
               f"deadline={self.deadline}, accuracy={self.accuracy}, hostname={self.hostname})"


class RequestRecords:
    def __init__(self):
        """
        初始化请求记录，按小时统计每种请求类型的数量及其发送到的主机。
        """
        # 使用一个字典来按每小时存储不同类型的请求和主机名的数量
        self.records = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    def get_hourly_key(self, timestamp):
        """
        将时间戳转换为每小时的时间段（精确到小时的时间戳）。
        :param timestamp: 请求的时间戳，单位为毫秒
        :return: 精确到小时的时间戳
        """
        # 将毫秒时间戳转换为秒，并去除分钟和秒部分
        return timestamp // 3600000 * 3600000  # 转换为精确到小时的时间戳

    def add_request(self, request):
        """
        添加一个请求到统计中
        :param request: RequestRecord 实例
        """
        # 获取该请求的到达时间并计算对应的每小时时间段
        hour_key = self.get_hourly_key(request.arrivetime)

        # 更新记录
        self.records[hour_key][request.req_type][request.hostname] += 1

    def predict_request(self, current_timestamp=None, y=0.6, window_size=3):
        """
        使用公式 L0 + L1*y + L2*y^2 + ... 预测未来某小时的请求繁忙程度。
        :param current_timestamp: 当前时间的时间戳，单位为毫秒
        :param y: 衰减系数
        :param window_size: 用于计算预测公式的窗口大小（最近的小时数）
        :return: 字典，表示预测的请求数量按类型和主机名分布
        """
        prediction = defaultdict(lambda: defaultdict(float))

        # 当前时间戳的小时键
        import time
        if current_timestamp is None:
            current_timestamp = int(time.time() * 1000)

        # 当前时间戳的小时键
        current_hour_key = self.get_hourly_key(current_timestamp)

        # 累计统计过去 window_size 小时的数据
        for i in range(window_size):
            past_hour_key = current_hour_key - i * 3600000
            if past_hour_key in self.records:
                for req_type, host_data in self.records[past_hour_key].items():
                    for hostname, count in host_data.items():
                        # 公式: L0 + L1*y + L2*y^2 + ...
                        prediction[req_type][hostname] += count * (y ** (i + 1))
        # 返回预测结果
        return dict(prediction)

    def get_stats(self):
        """
        获取当前统计信息
        :return: 返回统计信息（按小时、请求类型和主机名分类）
        """
        return dict(self.records)

    def __repr__(self):
        # 打印请求记录的统计信息
        return f"RequestRecords(stats={self.get_stats()})"

# 示例用法：

# 创建 RequestRecords 实例
# request_records = RequestRecords()

# # 模拟一些请求数据
# request1 = Request(req_type=1, arrivetime=int(time.time() * 1000), deadline=int(time.time() * 1000) + 10000, accuracy=0.95, hostname="host1")
# time.sleep(1)
# request2 = Request(req_type=2, arrivetime=int(time.time() * 1000), deadline=int(time.time() * 1000) + 20000, accuracy=0.85, hostname="host2")
# time.sleep(1)
# request3 = Request(req_type=1, arrivetime=int(time.time() * 1000), deadline=int(time.time() * 1000) + 15000, accuracy=0.92, hostname="host1")
# time.sleep(1)
# request4 = Request(req_type=2, arrivetime=int(time.time() * 1000), deadline=int(time.time() * 1000) + 25000, accuracy=0.88, hostname="host3")

# # 添加请求到记录中
# request_records.add_request(request1)
# request_records.add_request(request2)
# request_records.add_request(request3)
# request_records.add_request(request4)

# # 打印统计信息
# print(request_records)

# # 测试预测方法
# prediction = request_records.predict_request(y=0.6, window_size=3)
# print("Prediction:", prediction)
