import time
from collections import defaultdict


class IsnRequest:
    def __init__(self, time_period, hostname, data_count, data_persize, accuracy=0):
        """
        初始化请求记录
        :param arrivetime: 请求到达时间 (int64，时间戳，单位为毫秒)
        :param deadline: 请求的期望截止时间 (int64，时间戳，单位为毫秒)
        :param accuracy: 期望准确率 (float)
        :param hostname: 发送请求的主机名 (string)
        """
        self.time_period = time_period      # 请求的周期
        self.accuracy = accuracy            # 期望准确率
        self.hostname = hostname            # 发送请求的主机名
        self.data_count = data_count        # 发送的数据数量
        self.data_per_size = data_persize   # 发送的每个数据的大小

    def __repr__(self):
        return f"Request({self.time_period}, {self.accuracy}, {self.hostname}, {self.data_count}, {self.data_per_size})"

