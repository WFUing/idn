import psutil
import GPUtil
import time
import subprocess


class LocalUtil:
    @staticmethod
    def get_cpu_usage():
        """获取当前的 CPU 使用率"""
        return psutil.cpu_percent(interval=1)

    @staticmethod
    def get_memory_usage():
        """获取当前的内存使用情况"""
        return psutil.virtual_memory().percent

    @staticmethod
    def get_gpu_usage():
        """获取当前的 GPU 使用率"""
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100  # 获取第一个 GPU 的使用率
        return 0.0

    @staticmethod
    def get_network_bandwidth():
        """计算当前的网络带宽使用情况"""
        net_io_1 = psutil.net_io_counters()  # 获取网络 I/O 初始数据
        time.sleep(1)  # 1 秒间隔
        net_io_2 = psutil.net_io_counters()  # 获取网络 I/O 结束数据

        # 计算上传和下载带宽（以 Mbps 为单位）
        upload_speed = (net_io_2.bytes_sent - net_io_1.bytes_sent) / (1024 * 1024) * 8  # 上传速度 Mbps
        download_speed = (net_io_2.bytes_recv - net_io_1.bytes_recv) / (1024 * 1024) * 8  # 下载速度 Mbps
        return upload_speed + download_speed  # 总带宽

    @staticmethod
    def ping_latency(host):
        """通过 ping 命令计算网络延时"""
        try:
            output = subprocess.check_output(['ping', '-c', '1', host], stderr=subprocess.STDOUT,
                                             universal_newlines=True)
            for line in output.splitlines():
                if "time=" in line:
                    return float(line.split("time=")[1].split(" ")[0])  # 提取 ping 延迟
        except subprocess.CalledProcessError:
            return float('inf')  # 如果 ping 失败，返回无穷大
        return float('inf')
