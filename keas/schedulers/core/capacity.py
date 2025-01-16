
class Capacity:
    def __init__(self, cpu, memory, gpu=0):
        """
        初始化 Capacity 实例，表示一个节点的资源容量。
        
        :param cpu: CPU 核数
        :param memory: 内存大小，单位为 GB
        :param gpu: GPU 核数，默认为 0
        :param vpu: VPU 核数，默认为 0
        """
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu

    def __sub__(self, other):
        """
        实现减法运算符重载，用于计算两个 Capacity 对象的差异。
        
        :param other: 另一个 Capacity 对象
        :return: 返回两个 Capacity 对象相减后的新 Capacity 对象
        """
        if not isinstance(other, Capacity):
            raise TypeError("Operands must be instances of the 'Capacity' class.")
        
        # 执行减法操作，确保每个属性都能正确相减
        return Capacity(
            self.cpu - other.cpu,
            self.memory - other.memory,
            self.gpu - other.gpu
        )

    def __add__(self, other):
        """
        实现加法运算符重载，用于计算两个 Capacity 对象的和。
        
        :param other: 另一个 Capacity 对象
        :return: 返回两个 Capacity 对象相加后的新 Capacity 对象
        """
        if not isinstance(other, Capacity):
            raise TypeError("Operands must be instances of the 'Capacity' class.")
        
        # 执行加法操作，确保每个属性都能正确相加
        return Capacity(
            self.cpu + other.cpu,
            self.memory + other.memory,
            self.gpu + other.gpu
        )

    def scale_down_to_fit(self, other):
        """
        按比例缩小当前 Capacity 对象以适应目标 Capacity 对象。
        
        :param other: 目标 Capacity 对象
        :return: 按比例缩小后的新 Capacity 对象
        """
        if not isinstance(other, Capacity):
            raise TypeError("Operands must be instances of the 'Capacity' class.")

        # 计算每种资源的缩小比例，不能用 memory 缩放
        ratios = []
        if other.cpu > 0 and self.cpu != 0:
            ratios.append(self.cpu / other.cpu)
        if other.gpu > 0 and self.gpu != 0:
            ratios.append(self.gpu / other.gpu)

        # 取最小比例作为缩小系数
        scale_factor = min(ratios) if ratios else 1

        # 按比例缩小
        return Capacity(
            self.cpu * scale_factor,
            self.memory,
            self.gpu * scale_factor
        )

    def is_larger_than(self, other):
        """
        比较当前 Capacity 是否大于目标 Capacity 对象。
        
        :param other: 目标 Capacity 对象
        :return: 如果当前 Capacity 大于或等于目标 Capacity 则返回 True，否则返回 False
        """
        if not isinstance(other, Capacity):
            raise TypeError("Operands must be instances of the 'Capacity' class.")

        return (
            self.cpu >= other.cpu and
            self.memory >= other.memory and
            self.gpu >= other.gpu
        )

    def __repr__(self):
        """
        定义 Capacity 对象的字符串表示，用于打印时显示资源的分配情况。
        
        :return: Capacity 对象的字符串表示
        """
        return f"Capacity(cpu={self.cpu}, memory={self.memory}, gpu={self.gpu})"

