from enum import Enum


class RequestType(Enum):
    # 定义一些深度学习请求类型
    IMAGE_RECOGNITION = 1  # 图像识别
    OBJECT_DETECTION = 2  # 目标检测
    SEMANTIC_SEGMENTATION = 3  # 语义分割
    INSTANCE_SEGMENTATION = 4  # 实例分割
    TEXT_RECOGNITION = 5  # 文本识别
    FACE_DETECTION = 6  # 人脸检测
    IMAGE_GENERATION = 7  # 图像生成

    @classmethod
    def get_name(cls, value):
        """通过整数获取名称"""
        for member in cls:
            if member.value == value:
                return member.name
        return None

    @classmethod
    def get_value(cls, name):
        """通过名称获取整数"""
        try:
            return cls[name].value
        except KeyError:
            return None


class TypeMetrics(Enum):
    # 定义 type 和指标表头的对应关系
    IMAGE_RECOGNITION = ["top_1", "top_5"]
    OBJECT_DETECTION = ["mAP"]
    SEMANTIC_SEGMENTATION = ["mIoU"]
    INSTANCE_SEGMENTATION = ["AP"]
    TEXT_RECOGNITION = ["CA"]
    FACE_DETECTION = ["Precision"]
    IMAGE_GENERATION = ["FID"]

    @classmethod
    def get_metrics(cls, request_type):
        """
        通过请求类型获取对应的指标表头
        :param request_type: RequestType 的枚举值
        :return: 对应的指标表头列表
        """
        if not isinstance(request_type, RequestType):
            raise ValueError("request_type must be an instance of RequestType")
        
        for metric_type in cls:
            if metric_type.name == request_type.name:
                return metric_type.value
        return None

    @classmethod
    def get_request_type(cls, metric_name):
        """
        通过指标表头获取对应的请求类型
        :param metric_name: 指标表头名称
        :return: 对应的 RequestType 枚举值列表
        """
        request_types = []
        for metric_type in cls:
            if metric_name in metric_type.value:
                request_types.append(metric_type.name)
        return request_types if request_types else None

def get_headers_by_type(type_value):
    """
    根据类型的整数值获取对应的指标表头。
    :param type_value: int, RequestType 的整数值
    :return: list, 对应的指标表头列表
    """
    # 获取对应的 RequestType
    request_type_name = RequestType.get_name(type_value)
    if not request_type_name:
        return None
    
    # 将名称转换为 RequestType 实例
    request_type = RequestType[request_type_name]
    
    # 获取对应的指标表头
    return TypeMetrics.get_metrics(request_type)

# 示例用法
if __name__ == "__main__":

    print(get_headers_by_type(1))
    print(RequestType.get_name(1))  # 输出: IMAGE_RECOGNITION
    print(RequestType.get_value("OBJECT_DETECTION"))  # 输出: 2
    print(RequestType.get_name(99))  # 输出: None
    print(RequestType.get_value("NON_EXISTENT"))  # 输出: None

    # 通过请求类型获取指标表头
    print(TypeMetrics.get_metrics(RequestType.IMAGE_RECOGNITION))  # 输出: ['top_1', 'top_5']
    print(TypeMetrics.get_metrics(RequestType.OBJECT_DETECTION))  # 输出: ['mAP']
    print(TypeMetrics.get_metrics(RequestType.IMAGE_GENERATION))  # 输出: ['FID']

    # 通过指标表头获取请求类型
    print(TypeMetrics.get_request_type("top_1"))  # 输出: RequestType.IMAGE_RECOGNITION
    print(TypeMetrics.get_request_type("mAP"))  # 输出: RequestType.OBJECT_DETECTION
    print(TypeMetrics.get_request_type("FID"))  # 输出: RequestType.IMAGE_GENERATION
    print(TypeMetrics.get_request_type("non_existent"))  # 输出: None