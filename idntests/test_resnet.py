import torch

from idntests.models.resnet import resnet50

if __name__ == "__main__":
    # 使用 ResNet-50 模型
    model = resnet50(num_classes=1000)

    # 生成随机输入 (32个样本，3个通道，224x224的图像)
    input_tensor = torch.randn(32, 3, 224, 224)

    # 进行前向传播
    output = model(input_tensor)

    # 打印输出形状
    print(output.shape)  # 期望输出: (32, 1000)
