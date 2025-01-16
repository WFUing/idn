import numpy as np

def dep_round(y, model_sizes, budget):
    """
    实现 DepRound 算法

    参数:
    - y: 分数分配向量 (list or numpy array)
    - model_sizes: 每个模型的资源大小 (list or numpy array)
    - budget: 节点的资源预算 (float)

    返回:
    - 离散分配向量 x
    """
    y = np.array(y)
    x = np.zeros_like(y)

    while np.any((y > 0) & (y < 1)):
        # 选取两个分数变量
        indices = np.where((y > 0) & (y < 1))[0]
        if len(indices) < 2:
            break

        i, j = np.random.choice(indices, 2, replace=False)

        # 计算调整范围
        delta = min(y[i], y[j])

        # 随机调整
        if np.random.rand() < y[i] / (y[i] + y[j]):
            y[i] -= delta
            y[j] += delta
        else:
            y[i] += delta
            y[j] -= delta

        # 检查是否取整
        if y[i] == 0 or y[i] == 1:
            x[i] = y[i]
            y[i] = 0
        if y[j] == 0 or y[j] == 1:
            x[j] = y[j]
            y[j] = 0

    # 处理剩余变量
    remaining_index = np.where((y > 0) & (y < 1))[0]
    if len(remaining_index) == 1:
        idx = remaining_index[0]
        x[idx] = 1 if np.random.rand() < y[idx] else 0

    return x

# 示例用法
y = [0.3, 0.5, 0.2]
model_sizes = [1, 2, 1]
budget = 3
x = dep_round(y, model_sizes, budget)
print("离散分配结果:", x)
