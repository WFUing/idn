import csv
import redis

# 连接到 Redis
r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# 读取当前目录中的 CSV 文件
accuracy_filename = '/home/wds/zhitai/graduate/idn/scheduler/utils/accuracy.csv'
flops_filename = '/home/wds/zhitai/graduate/idn/scheduler/utils/Model_FLOPS.csv'
latency_filename = '/home/wds/zhitai/graduate/idn/scheduler/utils/latency.csv'
bandwidth_filename = '/home/wds/zhitai/graduate/idn/scheduler/utils/bandwidth.csv'
resources_filename = '/home/wds/zhitai/graduate/idn/scheduler/utils/resources.csv'

def import_accuracy_data():

    # 读取 CSV 文件并将数据存储到 Redis
    with open(accuracy_filename, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # 逐行读取 CSV 数据
        for row in reader:
            # 获取 'type' 作为 Redis 键
            key = f"type:{row['type']}"
            
            # 根据表头处理字段和值
            # 我们会遍历所有字段，并根据字段名选择准确率列
            for column in row:
                if column == 'type':
                    continue  # 跳过 'type' 字段
                if column == 'name':
                    continue  # 跳过 'type' 字段
                
                # 根据不同的准确率字段来选择值
                value = row[column] if row[column] else None
                if value:  # 如果该字段有值
                    field = f"{row['name']}:{column}"
                    r.hset(key, field, value)

def import_flops_data():
    # 读取 CSV 文件并将数据存储到 Redis
    with open(flops_filename, mode='r') as csvfile:
        grouped_data = csv.DictReader(csvfile)
        # 逐行读取 CSV 数据
        for row in grouped_data:
            redis_key = f"type:{row['type']}"
            field_map = {f"{row['name']}:flops": row['flops']}
            r.hset(redis_key, mapping=field_map)

def import_latency_data():
    # 加载 latency.csv
    with open(latency_filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            source = row[0]
            # 添加到节点集合
            r.sadd('nodes', source)
            # 存储延迟数据
            data = {headers[i]: row[i] for i in range(1, len(headers))}
            r.hset(f'latency:{source}', mapping=data)  # 使用 `mapping` 参数


def import_bandwidth_data():
    # 加载 latency.csv
    with open(bandwidth_filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            source = row[0]
            # 添加到节点集合
            r.sadd('nodes', source)
            # 存储延迟数据
            data = {headers[i]: row[i] for i in range(1, len(headers))}
            r.hset(f'bandwidth:{source}', mapping=data)

def import_resources_data():
    # 加载 resources.csv
    with open(resources_filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            node = row[0]
            # 存储资源数据
            data = {headers[i]: row[i] for i in range(1, len(headers))}
            r.hset(f'resources:{node}', mapping=data)

# 新增节点
def add_node(node, latency_data, bandwidth_data, resource_data):
    # 添加到节点集合
    r.sadd('nodes', node)
    
    # 添加延迟数据
    r.hmset(f'latency:{node}', latency_data)
    for target_node, latency in latency_data.items():
        r.hset(f'latency:{target_node}', node, latency)

    # 添加带宽数据
    r.hmset(f'bandwidth:{node}', bandwidth_data)
    for target_node, bandwidth in bandwidth_data.items():
        r.hset(f'bandwidth:{target_node}', node, bandwidth)

    # 添加资源数据
    r.hset(f'resources:{node}', resource_data)

import_accuracy_data()
import_flops_data()
import_latency_data()
import_bandwidth_data()
import_resources_data()

print("Data has been added to Redis successfully!")
