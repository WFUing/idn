import redis

# 连接到 Redis
r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# # 清空当前数据库的所有数据
# r.flushdb()
# print("The current database has been cleared.")

def show_models_info():
    # 检查 type:1 到 type:7 是否存在数据
    for i in range(1, 8):
        key = f"type:{i}"
        # 获取该键的所有字段和值
        fields = r.hgetall(key)
        
        print(f"Data for {key}:")
        for field, value in fields.items():
            print(f"  {field}: {value}")
        print()

def show_models():
    models = {}
    
    # 遍历 type:1 到 type:7 获取每种类型的模型数据
    for i in range(1, 8):
        key = f"type:{i}"
        
        type = {}

        # 获取该键的所有字段和值
        fields = r.hgetall(key)
        
        # 格式化数据并保存到 models 字典
        for field, value in fields.items():
            # 获取模型名称与指标名（如 ResNet18:top_1）
            model_name, metric = field.split(":")
            
            # 将值转换为浮动数值
            # print(value)
            value = float(value)
            
            # 确保该模型在 models 字典中已存在
            if model_name not in models:
                type[model_name] = {}
            
            # 保存该指标
            type[model_name][metric] = value

        models[key] = type

    print(models)

def show_nodes():
    nodes = r.smembers('nodes')
    print("All nodes:", nodes)

def show_latency():
    for node in r.smembers('nodes'):
        latency_data = r.hgetall(f'latency:{node}')
        print(f"Latency data for {node}:", latency_data)

def show_bandwidth():
    for node in r.smembers('nodes'):
        bandwidth_data = r.hgetall(f'bandwidth:{node}')
        print(f"Bandwidth data for {node}:", bandwidth_data)

def show_resource():
    for node in r.smembers('nodes'):
        resources_data = r.hgetall(f'resources:{node}')
        print(f"Resources data for {node}:", resources_data)


show_nodes()
# show_latency()
# show_bandwidth()
# show_resource()
# show_models()
show_models_info()