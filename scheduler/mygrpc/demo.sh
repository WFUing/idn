# proto代码生成 

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. scheduler.proto

redis-cli keys "type:*"

#请你设计10个节点，8个边端节点，2个云端节点，其中设计两个二维矩阵，一个矩阵记录节点间的传输时延，另一个记录节点之间的带宽信息，要符合元边端的条件，体现出边缘端节点之间距离的差异写在两个csv中，以latency.csv和bandwidth.csv命名