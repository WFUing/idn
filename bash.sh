#!/bin/bash

# 创建恢复目录
RESTORE_DIR="restored_files"
mkdir -p "$RESTORE_DIR"

# 提取所有对象的元数据
echo "提取所有对象..."
git verify-pack -v .git/objects/pack/*.idx > objects_metadata.txt

# 过滤出 blob 和 tree 对象
blob_objects=$(grep " blob " objects_metadata.txt | awk '{print $1}')
tree_objects=$(grep " tree " objects_metadata.txt | awk '{print $1}')

# 恢复 blob 对象到文件（文件内容单独提取）
echo "恢复 blob 对象..."
for blob in $blob_objects; do
    # 输出文件内容到 RESTORE_DIR 中的临时文件
    git cat-file -p "$blob" > "$RESTORE_DIR/$blob"
done

# 递归解析 tree 对象并构建目录结构
function process_tree() {
    local tree_object=$1
    local base_path=$2

    # 解析 tree 对象内容
    git cat-file -p "$tree_object" | while read -r mode type object name; do
        if [[ $type == "blob" ]]; then
            # 是文件，恢复内容
            mkdir -p "$base_path"
            echo "恢复文件: $base_path/$name"
            mv "$RESTORE_DIR/$object" "$base_path/$name" 2>/dev/null || {
                # 如果文件已经存在，则直接写入
                git cat-file -p "$object" > "$base_path/$name"
            }
        elif [[ $type == "tree" ]]; then
            # 是子目录，递归解析
            echo "进入目录: $base_path/$name"
            process_tree "$object" "$base_path/$name"
        fi
    done
}

# 获取所有 commit 对象并处理 tree
echo "解析 commit 对象..."
commit_objects=$(grep " commit " objects_metadata.txt | awk '{print $1}')
for commit in $commit_objects; do
    echo "处理 commit: $commit"

    # 获取根 tree 对象
    root_tree=$(git cat-file -p "$commit" | grep "tree" | awk '{print $2}')

    # 从根 tree 开始恢复
    process_tree "$root_tree" "$RESTORE_DIR"
done

echo "所有文件已成功恢复到 $RESTORE_DIR 目录中。"
