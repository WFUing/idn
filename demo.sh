#!/bin/bash

# 检查是否在 Git 仓库中
if [ ! -d .git ]; then
  echo "当前目录不是一个 Git 仓库！"
  exit 1
fi

# 创建恢复目录
RECOVERY_DIR="recovered_files"
mkdir -p "$RECOVERY_DIR"

echo "开始从 .git 中恢复文件..."

# 查找所有的 dangling blob 对象
git fsck --lost-found | grep 'dangling blob' | awk '{print $3}' | while read -r blob; do
  # 获取 blob 的文件内容
  file_content=$(git cat-file -p "$blob")

  # 获取文件的 SHA1 哈希并生成路径
  file_name=$(echo "$file_content" | sha1sum | awk '{print $1}')
  file_path="$RECOVERY_DIR/$file_name"

  # 将内容写入恢复目录
  echo "$file_content" > "$file_path"
  echo "恢复文件：$file_name -> $file_path"
done

echo "文件恢复完成。所有文件已保存到 $RECOVERY_DIR 目录中。"

# 提示用户下一步操作
echo "您可以手动检查文件并将其移动到合适的目录中。"
