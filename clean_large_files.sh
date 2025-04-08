#!/bin/bash

# === 设置你的 GitHub 仓库地址（建议用 SSH，免密登录） ===
REMOTE_URL="git@github.com:Jiao-xy/AIGC_code.git"

# === 待清除的超大文件路径（相对于仓库根目录）===
LARGE_FILES=(
  "my_project/data/train_shuffled_curriculum.jsonl"
  "ReoraganizationTraining/ModelTest/your_cache_dir/models--google--ul2/blobs/6226f933c6950d796635e243250c47fca09411a8a540903064708301b7ab8b64.incomplete"
)

echo "✅ [1] 正在检查 git-filter-repo 是否安装..."
pip show git-filter-repo &>/dev/null
if [ $? -ne 0 ]; then
  echo "🔧 安装 git-filter-repo..."
  pip install git-filter-repo
fi

# === 备份当前仓库 ===
echo "✅ [2] 正在备份当前仓库..."
BACKUP_NAME="../backup_$(basename $PWD)_$(date +%Y%m%d_%H%M%S).tar.gz"
git gc --prune=now
tar -czf "$BACKUP_NAME" .
echo "🗂️  已备份为: $BACKUP_NAME"

# === 清理大文件 ===
echo "✅ [3] 正在从 Git 历史中移除指定大文件..."
FILTER_CMD="git filter-repo --force"
for FILE in "${LARGE_FILES[@]}"; do
  FILTER_CMD+=" --path \"$FILE\" --invert-paths"
done
eval "$FILTER_CMD"

# === 恢复远程仓库并 push ===
echo "✅ [4] 正在重新添加远程仓库 origin..."
git remote add origin "$REMOTE_URL"

echo "✅ [5] 正在强制推送到远程 main 分支..."
git push -u origin main --force

echo "🎉 完成！大文件已清除，代码已推送。"
