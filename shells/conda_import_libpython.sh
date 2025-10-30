#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

# ----------------------------------------------------------------------------------------------------

echo "Current conda environment: $CONDA_PREFIX"

# 检查并创建相关目录
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# 定义文件路径
ACTIVATE_FILE="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
DEACTIVATE_FILE="$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"

# 定义预期的内容
EXPECTED_ACTIVATE_CONTENT="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CONDA_PREFIX/lib"
EXPECTED_DEACTIVATE_CONTENT=$(cat <<'EOL'
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
DIRECTORY_TO_REMOVE=$CONDA_PREFIX/lib
NEW_LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "$DIRECTORY_TO_REMOVE" | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$NEW_LD_LIBRARY_PATH
EOL
)

# 检查并创建 activate 文件
if [ ! -f "$ACTIVATE_FILE" ]; then
    echo "Creating activate file: $ACTIVATE_FILE"
    echo "$EXPECTED_ACTIVATE_CONTENT" > "$ACTIVATE_FILE"
else
    echo "Remove the old activate file: $ACTIVATE_FILE"
    rm -f "$ACTIVATE_FILE"
    echo "Creating new activate file: $ACTIVATE_FILE"
    echo "$EXPECTED_ACTIVATE_CONTENT" > "$ACTIVATE_FILE"
fi

# 检查并创建 deactivate 文件
if [ ! -f "$DEACTIVATE_FILE" ]; then
    echo "Creating deactivate file: $DEACTIVATE_FILE"
    echo "$EXPECTED_DEACTIVATE_CONTENT" > "$DEACTIVATE_FILE"
else
    echo "Remove the old deactivate file: $DEACTIVATE_FILE"
    rm -f "$DEACTIVATE_FILE"
    echo "Creating new deactivate file: $DEACTIVATE_FILE"
    echo "$EXPECTED_DEACTIVATE_CONTENT" > "$DEACTIVATE_FILE"
fi

# 输出结果
echo ""
echo "Environment variables for conda environment '$CONDA_PREFIX' have been set up successfully."
echo "Activate script created at: $ACTIVATE_FILE"
echo "Deactivate script created at: $DEACTIVATE_FILE"
echo ""

exit 0  # Exit the script successfully
