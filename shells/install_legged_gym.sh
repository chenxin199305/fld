#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

# ----------------------------------------------------------------------------------------------------

echo "Installing legged-gym..."

# Install Isaac Gym
pip install -e ./legged_gym -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

echo "legged-gym installed successfully."
echo ""

exit 0  # Exit the script successfully
