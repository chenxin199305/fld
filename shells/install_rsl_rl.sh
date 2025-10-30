#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

# ----------------------------------------------------------------------------------------------------

echo "Installing rsl-rl library..."

# Install Isaac Gym
pip install -e ./rsl_rl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

echo "rsl-rl library installed successfully."
echo ""

exit 0  # Exit the script successfully
