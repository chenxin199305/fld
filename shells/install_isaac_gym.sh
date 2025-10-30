#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

# ----------------------------------------------------------------------------------------------------

echo "Installing Isaac Gym..."

# Install Isaac Gym
pip install -e ./IsaacGym_Preview_4_Package/isaacgym/python -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

echo "Isaac Gym installed successfully."
echo ""

exit 0  # Exit the script successfully
