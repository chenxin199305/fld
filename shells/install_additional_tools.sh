#!/usr/bin/env bash

set -e

# ----------------------------------------------------------------------------------------------------

# Install the pre-commit hooks
echo "Installing pre-commit hooks..."

pip install pre-commit --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple || {
    echo "Failed to install pre-commit. Please check your internet connection or package configuration."
    exit 1
}

pre-commit install || {
    echo "Failed to install pre-commit hooks. Please check your pre-commit configuration."
    exit 1
}

echo "Pre-commit hooks installation complete."
echo ""

exit 0  # Exit the script successfully
