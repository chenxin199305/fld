#!/usr/bin/env bash

set -e

# ----------------------------------------------------------------------------------------------------

# Install Git LFS (Large File Storage)
echo "Installing Git LFS (Large File Storage)..."
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install git-lfs -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-lfs
    else
        echo "Unsupported OS. Please install Git LFS manually."
        exit 1
    fi
else
    echo "Git LFS is already installed."
fi

git lfs install
echo "Git LFS installation complete."
echo ""
