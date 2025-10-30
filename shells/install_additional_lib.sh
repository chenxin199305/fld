#!/usr/bin/env bash

set -e  # Exit immediately if a command exits with a non-zero status

# ----------------------------------------------------------------------------------------------------

echo "Installing additional libraries..."

pip install numpy==1.20.0
pip install tensorboard==2.14.0
pip install protobuf==3.20.1

echo "Additional libraries installed successfully."
echo ""

exit 0  # Exit the script successfully