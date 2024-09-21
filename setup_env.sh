#!/bin/bash

PYTHON_PATH=$(which python)

# Get the Python site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

export LIBTORCH=$SITE_PACKAGES/torch

# Set the LD_LIBRARY_PATH to include libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# For macOS, we also need to set DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

echo "LIBTORCH set to: $LIBTORCH"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
echo "DYLD_LIBRARY_PATH set to: $DYLD_LIBRARY_PATH"
echo "LIBTORCH_USE_PYTORCH set to: $LIBTORCH_USE_PYTORCH"
echo "LIBTORCH_BYPASS_VERSION_CHECK set to: $LIBTORCH_BYPASS_VERSION_CHECK"

# Verify that the libtorch_cpu.dylib file exists
if [ -f "$LIBTORCH/lib/libtorch_cpu.dylib" ]; then
    echo "libtorch_cpu.dylib found at $LIBTORCH/lib/libtorch_cpu.dylib"
else
    echo "Error: libtorch_cpu.dylib not found at $LIBTORCH/lib/libtorch_cpu.dylib"
fi