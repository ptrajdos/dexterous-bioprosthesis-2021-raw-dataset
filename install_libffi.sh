#!/bin/env bash

# Libffi version and download URL
VERSION=3.3
# Get the major and minor version of Python
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

# Set variable based on version
if [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -eq 9 ]]; then
    VERSION=3.3
elif [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -eq 11 ]]; then
    VERSION=3.5.2
else
    VERSION=3.3
fi

URL="https://github.com/libffi/libffi/releases/download/v${VERSION}/libffi-${VERSION}.tar.gz"


# Download and extract libffi
wget "$URL" -O libffi-${VERSION}.tar.gz
tar xzf libffi-${VERSION}.tar.gz
cd libffi-${VERSION}

# Configure, build, and install
./configure
make -j 4
sudo make install

# Update linker cache (optional, but recommended)
sudo ldconfig 

