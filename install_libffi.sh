#!/bin/env bash

# Libffi version and download URL
VERSION=3.3
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

