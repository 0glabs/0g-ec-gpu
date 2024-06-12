#!/bin/bash

set +e

cd $(dirname "$0")

cleanup() {
    rm -f test_cuda.cu test_cuda
}

trap cleanup EXIT

# Detect if cuda installed
if ! command -v nvcc >/dev/null 2>&1; then
    echo -e "\033[0;32m[CUDA environment]\033[0m nvcc is not installed."
    exit 80
fi

# 创建一个临时 CUDA 文件
cat <<EOF > test_cuda.cu
#include <stdio.h>

__global__ void helloFromGPU() {
    printf("CUDA test success!\\n");
}

int main() {
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF

nvcc -o test_cuda test_cuda.cu 
if [ $? -ne 0 ]; then
    echo -e "\033[0;32m[CUDA environment]\033[0m nvcc cannot compile."
    exit 80
fi

output=$("./test_cuda")
if [ $? -ne 0 ] || [[ $output != *"CUDA test success!"* ]]; then
    echo -e "\033[0;32m[CUDA environment]\033[0m cuda cannot run."
    exit 81
fi

echo -e "\033[0;32m[CUDA environment]\033[0m check success."
exit 0