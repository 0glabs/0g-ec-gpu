#!/bin/bash

echoStep() {
    echo -e "\n\033[1;34m────────────────────────────────────────────────────────"
    echo -e "\033[1;34m$1."
    echo -e "\033[1;34m────────────────────────────────────────────────────────\033[0m"
}

chmod +x ./dev_support/check_cuda.sh
./dev_support/check_cuda.sh
CUDA_TEST_EXITCODE=$?

if [[ $CUDA_TEST_EXITCODE -eq 80 ]]; then
    echo -e "Cannot continue because of no cuda environment"
    exit 1
fi

if [[ $CUDA_TEST_EXITCODE -ne 0 ]]; then
    echo ""
    echo -e "    \033[1;33mCUDA Environment check fails, some CUDA related tests will be skipped\033[0m"
    echo ""
fi


set -e
echoStep "Check fmt"
./cargo_fmt.sh -- --check


mkdir -p .cache
export ARK_GPU_BUILD_DIR="$(pwd)/.cache"

export RUSTFLAGS="-D warnings" 

echoStep "Check all"
cargo check --all
echoStep "Check all tests"
cargo check --all --tests --benches

echoStep "Check clippy"
cargo clippy

if [[ $CUDA_TEST_EXITCODE -eq 0 ]]; then
    echoStep "Test (bn254)"
    cargo test -r --all
    echoStep "Test (bls12-381)"
    cargo test -r -p ag-cuda-ec --no-default-features --features ag-cuda-ec/bls12-381

    # echoStep "Bench (bn254)"
    # cargo bench
fi