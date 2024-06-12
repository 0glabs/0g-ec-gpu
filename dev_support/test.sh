#!/bin/bash

echoStep() {
    echo -e "\n\033[1;34m────────────────────────────────────────────────────────"
    echo -e "\033[1;34m$1."
    echo -e "\033[1;34m────────────────────────────────────────────────────────\033[0m"
}

chmod +x ./dev_support/check_cuda.sh
./dev_support/check_cuda.sh
CUDA_TEST_EXITCODE=$?
if [[ $CUDA_TEST_EXITCODE -ne 0 ]]; then
    echo ""
    echo -e "    \033[1;33mCUDA Environment check fails, some CUDA related tests will be delete\033[0m"
    echo ""
fi


set -e


echoStep "Check fmt"
./cargo_fmt.sh -- --check


export RUSTFLAGS="-D warnings" 
if [[ $CUDA_TEST_EXITCODE -ne 80 ]]; then
    echoStep "Check all (cuda)"
    cargo check --all
    echoStep "Check all tests (cuda)"
    cargo check --all --tests --benches

    echoStep "Check clippy (cuda)"
    cargo clippy
fi

mkdir "./gpu_build/"
export OUT_DIR="../gpu_build/"

if [[ $CUDA_TEST_EXITCODE -eq 0 ]]; then
    echoStep "Test (cuda, bn254)"
    cargo test -r --all
    echoStep "Test (cuda, bls12-381)"
    cargo test -r -p ag-cuda-ec --no-default-features --features ag-cuda-ec/bls12-381

    echoStep "Bench (cuda, bn254)"
    cargo bench
fi

rm -rf "./gpu_build/"