# GPU-Accelerated Elliptic Curve Calculations in 0g

This project is a module for GPU-accelerated elliptic curve calculations for the 0g projects. The core elliptic operations in CUDA are sourced from the open-source project [ec-gpu](https://github.com/filecoin-project/ec-gpu).

## Project Structure

- **ag-build**: Generates and compiles GPU-related functions.
- **ag-types**: Defines interfaces for ag-build to adapt to different elliptic curves and finite fields.
- **ag-cuda-proxy**: An intermediary layer for CPU-GPU communication, abstracting CUDA context mainAPI calls.
- **ag-cuda-workspace-macro**: A procedural macro tool used in conjunction with `ag-cuda-proxy`.
- **ag-cuda-ec**: GPU-accelerated computation module required by the 0g project.

## Project Development

### Compilation Environment

Set up the CUDA build cache directory to reduce the number of nvcc compilations:

```sh
mkdir .cache
export ARK_GPU_BUILD_DIR=$(pwd)/.cache
```

### Writing CUDA Kernel Functions

1. Write CUDA kernel functions in the `ag-build/cl` directory.

2. Add new files to `SourceBuilder`, as seen in `SourceBuilder::add_ec_fft` and `EcFft<C>` in [the example code](ag-build/src/source/builder.rs).

### Writing Rust Proxy Functions to Call GPU Using `ag-cuda-proxy`

1. Add GPU code compilation to the Rust build process. In the `build.rs` of the crate containing the proxy functions, include the newly added CL code in `SourceBuilder`. An example: 

    ```rust
    use ag_build::{generate, SourceBuilder};

    fn main() {
        let source = SourceBuilder::new()
            .add_ec_fft::<ark_bls12_381::G1Affine>()
            .add_multiexp::<ark_bls12_381::G1Affine>()
            .add_ec_fft::<ark_bn254::G1Affine>()
            .add_multiexp::<ark_bn254::G1Affine>();

        generate(&source);
    }
    ```

2. Construct the workspace by loading the compiled GPU binary code.
    ```rust
    use ag_cuda_proxy::CudaWorkspace;
    use ag_cuda_workspace_macro::construct_workspace;

    const FATBIN: &'static [u8] =
        include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));

    construct_workspace!(|| CudaWorkspace::from_bytes(FATBIN).unwrap());
    ```

3. Define a function with the first parameter as `workspace`, for example:

    ```rust
    use ag_cuda_workspace_macro::auto_workspace;
    use ag_cuda_proxy::ActiveWorkspace;

    #[auto_workspace]
    pub fn radix_ec_fft(
        workspace: &ActiveWorkspace, input: &mut Vec<Curve>, omegas: &[Scalar],
    )
    ```

4. Call the kernel function, as seen in the creation and usage of `kernel` in [the example code](ag-cuda-ec/src/ec_fft.rs).

### Compiling and Calling GPU Functions

1. In the example above, ag-ec-proxy adds two new functions on top of `radix_ec_fft`: `radix_ec_fft_st` and `radix_ec_fft_mt`. The `_st` suffix represents single-threaded mode, using a global unique CUDA context, while `_mt` suffix represents multi-threaded mode, where each thread has its own CUDA context. (Note: there are bugs when running multi-threaded mode concurrently).

   For the original function:

    ```rust
    pub fn radix_ec_fft(
        workspace: &ActiveWorkspace, input: &mut Vec<Curve>, omegas: &[Scalar],
    )
    ```

   The modified functions without `ActiveWorkspace` parameter are:

    ```rust
    pub fn radix_ec_fft_st(
        input: &mut Vec<Curve>, omegas: &[Scalar],
    )
    ```

2. Warmup

   CUDA context initializes lazily, increasing the time cost of the first call. Pre-initialization can be done using:
   - `ag_ec_proxy::init_global_workspace` (for `_st` functions)
   - `ag_ec_proxy::init_local_workspace` (for `_mt` functions)

## Developed Functions

### Elliptic Curve FFT

```rust
pub fn radix_ec_fft(
    workspace: &ActiveWorkspace, input: &mut Vec<Curve>, omegas: &[Scalar],
)
```

Takes bn254 or bls12-381 G1 projective points as input (from the arkworks library) and a list of FFT omega powers \([w^{2^0}, w^{2^1}, ...]\), and computes the FFT.

    

### Multi-base Multi-scalar Multiplication

```rust
pub fn multiple_multiexp(
    workspace: &ActiveWorkspace, bases_gpu: &DeviceData,
    exponents: &[<Scalar as PrimeFieldRepr>::Repr], num_chunks: usize,
    window_size: usize, neg_is_cheap: bool,
)
```

Takes multiple rows of bn254 or bls12-381 G1 affine points as input and a row of exponents (G1 group's scalar field, non-Montgomery form). Outputs a row of MSM results for the input and each row of G1 affine points.
- G1 affine points need to be uploaded to GPU memory in advance via `upload_multiexp_bases` as they are repeatedly used in actual operations.
- Additional parameters:
    - `num_chunks`: Number of chunks the task is divided into, each chunk completed by a set of CUDA kernels.
    - `window_size`: Parameter for the MSM algorithm, generally between 2 and 10. Larger values might exceed GPU memory.
    - `neg_is_cheap`: Indicates if computing the negation of a G1 point is easy.

## Open Source Project Derivative Work Statement

The GPU code for prime field and elliptic curve operations is derived from [ec-gpu](https://github.com/filecoin-project/ec-gpu), with the following modifications:

- Original project only supports the bls12-381 curve; modifications add support for the bn254 curve.
- Original project uses zkcrypto's cryptographic library; modifications use arkworks.
- `ec-fft` is implemented based on the original `field-fft`, and `multiexp` has a rewritten algorithm.
- Original project supports both OpenCL and CUDA; this project fully tests only the CUDA part, with no adaptations or tests for OpenCL.

# Licenses

This project, including the modified parts of the derivative, has not provided any license yet. 
