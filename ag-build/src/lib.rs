//! CUDA/OpenCL code generator for finite-field arithmetic over prime fields and
//! elliptic curve arithmetic constructed with Rust.
//!
//! There is also support for Fast Fourier Transform and Multiexponentiation.
//!
//! This crate usually creates GPU kernels at compile-time. CUDA generates a
//! [fatbin], which OpenCL only generates the source code, which is then
//! compiled at run-time.
//!
//! In order to make things easier to use, there are helper functions available.
//! You would put some code into `build.rs`, that generates the kernels, and
//! some code into your library which then consumes those generated kernels. The
//! kernels will be directly embedded into your program/library. If something
//! goes wrong, you will get an error at compile-time.
//!
//! In this example we will make use of the FFT functionality. Add to your
//! `build.rs`:
//!
//!
//! The `ag_build::generate()` takes care of the actual code
//! generation/compilation. It will automatically create a CUDA and/or OpenCL
//! kernel. It will define two environment variables, which are meant for
//! internal use. `_EC_GPU_CUDA_KERNEL_FATBIN` that points to the compiled CUDA
//! kernel, and `_EC_GPU_OPENCL_KERNEL_SOURCE` that points to the generated
//! OpenCL source.
//!
//!
//! Feature flags
//! -------------
//!
//! CUDA and OpenCL are supprted, each be enabled with the `cuda` and `opencl`
//! [feature flags].
//!
//! [fatbin]: https://en.wikipedia.org/wiki/Fat_binary#Heterogeneous_computing
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section

pub use source::SourceBuilder;

mod source;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod compile;

#[cfg(all(test, any(feature = "cuda", feature = "opencl")))]
mod tests;

#[allow(unused_variables)]
pub fn generate(source_builder: &SourceBuilder) {
    #[cfg(feature = "cuda")]
    compile::generate_cuda(source_builder);
    #[cfg(feature = "opencl")]
    compile::generate_opencl(source_builder);
}
