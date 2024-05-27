use ark_ff::PrimeField;
pub use chosen_ark_suite::{
    Fq as Base, Fr as Scalar, G1Affine, G1Projective as Curve,
};

macro_rules! impl_kernel_wrapper {
    ($name:ident) => {
        #[cfg(feature = "cuda")]
        impl rust_gpu_tools::cuda::KernelArgument for $name {
            fn as_c_void(&self) -> *mut std::ffi::c_void {
                &self.0 as *const _ as _
            }
        }

        #[cfg(feature = "opencl")]
        impl rust_gpu_tools::opencl::KernelArgument for $name {
            fn push(&self, kernel: &mut rust_gpu_tools::opencl::Kernel) {
                unsafe { kernel.builder.set_arg(&self.0) };
            }
        }
    };
}

#[derive(Default, PartialEq, Debug, Clone, Copy)]
#[repr(transparent)]
pub struct GpuScalar(pub Scalar);
impl_kernel_wrapper!(GpuScalar);

#[repr(transparent)]
pub struct GpuCurve(pub Curve);
impl_kernel_wrapper!(GpuCurve);

#[repr(transparent)]
pub struct GpuBigInt(pub <Scalar as PrimeField>::BigInt);
impl_kernel_wrapper!(GpuBigInt);
