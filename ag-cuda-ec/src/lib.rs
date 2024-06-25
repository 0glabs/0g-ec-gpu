#[cfg(any(feature = "fr-fft", feature = "g1-fft", feature = "g2-fft"))]
pub mod fft;
#[cfg(any(feature = "g1-msm", feature = "g2-msm"))]
pub mod multiexp;
pub mod pairing_suite;
pub mod test_tools;

use ag_cuda_proxy::CudaWorkspace;
use ag_cuda_workspace_macro::construct_workspace;

pub use ag_cuda_proxy::DeviceData;

const FATBIN: &[u8] = include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));

construct_workspace!(|| CudaWorkspace::from_bytes(FATBIN).unwrap());
