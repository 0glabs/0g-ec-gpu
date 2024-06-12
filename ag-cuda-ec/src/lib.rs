pub mod ec_fft;
pub mod multiexp;
pub mod pairing_suite;
pub mod test_tools;

use ag_cuda_proxy::CudaWorkspace;
use ag_cuda_workspace_macro::construct_workspace;

pub use ag_cuda_proxy::DeviceData;

const FATBIN: &[u8] =
    include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));

construct_workspace!(|| CudaWorkspace::from_bytes(FATBIN).unwrap());
