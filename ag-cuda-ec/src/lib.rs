pub mod fft;
pub mod multiexp;
pub mod pairing_suite;
pub mod test_tools;

pub mod ec_fft {
    // Re-export ec-fft for back-forward compatible.
    pub use crate::fft::{radix_ec_fft, radix_ec_fft_mt, radix_ec_fft_st};
}

use ag_cuda_proxy::CudaWorkspace;
use ag_cuda_workspace_macro::construct_workspace;

pub use ag_cuda_proxy::DeviceData;

const FATBIN: &'static [u8] =
    include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));

construct_workspace!(|| CudaWorkspace::from_bytes(FATBIN).unwrap());
