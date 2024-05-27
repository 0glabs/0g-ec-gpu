mod context;
mod ctx_stack_guard;
mod kernel;
mod module;
mod params;

pub use kernel::KernelConfig;
pub use module::{ActiveWorkspace, CudaWorkspace};
pub use params::{DeviceData, DeviceParam, ParamIO};

pub fn cuda_init() {
    use rustacuda::{init, CudaFlags};
    use std::sync::Once;

    static CUDA_INIT: Once = Once::new();
    CUDA_INIT.call_once(|| {
        init(CudaFlags::empty()).unwrap();
    });
}
