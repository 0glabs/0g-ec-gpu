use crate::{
    context::CudaContext, ctx_stack_guard::WorkspaceContextGuard, cuda_init,
    kernel::Kernel,
};

use rustacuda::{
    context::{Context, ContextFlags, ContextStack},
    device::Device,
    error::CudaResult,
    module::Module,
    stream::{Stream, StreamFlags},
};

pub struct CudaWorkspace {
    // TODO: support multiple module
    module: Module,
    context: CudaContext,
}

unsafe impl Send for CudaWorkspace {}
unsafe impl Sync for CudaWorkspace {}

impl CudaWorkspace {
    pub fn from_bytes(bytes: &[u8]) -> CudaResult<Self> {
        cuda_init();

        let device = Device::get_device(0)?;

        // Create a context associated to this device
        let ctx = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        )?;

        let maybe_module = Module::load_from_bytes(bytes);
        ContextStack::pop().expect("Cannot remove context.");

        Ok(Self {
            context: CudaContext::new(ctx),
            module: maybe_module?,
        })
    }

    pub fn activate(&self) -> CudaResult<ActiveWorkspace<'_>> {
        let guard = WorkspaceContextGuard::new(&self.context)?;
        Ok(ActiveWorkspace(self, guard))
    }
}

pub struct ActiveWorkspace<'a>(
    &'a CudaWorkspace,
    #[allow(dead_code)] WorkspaceContextGuard<'a>,
);

impl<'a> ActiveWorkspace<'a> {
    pub fn create_kernel(&self) -> CudaResult<Kernel<'a>> {
        Ok(Kernel::new(&self.0.module, self.stream()?))
    }

    pub fn stream(&self) -> CudaResult<Stream> {
        Stream::new(StreamFlags::NON_BLOCKING, None)
    }
}
