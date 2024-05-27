use crate::{
    params::{DeviceParam, NullPointer},
    DeviceData,
};

use super::params::{Param, ParamIO};

use std::{
    ffi::{c_void, CString},
    iter::once,
};

use rustacuda::{
    error::CudaResult, function::Function, module::Module, stream::Stream,
};

#[cfg(feature = "timer")]
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    pub global_work_size: usize,
    pub local_work_size: usize,
    pub shared_mem: usize,
}

pub struct Kernel<'a> {
    module: &'a Module,
    stream: Stream,
}

impl<'a> Kernel<'a> {
    pub fn new(module: &'a Module, stream: Stream) -> Self {
        Self { module, stream }
    }

    pub fn func<'b>(self, name: &str) -> CudaResult<KernelTask<'a, 'b>> {
        KernelTask::new(self, name)
    }
}

type Params<'b> = Vec<Box<dyn ParamIO + 'b>>;

pub struct KernelTask<'a, 'b> {
    k: Kernel<'a>,
    launched: Vec<Params<'b>>,
    function: Function<'a>,
    args: Params<'b>,
    #[cfg(feature = "timer")]
    instant: Instant,
}

pub struct PendingTask<'a, 'b>(KernelTask<'a, 'b>);

impl<'a, 'b> KernelTask<'a, 'b> {
    pub fn new(kernel: Kernel<'a>, name: &str) -> CudaResult<Self> {
        #[cfg(feature = "timer")]
        let instant = Instant::now();

        let function_name =
            CString::new(name).expect("Kernel name must not contain nul bytes");
        let function = kernel.module.get_function(&function_name)?;

        #[cfg(feature = "timer")]
        println!(
            "[{:?}] Elapsed {:.3} us (get function)",
            std::thread::current().id(),
            instant.elapsed().as_nanos() as f64 / 1000.0,
        );

        Ok(KernelTask {
            k: kernel,
            launched: vec![],
            function,
            args: Vec::new(),
            #[cfg(feature = "timer")]
            instant,
        })
    }

    #[inline]
    #[allow(unused_variables)]
    pub fn elapsed(&self, note: &str) {
        #[cfg(feature = "timer")]
        {
            println!(
                "[{:?}] Elapsed {:.3} us ({})",
                std::thread::current().id(),
                self.instant.elapsed().as_nanos() as f64 / 1000.0,
                note
            );
        }
    }

    pub fn val<T: 'static>(mut self, input: T) -> CudaResult<Self> {
        self.receive_param(Param::InVal(input))?;
        Ok(self)
    }

    pub fn in_ref<T>(mut self, input: &'b T) -> CudaResult<Self> {
        self.receive_param(Param::InRef(input))?;
        Ok(self)
    }

    pub fn in_mut<T>(mut self, input: &'b mut T) -> CudaResult<Self> {
        self.receive_param(Param::InMut(input))?;
        Ok(self)
    }

    pub fn out<T>(mut self, output: &'b mut T) -> CudaResult<Self> {
        self.receive_param(Param::Out(output))?;
        Ok(self)
    }

    pub fn in_ref_slice<T>(mut self, input: &'b [T]) -> CudaResult<Self> {
        self.receive_param(Param::InRefSlice(input))?;
        Ok(self)
    }

    pub fn in_mut_slice<T>(mut self, input: &'b mut [T]) -> CudaResult<Self> {
        self.receive_param(Param::InMutSlice(input))?;
        Ok(self)
    }

    pub fn out_slice<T>(mut self, output: &'b mut [T]) -> CudaResult<Self> {
        self.receive_param(Param::OutSlice(output))?;
        Ok(self)
    }

    pub fn dev_arg<T>(
        mut self, output: &'b DeviceParam<'_, T>,
    ) -> CudaResult<Self> {
        self.args.push(Box::new(output));
        self.elapsed("device param");
        Ok(self)
    }

    pub fn dev_data(mut self, output: &'b DeviceData) -> CudaResult<Self> {
        self.args.push(Box::new(output));
        self.elapsed("device data");
        Ok(self)
    }

    pub fn empty(mut self) -> CudaResult<Self> {
        self.args.push(Box::new(NullPointer));
        self.elapsed("empty param");
        Ok(self)
    }

    fn receive_param<T>(&mut self, arg: Param<'b, T>) -> CudaResult<()> {
        let b = arg.before_call(&self.k.stream)?;
        self.args.push(Box::new((arg, b)));
        self.elapsed("param");
        Ok(())
    }

    pub fn launch(
        self, config: KernelConfig,
    ) -> CudaResult<PendingTask<'a, 'b>> {
        let args: Vec<*mut c_void> =
            self.args.iter().map(|x| x.param_pointer()).collect();
        self.elapsed("before launch");
        unsafe {
            self.k.stream.launch(
                &self.function,
                config.global_work_size as u32,
                config.local_work_size as u32,
                config.shared_mem as u32,
                &args,
            )?
        }

        self.elapsed("after launch");
        Ok(PendingTask(self))
    }
}

impl<'a, 'b> PendingTask<'a, 'b> {
    pub fn next_call(mut self) -> CudaResult<KernelTask<'a, 'b>> {
        self.sync_back()?;

        let mut task = self.0;
        task.launched.push(std::mem::take(&mut task.args));

        Ok(task)
    }

    pub fn next_call_function(
        self, name: &str,
    ) -> CudaResult<KernelTask<'a, 'b>> {
        let mut task = self.next_call()?;
        let function_name =
            CString::new(name).expect("Kernel name must not contain nul bytes");
        let function = task.k.module.get_function(&function_name)?;
        task.function = function;

        Ok(task)
    }

    pub fn complete(mut self) -> CudaResult<Kernel<'a>> {
        self.sync_back()?;

        let kernel = self.0.k;
        kernel.stream.synchronize()?;
        Ok(kernel)
    }

    fn sync_back(&mut self) -> CudaResult<()> {
        let kernel = &mut self.0;
        let tasks = kernel.launched.iter_mut().chain(once(&mut kernel.args));
        for task_args in tasks {
            for mut param in task_args.drain(..) {
                param.after_call(&kernel.k.stream)?;
                #[cfg(feature = "timer")]
                println!(
                    "[{:?}] Elapsed {} us (back)",
                    std::thread::current().id(),
                    kernel.instant.elapsed().as_nanos() as f64 / 1000.0,
                );
            }
        }

        Ok(())
    }
}
