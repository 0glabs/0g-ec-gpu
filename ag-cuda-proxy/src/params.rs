use rustacuda::{error::CudaResult, memory::DeviceBuffer, stream::Stream};
use std::ffi::c_void;

pub trait ParamIO {
    fn param_pointer(&self) -> *mut c_void;
    fn after_call(&mut self, stream: &Stream) -> CudaResult<()>;
}

pub(crate) enum Param<'a, T> {
    InVal(T),
    InRef(&'a T),
    InMut(&'a mut T),
    Out(&'a mut T),

    InRefSlice(&'a [T]),
    InMutSlice(&'a mut [T]),
    OutSlice(&'a mut [T]),
}

impl<'a, T> Param<'a, T> {
    fn items(&self) -> usize {
        match self {
            Param::InVal(_)
            | Param::InRef(_)
            | Param::InMut(_)
            | Param::Out(_) => 1,
            Param::InRefSlice(x) => x.len(),
            Param::InMutSlice(x) | Param::OutSlice(x) => x.len(),
        }
    }

    fn size(&self) -> usize { std::mem::size_of::<T>() * self.items() }

    pub(crate) fn input_pointer(&self) -> Option<*const T> {
        match self {
            Param::InVal(_) => None,
            Param::InRef(x) => Some(*x as *const T),
            Param::InMut(x) => Some(*x as *const T),
            Param::Out(_) => None,
            Param::InRefSlice(x) => Some((*x).as_ptr()),
            Param::InMutSlice(x) => Some((*x).as_ptr()),
            Param::OutSlice(_) => None,
        }
    }

    pub(crate) fn output_pointer(&mut self) -> Option<*mut T> {
        match self {
            Param::InVal(_) => None,
            Param::InRef(_) => None,
            Param::InMut(x) => Some(*x as *mut T),
            Param::Out(x) => Some(*x as *mut T),
            Param::InRefSlice(_) => None,
            Param::InMutSlice(x) => Some((*x).as_mut_ptr()),
            Param::OutSlice(x) => Some((*x).as_mut_ptr()),
        }
    }

    pub(crate) fn before_call(
        &self, stream: &Stream,
    ) -> CudaResult<Option<DeviceBuffer<u8>>> {
        use rustacuda::memory::AsyncCopyDestination;
        if let Param::InVal(_) = self {
            return Ok(None);
        }

        let size = self.size();

        let mut buffer = unsafe { DeviceBuffer::<u8>::uninitialized(size)? };

        if let Some(pointer) = self.input_pointer() {
            let bytes = unsafe {
                std::slice::from_raw_parts(pointer as *const u8, size)
            };
            unsafe { buffer.async_copy_from(bytes, stream)? };
        }

        Ok(Some(buffer))
    }
}

impl<'a, T> ParamIO for (Param<'a, T>, Option<DeviceBuffer<u8>>) {
    fn param_pointer(&self) -> *mut c_void {
        if let Param::InVal(x) = &self.0 {
            x as *const T as *mut c_void
        } else {
            self.1.as_ref().unwrap() as *const _ as *mut c_void
        }
    }

    fn after_call(&mut self, stream: &Stream) -> CudaResult<()> {
        use rustacuda::memory::AsyncCopyDestination;
        let buffer = if let Some(x) = &mut self.1 {
            x
        } else {
            return Ok(());
        };

        if let Some(pointer) = self.0.output_pointer() {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    pointer as *mut u8,
                    self.0.size(),
                )
            };
            unsafe { buffer.async_copy_to(bytes, stream)? };
        }

        Ok(())
    }
}

pub struct DeviceParam<'a, T> {
    host_mem: &'a mut [T],
    device_mem: DeviceBuffer<u8>,
}

impl<'a, T> DeviceParam<'a, T> {
    pub fn new(val: &'a mut [T]) -> CudaResult<Self> {
        let size = val.len() * std::mem::size_of::<T>();
        let buffer = unsafe { DeviceBuffer::<u8>::uninitialized(size)? };

        Ok(Self {
            host_mem: val,
            device_mem: buffer,
        })
    }

    pub fn to_device(&mut self, stream: &Stream) -> CudaResult<()> {
        use rustacuda::memory::AsyncCopyDestination;
        let size = self.host_mem.len() * std::mem::size_of::<T>();

        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.host_mem.as_ptr() as *const u8,
                size,
            )
        };
        unsafe { self.device_mem.async_copy_from(bytes, stream)? };
        stream.synchronize()?;
        Ok(())
    }

    pub fn to_host(&mut self, stream: &Stream) -> CudaResult<()> {
        use rustacuda::memory::AsyncCopyDestination;
        let size = self.host_mem.len() * std::mem::size_of::<T>();

        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.host_mem.as_mut_ptr() as *mut u8,
                size,
            )
        };
        unsafe { self.device_mem.async_copy_to(bytes, stream)? };
        stream.synchronize()?;
        Ok(())
    }

    pub fn swap_device_pointer(me: &mut Self, another: &mut Self) {
        assert_eq!(me.host_mem.len(), another.host_mem.len());

        std::mem::swap(&mut me.device_mem, &mut another.device_mem);
    }
}

impl<'a, 'b, T> ParamIO for &'b DeviceParam<'a, T> {
    fn param_pointer(&self) -> *mut c_void {
        (&self.device_mem) as *const _ as *mut c_void
    }

    fn after_call(&mut self, _stream: &Stream) -> CudaResult<()> { Ok(()) }
}

pub struct DeviceData {
    size: usize,
    device_mem: DeviceBuffer<u8>,
}

impl DeviceData {
    pub fn uninitialized(size: usize) -> CudaResult<Self> {
        Ok(Self {
            size,
            device_mem: unsafe { DeviceBuffer::<u8>::uninitialized(size)? },
        })
    }

    pub fn upload<T>(val: &[T], stream: &Stream) -> CudaResult<Self> {
        use rustacuda::memory::AsyncCopyDestination;

        let size = val.len() * std::mem::size_of::<T>();
        let mut buffer = unsafe { DeviceBuffer::<u8>::uninitialized(size)? };

        let bytes = unsafe {
            std::slice::from_raw_parts(val.as_ptr() as *const u8, size)
        };
        unsafe { buffer.async_copy_from(bytes, stream)? };

        Ok(Self {
            size,
            device_mem: buffer,
        })
    }

    pub fn swap_device_pointer(me: &mut Self, another: &mut Self) {
        assert_eq!(me.size, another.size);

        std::mem::swap(&mut me.device_mem, &mut another.device_mem);
    }

    pub fn size(&self) -> usize { self.size }
}

impl<'b> ParamIO for &'b DeviceData {
    fn param_pointer(&self) -> *mut c_void {
        (&self.device_mem) as *const _ as *mut c_void
    }

    fn after_call(&mut self, _stream: &Stream) -> CudaResult<()> { Ok(()) }
}

pub(crate) struct NullPointer;

impl<'a> ParamIO for NullPointer {
    fn param_pointer(&self) -> *mut c_void {
        const NULL: &'static [u8; 0] = &[];
        (&NULL) as *const _ as *mut c_void
    }

    fn after_call(&mut self, _stream: &Stream) -> CudaResult<()> { Ok(()) }
}
