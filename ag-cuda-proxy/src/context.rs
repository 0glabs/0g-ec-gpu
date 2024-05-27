use rustacuda::{
    context::Context,
    error::{CudaError, CudaResult},
};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct CudaContext {
    context: Context,
    in_use: AtomicBool,
}

impl CudaContext {
    pub fn new(context: Context) -> Self {
        Self {
            context,
            in_use: AtomicBool::new(false),
        }
    }

    pub fn lock(&self) -> CudaResult<&Context> {
        let in_use = self.in_use.fetch_or(true, Ordering::SeqCst);
        if in_use {
            Err(CudaError::ContextAlreadyInUse)
        } else {
            Ok(&self.context)
        }
    }

    pub fn unlock(&self) { self.in_use.store(false, Ordering::SeqCst); }
}
