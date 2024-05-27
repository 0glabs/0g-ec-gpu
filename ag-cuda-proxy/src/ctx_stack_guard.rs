use std::{cell::Cell, thread_local};

use rustacuda::{context::ContextStack, error::CudaResult};

use crate::context::CudaContext;

thread_local! {
    static CONTEXT_GUARD: Cell<bool> = Cell::new(false);
}

fn lock_cu_context() -> bool { !CONTEXT_GUARD.replace(true) }

fn release_cu_context() { CONTEXT_GUARD.replace(false); }

/// A guard guarantee that only a single workspace can be activated at one time.
pub struct WorkspaceContextGuard<'a>(&'a CudaContext);

impl<'a> WorkspaceContextGuard<'a> {
    pub(crate) fn new(context: &'a CudaContext) -> CudaResult<Self> {
        assert!(lock_cu_context(), "Duplicated workspace context");
        ContextStack::push(&context.lock()?.get_unowned())?;
        Ok(WorkspaceContextGuard(context))
    }
}

impl<'a> Drop for WorkspaceContextGuard<'a> {
    fn drop(&mut self) {
        ContextStack::pop().expect("Cannot remove context.");
        self.0.unlock();
        release_cu_context();
    }
}
