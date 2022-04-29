use crate::stack_req::StackReq;
use alloc::alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout};
use core::mem::MaybeUninit;
use core::ptr::NonNull;

/// Buffer of uninitialized bytes to serve as workspace for dynamic arrays.
pub struct MemBuffer<A: Allocator = Global> {
    alloc: A,
    ptr: NonNull<u8>,
    size: usize,
    align: usize,
}

unsafe impl Sync for MemBuffer {}
unsafe impl Send for MemBuffer {}

impl<A: Allocator> Drop for MemBuffer<A> {
    fn drop(&mut self) {
        // SAFETY: this was initialized with std::alloc::alloc
        unsafe {
            self.alloc.deallocate(
                self.ptr,
                Layout::from_size_align_unchecked(self.size, self.align),
            )
        }
    }
}

fn to_layout(req: StackReq) -> Layout {
    unsafe { Layout::from_size_align_unchecked(req.size_bytes(), req.align_bytes()) }
}

/// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
/// provided allocator.
///
/// Calls [`std::alloc::handle_alloc_error`] in the case of failure.
///
/// # Example
/// ```
/// #![feature(allocator_api)]
///
/// use dyn_stack::{DynStack, StackReq, uninit_mem_in};
/// use std::alloc::Global;
///
/// let req = StackReq::new::<i32>(3);
/// let mut buf = uninit_mem_in(Global, req);
/// let stack = DynStack::new(&mut buf);
///
/// // use the stack
/// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
/// ```
pub fn uninit_mem_in<A: Allocator>(alloc: A, req: StackReq) -> MemBuffer<A> {
    try_uninit_mem_in(alloc, req).unwrap_or_else(|_| handle_alloc_error(to_layout(req)))
}

/// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
/// global allocator.
///
/// Calls [`std::alloc::handle_alloc_error`] in the case of failure.
///
/// # Example
/// ```
/// #![feature(allocator_api)]
///
/// use dyn_stack::{DynStack, StackReq, uninit_mem};
///
/// let req = StackReq::new::<i32>(3);
/// let mut buf = uninit_mem(req);
/// let stack = DynStack::new(&mut buf);
///
/// // use the stack
/// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
/// ```
pub fn uninit_mem(req: StackReq) -> MemBuffer {
    uninit_mem_in(Global, req)
}

/// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
/// provided allocator, or an `AllocError` in the case of failure.
///
/// # Example
/// ```
/// #![feature(allocator_api)]
///
/// use dyn_stack::{DynStack, StackReq, try_uninit_mem_in};
/// use std::alloc::Global;
///
/// let req = StackReq::new::<i32>(3);
/// let mut buf = try_uninit_mem_in(Global, req).unwrap();
/// let stack = DynStack::new(&mut buf);
///
/// // use the stack
/// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
/// ```
pub fn try_uninit_mem_in<A: Allocator>(
    alloc: A,
    req: StackReq,
) -> Result<MemBuffer<A>, AllocError> {
    unsafe {
        let ptr = alloc.allocate(to_layout(req))?;
        let size = ptr.len();
        let ptr = NonNull::new_unchecked(ptr.as_mut_ptr());
        Ok(MemBuffer {
            alloc,
            ptr,
            size,
            align: req.align_bytes(),
        })
    }
}

/// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
/// global allocator, or an `AllocError` in the case of failure.
///
/// # Example
/// ```
/// #![feature(allocator_api)]
///
/// use dyn_stack::{DynStack, StackReq, try_uninit_mem};
///
/// let req = StackReq::new::<i32>(3);
/// let mut buf = try_uninit_mem(req).unwrap();
/// let stack = DynStack::new(&mut buf);
///
/// // use the stack
/// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
/// ```
pub fn try_uninit_mem(req: StackReq) -> Result<MemBuffer, AllocError> {
    try_uninit_mem_in(Global, req)
}

impl<A: Allocator> core::ops::Deref for MemBuffer<A> {
    type Target = [MaybeUninit<u8>];

    fn deref(&self) -> &Self::Target {
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr() as *const MaybeUninit<u8>, self.size)
        }
    }
}

impl<A: Allocator> core::ops::DerefMut for MemBuffer<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut MaybeUninit<u8>, self.size)
        }
    }
}
