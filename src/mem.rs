use crate::stack_req::StackReq;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use std::alloc::{handle_alloc_error, AllocError, Allocator, Global, Layout};

pub struct Mem<A: Allocator = Global> {
    alloc: A,
    ptr: NonNull<u8>,
    size: usize,
    align: usize,
}

impl<A: Allocator> Drop for Mem<A> {
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

pub fn try_uninit_mem_in<A: Allocator>(alloc: A, req: StackReq) -> Result<Mem<A>, AllocError> {
    unsafe {
        let ptr = alloc.allocate(to_layout(req))?;
        let size = ptr.len();
        let ptr = NonNull::new_unchecked(ptr.as_mut_ptr());
        Ok(Mem {
            alloc,
            ptr,
            size,
            align: req.align_bytes(),
        })
    }
}

pub fn try_uninit_mem(req: StackReq) -> Result<Mem, AllocError> {
    try_uninit_mem_in(Global, req)
}

pub fn uninit_mem_in<A: Allocator>(alloc: A, req: StackReq) -> Mem<A> {
    try_uninit_mem_in(alloc, req).unwrap_or_else(|_| handle_alloc_error(to_layout(req)))
}

pub fn uninit_mem(req: StackReq) -> Mem {
    uninit_mem_in(Global, req)
}

impl<A: Allocator> core::ops::Deref for Mem<A> {
    type Target = [MaybeUninit<u8>];

    fn deref(&self) -> &Self::Target {
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr() as *const MaybeUninit<u8>, self.size)
        }
    }
}

impl<A: Allocator> core::ops::DerefMut for Mem<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut MaybeUninit<u8>, self.size)
        }
    }
}
