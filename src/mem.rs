use crate::stack_req::StackReq;
use alloc::alloc::handle_alloc_error;
use alloc::alloc::Layout;
use core::mem::ManuallyDrop;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

/// Buffer of uninitialized bytes to serve as workspace for dynamic arrays.
pub struct GlobalMemBuffer {
    ptr: NonNull<u8>,
    size: usize,
    align: usize,
}

impl GlobalMemBuffer {
    /// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
    /// global allocator.
    ///
    /// Calls [`alloc::alloc::handle_alloc_error`] in the case of failure.
    ///
    /// # Example
    /// ```
    /// use dyn_stack::{DynStack, StackReq, GlobalMemBuffer};
    ///
    /// let req = StackReq::new::<i32>(3);
    /// let mut buf = GlobalMemBuffer::new(req);
    /// let stack = DynStack::new(&mut buf);
    ///
    /// // use the stack
    /// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
    /// ```
    pub fn new(req: StackReq) -> Self {
        Self::try_new(req).unwrap_or_else(|_| handle_alloc_error(to_layout(req)))
    }

    /// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
    /// global allocator, or an error if the allocation did not succeed.
    ///
    /// # Example
    /// ```
    /// use dyn_stack::{DynStack, StackReq, GlobalMemBuffer};
    ///
    /// let req = StackReq::try_new::<i32>(3).unwrap();
    /// let mut buf = GlobalMemBuffer::new(req);
    /// let stack = DynStack::new(&mut buf);
    ///
    /// // use the stack
    /// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
    /// ```
    pub fn try_new(req: StackReq) -> Result<Self, AllocError> {
        unsafe {
            if req.size_bytes() == 0 {
                let ptr = core::ptr::null_mut::<u8>().wrapping_add(req.align_bytes());
                Ok(GlobalMemBuffer {
                    ptr: NonNull::<u8>::new_unchecked(ptr),
                    size: 0,
                    align: req.align_bytes(),
                })
            } else {
                let layout = to_layout(req);
                let ptr = alloc::alloc::alloc(layout);
                if ptr.is_null() {
                    return Err(AllocError);
                }
                let size = layout.size();
                let ptr = NonNull::<u8>::new_unchecked(ptr);
                Ok(GlobalMemBuffer {
                    ptr,
                    size,
                    align: req.align_bytes(),
                })
            }
        }
    }

    /// Creates a `GlobalMemBuffer`	from its raw components.
    ///
    /// # Safety
    ///
    /// The arguments to this function must have been acquired from a call to
    /// [`GlobalMemBuffer::into_raw_parts`]
    pub unsafe fn from_raw_parts(ptr: *mut u8, size: usize, align: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr),
            size,
            align,
        }
    }

    /// Decomposes a `GlobalMemBuffer` into its raw components.
    pub fn into_raw_parts(self) -> (*mut u8, usize, usize) {
        let no_drop = ManuallyDrop::new(self);
        (no_drop.ptr.as_ptr(), no_drop.size, no_drop.align)
    }
}

unsafe impl Sync for GlobalMemBuffer {}
unsafe impl Send for GlobalMemBuffer {}

fn to_layout(req: StackReq) -> Layout {
    unsafe { Layout::from_size_align_unchecked(req.size_bytes(), req.align_bytes()) }
}

impl Drop for GlobalMemBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.size != 0 {
                alloc::alloc::dealloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align_unchecked(self.size, self.align),
                );
            }
        }
    }
}

impl core::ops::Deref for GlobalMemBuffer {
    type Target = [MaybeUninit<u8>];

    fn deref(&self) -> &Self::Target {
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr() as *const MaybeUninit<u8>, self.size)
        }
    }
}

impl core::ops::DerefMut for GlobalMemBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut MaybeUninit<u8>, self.size)
        }
    }
}

/// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
/// global allocator.
///
/// Calls [`alloc::alloc::handle_alloc_error`] in the case of failure.
///
/// # Example
/// ```
/// use dyn_stack::{DynStack, StackReq, uninit_mem_in_global};
///
/// let req = StackReq::new::<i32>(3);
/// let mut buf = uninit_mem_in_global(req);
/// let stack = DynStack::new(&mut buf);
///
/// // use the stack
/// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
/// ```
#[deprecated = "use GlobalMemBuffer::new instead"]
pub fn uninit_mem_in_global(req: StackReq) -> GlobalMemBuffer {
    GlobalMemBuffer::new(req)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocError;

impl core::fmt::Display for AllocError {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        fmt.write_str("memory allocation failed")
    }
}

/// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
/// global allocator, or an error if the allocation did not succeed.
///
/// # Example
/// ```
/// use dyn_stack::{DynStack, StackReq, try_uninit_mem_in_global};
///
/// let req = StackReq::new::<i32>(3);
/// let mut buf = try_uninit_mem_in_global(req).unwrap();
/// let stack = DynStack::new(&mut buf);
///
/// // use the stack
/// let (arr, _) = stack.make_with::<i32, _>(3, |i| i as i32);
/// ```
#[deprecated = "use GlobalMemBuffer::try_new instead"]
pub fn try_uninit_mem_in_global(req: StackReq) -> Result<GlobalMemBuffer, AllocError> {
    GlobalMemBuffer::try_new(req)
}

#[cfg(feature = "nightly")]
pub use nightly::*;

#[cfg(feature = "nightly")]
mod nightly {
    use super::*;
    use alloc::alloc::{AllocError, Allocator, Global};

    /// Buffer of uninitialized bytes to serve as workspace for dynamic arrays.
    pub struct MemBuffer<A: Allocator = Global> {
        alloc: A,
        ptr: NonNull<u8>,
        size: usize,
        align: usize,
    }

    unsafe impl<A: Allocator> Sync for MemBuffer<A> {}
    unsafe impl<A: Allocator> Send for MemBuffer<A> {}

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

    /// Allocate a memory buffer with sufficient storage for the given stack requirements, using the
    /// provided allocator.
    ///
    /// Calls [`alloc::alloc::handle_alloc_error`] in the case of failure.
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
                core::slice::from_raw_parts_mut(
                    self.ptr.as_ptr() as *mut MaybeUninit<u8>,
                    self.size,
                )
            }
        }
    }
}
