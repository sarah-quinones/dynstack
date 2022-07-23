#![cfg_attr(
    feature = "nightly",
    feature(allocator_api, slice_ptr_len, slice_ptr_get, dropck_eyepatch)
)]
#![cfg_attr(not(feature = "std"), no_std)]

//! Stack that allows users to allocate dynamically sized arrays.
//!
//! The stack wraps a buffer of bytes that it uses as a workspace.
//! Allocating an array takes a chunk of memory from the stack, which can be reused once the array
//! is dropped.
//!
//! # Examples:
//! ```
//! use core::mem::MaybeUninit;
//! use dyn_stack::{DynStack, StackReq};
//! use reborrow::ReborrowMut;
//!
//! // We allocate enough storage for 3 `i32` and 4 `u8`.
//! let mut buf = [MaybeUninit::uninit();
//!     StackReq::new::<i32>(3)
//!         .and(StackReq::new::<u8>(4))
//!         .unaligned_bytes_required()];
//! let mut stack = DynStack::new(&mut buf);
//!
//! {
//!     // We can have nested allocations.
//!     // 3×`i32`
//!     let (array_i32, substack) = stack.rb_mut().make_with::<i32, _>(3, |i| i as i32);
//!     // and 4×`u8`
//!     let (mut array_u8, _) = substack.make_with::<u8, _>(4, |_| 0);
//!
//!     // We can read from the arrays,
//!     assert_eq!(array_i32[0], 0);
//!     assert_eq!(array_i32[1], 1);
//!     assert_eq!(array_i32[2], 2);
//!
//!     // and write to them.
//!     array_u8[0] = 1;
//!
//!     assert_eq!(array_u8[0], 1);
//!     assert_eq!(array_u8[1], 0);
//!     assert_eq!(array_u8[2], 0);
//!     assert_eq!(array_u8[3], 0);
//! }
//!
//! // We can also have disjoint allocations.
//! {
//!     // 3×`i32`
//!     let (mut array_i32, _) = stack.rb_mut().make_with::<i32, _>(3, |i| i as i32);
//!     assert_eq!(array_i32[0], 0);
//!     assert_eq!(array_i32[1], 1);
//!     assert_eq!(array_i32[2], 2);
//! }
//!
//! {
//!     // or 4×`u8`
//!     let (mut array_u8, _) = stack.rb_mut().make_with::<i32, _>(4, |i| i as i32 + 3);
//!     assert_eq!(array_u8[0], 3);
//!     assert_eq!(array_u8[1], 4);
//!     assert_eq!(array_u8[2], 5);
//!     assert_eq!(array_u8[3], 6);
//! }
//! ```

#[cfg(feature = "std")]
extern crate alloc;

#[cfg(feature = "std")]
pub mod mem;

#[cfg(all(feature = "nightly", feature = "std"))]
pub use mem::MemBuffer;

#[cfg(feature = "std")]
pub use mem::GlobalMemBuffer;

mod stack_req;
pub use stack_req::{SizeOverflow, StackReq};

use core::fmt::Debug;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
pub use reborrow::ReborrowMut;

/// Stack wrapper around a buffer of uninitialized bytes.
pub struct DynStack<'a> {
    buffer: &'a mut [MaybeUninit<u8>],
}

/// Owns an unsized array of data, allocated from some stack.
pub struct DynArray<'a, T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: (PhantomData<&'a ()>, PhantomData<T>),
}

impl<'a, T: Debug> Debug for DynArray<'a, T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        fmt.debug_list().entries(&**self).finish()
    }
}

unsafe impl<'a, T> Send for DynArray<'a, T> where T: Send {}
unsafe impl<'a, T> Sync for DynArray<'a, T> where T: Sync {}

impl<'a, T> DynArray<'a, T> {
    #[inline(always)]
    fn get_data(self) -> &'a mut [T] {
        let len = self.len;
        let data = self.ptr.as_ptr();
        core::mem::forget(self);
        unsafe { core::slice::from_raw_parts_mut(data, len) }
    }
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] 'a, #[may_dangle] T> Drop for DynArray<'a, T> {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(
                core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) as *mut [T]
            )
        };
    }
}

#[cfg(not(feature = "nightly"))]
impl<'a, T> Drop for DynArray<'a, T> {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(
                core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) as *mut [T]
            )
        };
    }
}

impl<'a, T> core::ops::Deref for DynArray<'a, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &'_ Self::Target {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<'a, T> core::ops::DerefMut for DynArray<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[inline(always)]
unsafe fn transmute_slice<T>(slice: &mut [MaybeUninit<u8>], size: usize) -> &mut [T] {
    core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, size)
}

struct DropGuard<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Drop for DropGuard<T> {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.ptr, self.len) as *mut [T])
        };
    }
}

#[inline(always)]
fn init_array_with<T, F: FnMut(usize) -> T>(mut f: F, array: &mut [MaybeUninit<T>]) -> &mut [T] {
    let len = array.len();
    let ptr = array.as_mut_ptr() as *mut T;

    let mut guard = DropGuard { ptr, len: 0 };

    for i in 0..len {
        guard.len = i;
        unsafe { ptr.add(i).write(f(i)) };
    }
    core::mem::forget(guard);

    unsafe { core::slice::from_raw_parts_mut(ptr, len) }
}

#[inline(always)]
unsafe fn init_array_with_iter<T, I: Iterator<Item = T>>(
    iter: I,
    ptr: *mut T,
    max_len: usize,
) -> usize {
    let mut guard = DropGuard { ptr, len: 0 };
    let mut len = 0;

    iter.take(max_len).enumerate().for_each(|(i, item)| {
        guard.len = i;
        ptr.add(i).write(item);
        len = i + 1;
    });

    core::mem::forget(guard);

    len
}

impl<'a, 'b> ReborrowMut<'b> for DynStack<'a>
where
    'a: 'b,
{
    type Target = DynStack<'b>;

    #[inline(always)]
    fn rb_mut(&'b mut self) -> Self::Target {
        DynStack {
            buffer: self.buffer,
        }
    }
}

impl<'a> DynStack<'a> {
    /// Returns a new [`DynStack`] from the provided memory buffer.
    pub fn new(buffer: &'a mut [MaybeUninit<u8>]) -> DynStack<'a> {
        DynStack { buffer }
    }

    #[inline(always)]
    fn split_buffer(
        buffer: &mut [MaybeUninit<u8>],
        size: usize,
        align: usize,
        sizeof_val: usize,
        alignof_val: usize,
    ) -> (&mut [MaybeUninit<u8>], &mut [MaybeUninit<u8>]) {
        assert!(alignof_val <= align);
        assert!(align.is_power_of_two());

        let align_offset = buffer.as_mut_ptr().align_offset(align);
        let buffer = &mut buffer[align_offset..];
        buffer.split_at_mut(size.checked_mul(sizeof_val).unwrap())
    }

    /// Returns a new aligned and uninitialized [`DynArray`] and a stack over the remainder of the
    /// buffer.
    ///
    /// # Panics
    ///
    /// Panics if the stack isn't large enough to allocate the array.
    #[inline(always)]
    pub fn make_aligned_uninit<T>(
        self,
        size: usize,
        align: usize,
    ) -> (DynArray<'a, MaybeUninit<T>>, DynStack<'a>) {
        let (taken, remaining) = Self::split_buffer(
            self.buffer,
            size,
            align,
            core::mem::size_of::<T>(),
            core::mem::align_of::<T>(),
        );

        let (len, ptr) = {
            let taken = unsafe { transmute_slice::<MaybeUninit<T>>(taken, size) };
            (taken.len(), taken.as_mut_ptr())
        };
        (
            DynArray {
                ptr: unsafe { NonNull::<MaybeUninit<T>>::new_unchecked(ptr) },
                len,
                _marker: (PhantomData, PhantomData),
            },
            DynStack::new(remaining),
        )
    }

    /// Returns a new aligned [`DynArray`], initialized with the provided function, and a stack
    /// over the remainder of the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the stack isn't large enough to allocate the array, or if the provided function
    /// panics.
    #[inline(always)]
    pub fn make_aligned_with<T, F: FnMut(usize) -> T>(
        self,
        size: usize,
        align: usize,
        f: F,
    ) -> (DynArray<'a, T>, DynStack<'a>) {
        let (taken, remaining) = self.make_aligned_uninit(size, align);
        let (len, ptr) = {
            let taken = init_array_with(f, taken.get_data());
            (taken.len(), taken.as_mut_ptr())
        };
        (
            DynArray {
                ptr: unsafe { NonNull::<T>::new_unchecked(ptr) },
                len,
                _marker: (PhantomData, PhantomData),
            },
            remaining,
        )
    }

    /// Returns a new uninitialized [`DynArray`] and a stack over the remainder of the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the stack isn't large enough to allocate the array.
    #[inline(always)]
    pub fn make_uninit<T>(self, size: usize) -> (DynArray<'a, MaybeUninit<T>>, DynStack<'a>) {
        self.make_aligned_uninit(size, core::mem::align_of::<T>())
    }

    /// Returns a new [`DynArray`], initialized with the provided function, and a stack over the
    /// remainder of the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the stack isn't large enough to allocate the array, or if the provided function
    /// panics.
    #[inline(always)]
    pub fn make_with<T, F: FnMut(usize) -> T>(
        self,
        size: usize,
        f: F,
    ) -> (DynArray<'a, T>, DynStack<'a>) {
        self.make_aligned_with(size, core::mem::align_of::<T>(), f)
    }

    /// Returns a new aligned [`DynArray`], initialized with the provided iterator, and a stack
    /// over the remainder of the buffer.  
    /// If there isn't enough space for all the iterator items, then the returned array only
    /// contains the first elements that fit into the stack.
    ///
    /// # Panics
    ///
    /// Panics if the provided iterator panics.
    #[inline(always)]
    pub fn collect_aligned<I: IntoIterator>(
        self,
        align: usize,
        iter: I,
    ) -> (DynArray<'a, I::Item>, DynStack<'a>) {
        self.collect_aligned_impl(align, iter.into_iter())
    }

    /// Returns a new [`DynArray`], initialized with the provided iterator, and a stack over the
    /// remainder of the buffer.  
    /// If there isn't enough space for all the iterator items, then the returned array only
    /// contains the first elements that fit into the stack.
    ///
    /// # Panics
    ///
    /// Panics if the provided iterator panics.
    #[inline(always)]
    pub fn collect<I: IntoIterator>(self, iter: I) -> (DynArray<'a, I::Item>, DynStack<'a>) {
        self.collect_aligned_impl(core::mem::align_of::<I::Item>(), iter.into_iter())
    }

    #[inline(always)]
    fn collect_aligned_impl<I: Iterator>(
        self,
        align: usize,
        iter: I,
    ) -> (DynArray<'a, I::Item>, DynStack<'a>) {
        let sizeof_val = core::mem::size_of::<I::Item>();
        let alignof_val = core::mem::align_of::<I::Item>();

        assert!(alignof_val <= align);
        assert!(align.is_power_of_two());

        let align_offset = self.buffer.as_mut_ptr().align_offset(align);
        let buffer = &mut self.buffer[align_offset..];
        let buffer_ptr = buffer.as_mut_ptr();
        unsafe {
            let len = init_array_with_iter(
                iter,
                buffer_ptr as *mut I::Item,
                if sizeof_val == 0 {
                    usize::MAX
                } else {
                    buffer.len() / sizeof_val
                },
            );

            let remaining_slice = core::slice::from_raw_parts_mut(
                buffer_ptr.wrapping_add(len * sizeof_val),
                buffer.len() - len * sizeof_val,
            );
            (
                DynArray {
                    ptr: NonNull::new_unchecked(buffer_ptr as *mut I::Item),
                    len,
                    _marker: (PhantomData, PhantomData),
                },
                Self {
                    buffer: remaining_slice,
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(0));
        let stack = DynStack::new(&mut buf);
        let (_arr0, _stack) = stack.make_with::<i32, _>(0, |i| i as i32);
    }

    #[test]
    #[should_panic]
    fn empty_overflow() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(0));
        let stack = DynStack::new(&mut buf);
        let (_arr0, _stack) = stack.make_with::<i32, _>(1, |i| i as i32);
    }

    #[test]
    fn empty_collect() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(0));
        let stack = DynStack::new(&mut buf);
        let (_arr0, _stack) = stack.collect(0..0);
    }

    #[test]
    fn empty_collect_overflow() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(0));
        let stack = DynStack::new(&mut buf);
        let (arr0, _stack) = stack.collect(0..1);
        assert!(arr0.is_empty());
    }

    #[test]
    #[should_panic]
    fn overflow() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(1));
        let stack = DynStack::new(&mut buf);
        let (_arr0, _stack) = stack.make_with::<i32, _>(2, |i| i as i32);
    }

    #[test]
    fn collect_overflow() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(1));
        let stack = DynStack::new(&mut buf);
        let (arr0, _stack) = stack.collect(1..3);
        assert_eq!(arr0.len(), 1);
        assert_eq!(arr0[0], 1)
    }

    #[test]
    fn basic_nested() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(6));

        let stack = DynStack::new(&mut buf);

        let (arr0, stack) = stack.make_with::<i32, _>(3, |i| i as i32);
        assert_eq!(arr0[0], 0);
        assert_eq!(arr0[1], 1);
        assert_eq!(arr0[2], 2);

        let (arr1, _) = stack.make_with::<i32, _>(3, |i| i as i32 + 3);

        // first values are untouched
        assert_eq!(arr0[0], 0);
        assert_eq!(arr0[1], 1);
        assert_eq!(arr0[2], 2);

        assert_eq!(arr1[0], 3);
        assert_eq!(arr1[1], 4);
        assert_eq!(arr1[2], 5);
    }

    #[test]
    fn basic_disjoint() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(3));

        let mut stack = DynStack::new(&mut buf);

        {
            let (arr0, _) = stack.rb_mut().make_with::<i32, _>(3, |i| i as i32);
            assert_eq!(arr0[0], 0);
            assert_eq!(arr0[1], 1);
            assert_eq!(arr0[2], 2);
        }
        {
            let (arr1, _) = stack.rb_mut().make_with::<i32, _>(3, |i| i as i32 + 3);

            assert_eq!(arr1[0], 3);
            assert_eq!(arr1[1], 4);
            assert_eq!(arr1[2], 5);
        }
    }

    #[test]
    fn basic_nested_collect() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(6));
        let stack = DynStack::new(&mut buf);

        let (arr0, stack) = stack.collect(0..3_i32);
        assert_eq!(arr0[0], 0);
        assert_eq!(arr0[1], 1);
        assert_eq!(arr0[2], 2);

        let (arr1, _) = stack.collect(3..6_i32);

        // first values are untouched
        assert_eq!(arr0[0], 0);
        assert_eq!(arr0[1], 1);
        assert_eq!(arr0[2], 2);

        assert_eq!(arr1[0], 3);
        assert_eq!(arr1[1], 4);
        assert_eq!(arr1[2], 5);
    }

    #[test]
    fn basic_disjoint_collect() {
        let mut buf = GlobalMemBuffer::new(StackReq::new::<i32>(3));

        let mut stack = DynStack::new(&mut buf);

        {
            let (arr0, _) = stack.rb_mut().collect(0..3_i32);
            assert_eq!(arr0[0], 0);
            assert_eq!(arr0[1], 1);
            assert_eq!(arr0[2], 2);
        }
        {
            let (arr1, _) = stack.rb_mut().collect(3..6_i32);

            assert_eq!(arr1[0], 3);
            assert_eq!(arr1[1], 4);
            assert_eq!(arr1[2], 5);
        }
    }

    #[test]
    fn drop_nested() {
        use core::sync::atomic::{AtomicI32, Ordering};
        static DROP_COUNT: AtomicI32 = AtomicI32::new(0);

        struct CountedDrop;
        impl Drop for CountedDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut buf = GlobalMemBuffer::new(StackReq::new::<CountedDrop>(6));
        let stack = DynStack::new(&mut buf);

        let stack = {
            let (_arr, stack) = stack.make_with(3, |_| CountedDrop);
            stack
        };
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
        let _stack = {
            let (_arr, stack) = stack.make_with(4, |_| CountedDrop);
            stack
        };
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 7);
    }

    #[test]
    fn drop_disjoint() {
        use core::sync::atomic::{AtomicI32, Ordering};
        static DROP_COUNT: AtomicI32 = AtomicI32::new(0);

        struct CountedDrop;
        impl Drop for CountedDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut buf = GlobalMemBuffer::new(StackReq::new::<CountedDrop>(6));
        let mut stack = DynStack::new(&mut buf);

        {
            let _ = stack.rb_mut().make_with(3, |_| CountedDrop);
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
        }

        {
            let _ = stack.rb_mut().make_with(4, |_| CountedDrop);
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 7);
        }
    }
}
