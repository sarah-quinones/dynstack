#![feature(new_uninit)]
#![feature(dropck_eyepatch)]

//! Stack that allows users to allocate dynamically sized arrays.
//!
//! The stack wraps a buffer of bytes that it uses as a workspace.
//! Allocating an array takes a chunk of memory from the stack, which can be reused once the array
//! is dropped.
//!
//! # Examples:
//! ```
//! use core::mem::MaybeUninit;
//! use dynstack::{DynStack, StackReq, uninit_box};
//! use reborrow::ReborrowMut;
//!
//! // We allocate enough storage for 3 `i32` and 4 `u8`.
//! let mut buf = [MaybeUninit::uninit();
//!     StackReq::new::<i32>(3)
//!         .and(StackReq::new::<u8>(4))
//!         .bytes_required()];
//! let mut stack = DynStack::new(&mut buf);
//!
//! // We can have nested allocations,
//! let (mut array_i32, substack) = stack.rb_mut().make_with::<i32, _>(3, Default::default);
//! let (mut array_u8, _) = substack.make_with::<u8, _>(3, Default::default);
//!
//! array_i32[0] = 1;
//! array_i32[1] = 2;
//! array_i32[2] = 3;
//!
//! assert_eq!(array_i32[0], 1);
//! assert_eq!(array_i32[1], 2);
//! assert_eq!(array_i32[2], 3);
//!
//! assert_eq!(array_u8[0], 0);
//! assert_eq!(array_u8[1], 0);
//! assert_eq!(array_u8[2], 0);
//!
//! // or disjoint ones.
//! let (mut array_i32, _) = stack.rb_mut().make_with::<i32, _>(3, Default::default);
//! assert_eq!(array_i32[0], 0);
//! assert_eq!(array_i32[1], 0);
//! assert_eq!(array_i32[2], 0);
//!
//! let (mut array_u8, _) = stack.rb_mut().make_with::<u8, _>(3, Default::default);
//! assert_eq!(array_u8[0], 0);
//! assert_eq!(array_u8[1], 0);
//! assert_eq!(array_u8[2], 0);
//! ```

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use reborrow::ReborrowMut;

/// Stack wrapper around a buffer of uninitialized bytes.
pub struct DynStack<'a> {
    buffer: &'a mut [MaybeUninit<u8>],
}

/// Owns an unsized array of data, allocated from some stack.
#[derive(Debug)]
pub struct DynArray<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: (PhantomData<&'a ()>, PhantomData<T>),
}

/// Stack allocation requirements.
#[derive(Debug, Clone, Copy)]
pub struct StackReq {
    size: usize,
    align: usize,
}

const fn unwrap<T: Copy>(o: Option<T>) -> T {
    match o {
        Some(x) => x,
        None => panic!(),
    }
}

const fn round_up_pow2(a: usize, b: usize) -> usize {
    unwrap(a.checked_add(!b.wrapping_neg())) & b.wrapping_neg()
}

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

impl StackReq {
    pub const fn new_aligned<T>(n: usize, align: usize) -> StackReq {
        assert!(align >= core::mem::align_of::<T>());
        assert!(align.is_power_of_two());
        StackReq {
            size: unwrap(core::mem::size_of::<T>().checked_mul(n)),
            align,
        }
    }

    pub const fn new<T>(n: usize) -> StackReq {
        StackReq::new_aligned::<T>(n, core::mem::align_of::<T>())
    }

    pub const fn bytes_required(&self) -> usize {
        unwrap(self.size.checked_add(self.align - 1))
    }

    pub const fn and(self, other: StackReq) -> StackReq {
        let align = max(self.align, other.align);
        StackReq {
            size: unwrap(
                round_up_pow2(self.size, align).checked_add(round_up_pow2(other.size, align)),
            ),
            align,
        }
    }

    pub const fn or(self, other: StackReq) -> StackReq {
        let align = max(self.align, other.align);
        StackReq {
            size: max(
                round_up_pow2(self.size, align),
                round_up_pow2(other.size, align),
            ),
            align,
        }
    }
}

impl<'a, T> DynArray<'a, T> {
    fn get_data(self) -> &'a mut [T] {
        let len = self.len;
        let data = self.ptr as *mut T;
        core::mem::forget(self);
        unsafe { core::slice::from_raw_parts_mut(data, len) }
    }
}

unsafe impl<#[may_dangle] 'a, #[may_dangle] T> Drop for DynArray<'a, T> {
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(
                core::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) as *mut [T],
            )
        };
    }
}

impl<'a, T> core::ops::Deref for DynArray<'a, T> {
    type Target = [T];

    fn deref<'s>(&'s self) -> &'s Self::Target {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<'a, T> core::ops::DerefMut for DynArray<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }
}

pub fn uninit_box(n: usize) -> Box<[MaybeUninit<u8>]> {
    Box::new_uninit_slice(n)
}

unsafe fn transmute_slice<T>(slice: &mut [MaybeUninit<u8>], size: usize) -> &mut [T] {
    core::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut T, size)
}

fn init_array_with<T, F: FnMut() -> T>(mut f: F, array: &mut [MaybeUninit<T>]) -> &mut [T] {
    struct DropGuard<T> {
        ptr: *mut T,
        len: usize,
    }

    impl<T> Drop for DropGuard<T> {
        fn drop(&mut self) {
            unsafe {
                core::ptr::drop_in_place(
                    core::slice::from_raw_parts_mut(self.ptr, self.len) as *mut [T]
                )
            };
        }
    }
    let len = array.len();
    let ptr = array.as_mut_ptr() as *mut T;

    let mut guard = DropGuard { ptr, len: 0 };

    for i in 0..len {
        guard.len = i;
        unsafe { ptr.add(i).write(f()) };
    }
    core::mem::forget(guard);

    unsafe { core::slice::from_raw_parts_mut(ptr, len) }
}

impl<'a, 'b> ReborrowMut<'b> for DynStack<'a>
where
    'a: 'b,
{
    type Target = DynStack<'b>;

    fn rb_mut(&'b mut self) -> Self::Target {
        DynStack {
            buffer: self.buffer,
        }
    }
}

impl<'a> DynStack<'a> {
    pub fn new(buffer: &'a mut [MaybeUninit<u8>]) -> DynStack<'a> {
        DynStack { buffer }
    }

    pub fn make_aligned_uninit<T>(
        self,
        size: usize,
        align: usize,
    ) -> (DynArray<'a, MaybeUninit<T>>, DynStack<'a>) {
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

        let (taken, remaining) = split_buffer(
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
                ptr,
                len,
                _marker: (PhantomData, PhantomData),
            },
            DynStack::new(remaining),
        )
    }

    pub fn make_aligned_with<T, F: FnMut() -> T>(
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
                ptr,
                len,
                _marker: (PhantomData, PhantomData),
            },
            remaining,
        )
    }

    pub fn make_uninit<T>(self, size: usize) -> (DynArray<'a, MaybeUninit<T>>, DynStack<'a>) {
        self.make_aligned_uninit(size, core::mem::align_of::<T>())
    }

    pub fn make_with<T, F: FnMut() -> T>(
        self,
        size: usize,
        f: F,
    ) -> (DynArray<'a, T>, DynStack<'a>) {
        self.make_aligned_with(size, core::mem::align_of::<T>(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up() {
        assert_eq!(round_up_pow2(0, 4), 0);
        assert_eq!(round_up_pow2(1, 4), 4);
        assert_eq!(round_up_pow2(2, 4), 4);
        assert_eq!(round_up_pow2(3, 4), 4);
        assert_eq!(round_up_pow2(4, 4), 4);
    }

    #[test]
    fn basic_nested() {
        let mut buf = uninit_box(StackReq::new::<i32>(6).bytes_required());

        let stack = DynStack::new(&mut buf);

        let mut i = 0;
        let (arr0, stack) = stack.make_with::<i32, _>(3, || {
            i += 1;
            i - 1
        });
        assert_eq!(arr0[0], 0);
        assert_eq!(arr0[1], 1);
        assert_eq!(arr0[2], 2);

        let (arr1, _stack) = stack.make_with::<i32, _>(3, || {
            i += 1;
            i - 1
        });

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
        let mut buf = uninit_box(StackReq::new::<i32>(3).bytes_required());

        let mut stack = DynStack::new(&mut buf);

        let mut i = 0;
        {
            let (arr0, _) = stack.rb_mut().make_with::<i32, _>(3, || {
                i += 1;
                i - 1
            });
            assert_eq!(arr0[0], 0);
            assert_eq!(arr0[1], 1);
            assert_eq!(arr0[2], 2);
        }
        {
            let (arr1, _) = stack.rb_mut().make_with::<i32, _>(3, || {
                i += 1;
                i - 1
            });

            assert_eq!(arr1[0], 3);
            assert_eq!(arr1[1], 4);
            assert_eq!(arr1[2], 5);
        }
    }

    #[test]
    fn drop_nested() {
        use std::sync::atomic::{AtomicI32, Ordering};
        static DROP_COUNT: AtomicI32 = AtomicI32::new(0);

        struct CountedDrop;
        impl Drop for CountedDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut buf = uninit_box(StackReq::new::<CountedDrop>(6).bytes_required());
        let stack = DynStack::new(&mut buf);

        let stack = {
            let (_arr, stack) = stack.make_with(3, || CountedDrop);
            stack
        };
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
        let _stack = {
            let (_arr, stack) = stack.make_with(4, || CountedDrop);
            stack
        };
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 7);
    }

    #[test]
    fn drop_disjoint() {
        use std::sync::atomic::{AtomicI32, Ordering};
        static DROP_COUNT: AtomicI32 = AtomicI32::new(0);

        struct CountedDrop;
        impl Drop for CountedDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut buf = uninit_box(StackReq::new::<CountedDrop>(6).bytes_required());
        let mut stack = DynStack::new(&mut buf);

        {
            let _ = stack.rb_mut().make_with(3, || CountedDrop);
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
        }

        {
            let _ = stack.rb_mut().make_with(4, || CountedDrop);
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 7);
        }
    }
}
