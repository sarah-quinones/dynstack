#![feature(new_uninit)]
#![feature(dropck_eyepatch)]

use core::marker::PhantomData;
use core::mem::MaybeUninit;

pub struct DynStack<'a> {
    buffer: &'a mut [MaybeUninit<u8>],
}

#[derive(Debug)]
pub struct DynArray<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: (PhantomData<&'a ()>, PhantomData<T>),
}

#[derive(Debug, Clone, Copy)]
pub struct StackReq {
    size: usize,
    align: usize,
}

impl StackReq {
    pub fn new_aligned<T>(n: usize, align: usize) -> StackReq {
        assert!(align >= core::mem::align_of::<T>());
        assert!(align.is_power_of_two());
        StackReq {
            size: core::mem::size_of::<T>().checked_mul(n).unwrap(),
            align,
        }
    }

    pub fn new<T>(n: usize) -> StackReq {
        StackReq::new_aligned::<T>(n, core::mem::align_of::<T>())
    }

    pub fn bytes_required(&self) -> usize {
        self.size + (self.align - 1)
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

impl<'a> DynStack<'a> {
    pub fn new(buffer: &'a mut [MaybeUninit<u8>]) -> DynStack<'a> {
        DynStack { buffer }
    }

    pub fn rb_mut<'s>(&'s mut self) -> DynStack<'s> {
        DynStack {
            buffer: self.buffer,
        }
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
