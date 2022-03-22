# dynstack
Stack that allows users to allocate dynamically sized arrays.
                                                                                               
The stack wraps a buffer of bytes that it uses as a workspace.
Allocating an array takes a chunk of memory from the stack, which can be reused once the array
is dropped.
                                                                                               
# Examples
```rust
use core::mem::MaybeUninit;
use dynstack::{DynStack, StackReq};
use reborrow::ReborrowMut;
                                                                                         
// We allocate enough storage for 3 `i32` and 4 `u8`.
let mut buf = [MaybeUninit::uninit();
    StackReq::new::<i32>(3)
        .and(StackReq::new::<u8>(4))
        .unaligned_bytes_required()];
let mut stack = DynStack::new(&mut buf);
                                                                                         
// We can have nested allocations,
let (mut array_i32, substack) = stack.rb_mut().make_with::<i32, _>(3, Default::default);
let (mut array_u8, _) = substack.make_with::<u8, _>(3, Default::default);
                                                                                         
array_i32[0] = 1;
array_i32[1] = 2;
array_i32[2] = 3;
                                                                                         
assert_eq!(array_i32[0], 1);
assert_eq!(array_i32[1], 2);
assert_eq!(array_i32[2], 3);
                                                                                         
assert_eq!(array_u8[0], 0);
assert_eq!(array_u8[1], 0);
assert_eq!(array_u8[2], 0);
                                                                                         
// or disjoint ones.
let (mut array_i32, _) = stack.rb_mut().make_with::<i32, _>(3, Default::default);
assert_eq!(array_i32[0], 0);
assert_eq!(array_i32[1], 0);
assert_eq!(array_i32[2], 0);
                                                                                         
let (mut array_u8, _) = stack.rb_mut().make_with::<u8, _>(3, Default::default);
assert_eq!(array_u8[0], 0);
assert_eq!(array_u8[1], 0);
assert_eq!(array_u8[2], 0);
```
