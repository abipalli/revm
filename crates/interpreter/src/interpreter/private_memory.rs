use circuit_sdk::prelude::GateIndexVec;
use core::{fmt, hash::Hash, mem};
use primitives::{hex, U256};

/// Simple mapping from SharedMemory to (Shared)PrivateMemory
///
/// > U256 (256-bits / 8 bits/B) = 32 bytes
///
/// > sizeof PrivateMemoryRef <= 32 bytes
pub struct PrivateMemoryRef {
    /// tag = 4 bytes
    tag: [u8; 4],
    /// index max 2^128
    index: [u8; 28],
    // index: Uint<192, >,
}

const PRIVATE_REF_TAG: &[u8; 4] = b"GATE";

pub(crate) fn is_bytes_private_ref(bytes: &[u8]) -> bool {
    if bytes.len() < 4 {
        return false;
    } else {
        return bytes[..4] == *PRIVATE_REF_TAG;
    }
}

impl Into<U256> for PrivateMemoryRef {
    fn into(self) -> U256 {
        let mut out: [u8; 32] = [0; 32];
        out[..PRIVATE_REF_TAG.len()].copy_from_slice(PRIVATE_REF_TAG);

        out[PRIVATE_REF_TAG.len()..].copy_from_slice(&self.index);

        U256::from_le_bytes(out)
    }
}

impl TryFrom<U256> for PrivateMemoryRef {
    type Error = ();

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        let out: [u8; 32] = value.to_le_bytes();
        if out[..PRIVATE_REF_TAG.len()] != *PRIVATE_REF_TAG {
            return Err(());
        }

        let mut index: [u8; 28] = Default::default();
        index.copy_from_slice(&out[PRIVATE_REF_TAG.len()..32]);

        Ok(Self {
            tag: PRIVATE_REF_TAG.clone(),
            index,
        })
    }
}

pub(crate) fn is_uint_256_private_ref(value: U256) -> bool {
    let bytes: [u8; 32] = value.to_le_bytes();
    is_bytes_private_ref(&bytes)
}

#[derive(Clone, Debug, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum PrivateMemoryValue {
    Private(GateIndexVec),
}

impl PartialEq for PrivateMemoryValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Private(l0), Self::Private(r0)) => l0 == r0,
        }
    }
}

impl Hash for PrivateMemoryValue {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrivateMemory {
    /// The underlying buffer.
    buffer: Vec<PrivateMemoryValue>,
    /// Memory checkpoints for each depth.
    /// Invariant: these are always in bounds of `data`.
    checkpoints: Vec<usize>,
    /// Invariant: equals `self.checkpoints.last()`
    last_checkpoint: usize,
    /// Memory limit. See [`Cfg`](context_interface::Cfg).
    #[cfg(feature = "memory_limit")]
    memory_limit: u64,
}

/// Empty private memory.
///
/// Used as placeholder inside Interpreter when it is not running.
/// > Do not use as default initializer
pub const EMPTY_PRIVATE_MEMORY: PrivateMemory = PrivateMemory {
    buffer: Vec::new(),
    checkpoints: Vec::new(),
    last_checkpoint: 0,
    #[cfg(feature = "memory_limit")]
    memory_limit: u64::MAX,
};

impl fmt::Debug for PrivateMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrivateMemory")
            .field("current_len", &self.len())
            // .field("context_memory", &hex::encode(self.context_memory()))
            .finish_non_exhaustive()
    }
}

impl Default for PrivateMemory {
    #[inline]
    fn default() -> Self {
        Self::with_capacity(1024 * 8)
    }
}

pub trait PrivateMemoryGetter {
    fn memory_mut(&mut self) -> &mut PrivateMemory;
    fn memory(&self) -> &PrivateMemory;
}

impl PrivateMemoryGetter for PrivateMemory {
    #[inline]
    fn memory_mut(&mut self) -> &mut PrivateMemory {
        self
    }

    #[inline]
    fn memory(&self) -> &PrivateMemory {
        self
    }
}

impl PrivateMemory {
    /// Creates a new memory instance that can be shared between calls.
    ///
    /// The default initial capacity is 4KiB.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(4 * 1024) // from evmone
    }

    /// Creates a new memory instance that can be shared between calls with the given `capacity`.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            checkpoints: Vec::with_capacity(32),
            last_checkpoint: 0,
            #[cfg(feature = "memory_limit")]
            memory_limit: u64::MAX,
        }
    }

    /// Creates a new memory instance that can be shared between calls,
    /// with `memory_limit` as upper bound for allocation size.
    ///
    /// The default initial capacity is 4KiB.
    #[cfg(feature = "memory_limit")]
    #[inline]
    pub fn new_with_memory_limit(memory_limit: u64) -> Self {
        Self {
            memory_limit,
            ..Self::new()
        }
    }

    /// Returns `true` if the `new_size` for the current context memory will
    /// make the shared buffer length exceed the `memory_limit`.
    #[cfg(feature = "memory_limit")]
    #[inline]
    pub fn limit_reached(&self, new_size: usize) -> bool {
        self.last_checkpoint.saturating_add(new_size) as u64 > self.memory_limit
    }

    /// Prepares the shared memory for a new context.
    #[inline]
    pub fn new_context(&mut self) {
        let new_checkpoint = self.buffer.len();
        self.checkpoints.push(new_checkpoint);
        self.last_checkpoint = new_checkpoint;
    }

    /// Prepares the shared memory for returning to the previous context.
    #[inline]
    pub fn free_context(&mut self) {
        if let Some(old_checkpoint) = self.checkpoints.pop() {
            self.last_checkpoint = self.checkpoints.last().cloned().unwrap_or_default();
            // SAFETY: `buffer` length is less than or equal `old_checkpoint`
            unsafe { self.buffer.set_len(old_checkpoint) };
        }
    }

    /// Returns the length of the current memory range.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len() - self.last_checkpoint
    }

    /// Returns `true` if the current memory range is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resizes the memory in-place so that `len` is equal to `new_len`.
    #[inline]
    pub fn resize(&mut self, new_size: usize) {
        self.buffer.resize(
            self.last_checkpoint + new_size,
            PrivateMemoryValue::Private(Default::default()),
        );
    }

    #[inline]
    pub fn push(&mut self, value: PrivateMemoryValue) -> PrivateMemoryRef {
        self.buffer.push(value);
        let id = self.buffer.len() - 1;
        println!("private_memory <- {}", id);

        let mut index: [u8; 28] = [0; 28];
        index[..id.to_le_bytes().len()].copy_from_slice(&id.to_le_bytes());

        PrivateMemoryRef {
            tag: PRIVATE_REF_TAG.clone(),
            index,
        }
    }

    #[inline]
    pub fn get(&self, reference: PrivateMemoryRef) -> Option<&PrivateMemoryValue> {
        // Ensure reference.index is not larger than usize
        if reference.index.len() > mem::size_of::<usize>() {
            return None; // Invalid index size
        }

        // Initialize the buffer for usize conversion
        let mut index_raw = [0u8; mem::size_of::<usize>()];

        // Copy the index bytes into the buffer
        index_raw[..reference.index.len()].copy_from_slice(&reference.index);

        // Convert to usize, assuming little-endian format
        let index = usize::from_le_bytes(index_raw);

        // Bounds checking to prevent out-of-bounds access
        self.buffer.get(index)
    }

    /// Returns a reference to the memory of the current context, the active memory.
    #[inline]
    pub fn context_memory(&self) -> &[PrivateMemoryValue] {
        // SAFETY: Access bounded by buffer length
        unsafe {
            self.buffer
                .get_unchecked(self.last_checkpoint..self.buffer.len())
        }
    }

    /// Returns a mutable reference to the memory of the current context.
    #[inline]
    pub fn context_memory_mut(&mut self) -> &mut [PrivateMemoryValue] {
        let buf_len = self.buffer.len();
        // SAFETY: Access bounded by buffer length
        unsafe { self.buffer.get_unchecked_mut(self.last_checkpoint..buf_len) }
    }
}

#[cfg(test)]
mod tests {
    use circuit_sdk::{
        prelude::{CircuitExecutor, WRK17CircuitBuilder},
        uint::{GarbledBoolean, GarbledUint},
    };

    use super::*;

    #[test]
    fn store_and_get_private_value() {
        let mut cb = WRK17CircuitBuilder::default();
        let a = cb.input(&GarbledUint::<256>::one());
        let b = cb.input(&GarbledUint::<256>::one());
        let sum_operation = cb.add(&a, &b);

        let value = PrivateMemoryValue::Private(sum_operation);

        let mut private_memory = PrivateMemory::new();
        let private_ref = private_memory.push(value);

        if let Some(PrivateMemoryValue::Private(gates_from_memory)) =
            private_memory.get(private_ref)
        {
            let expected_sum = &cb.input(&GarbledUint::<256>::from(2_u8));
            let result = cb
                .compile_and_execute::<256>(expected_sum)
                .expect("Error compiling circuit");
            assert_eq!(result, GarbledUint::zero())
        } else {
            ()
        }
    }

    #[test]
    fn new_free_context() {
        let mut private_memory = PrivateMemory::new();
        private_memory.new_context();

        assert_eq!(private_memory.buffer.len(), 0);
        assert_eq!(private_memory.checkpoints.len(), 1);
        assert_eq!(private_memory.last_checkpoint, 0);

        unsafe { private_memory.buffer.set_len(32) };
        assert_eq!(private_memory.len(), 32);
        private_memory.new_context();

        assert_eq!(private_memory.buffer.len(), 32);
        assert_eq!(private_memory.checkpoints.len(), 2);
        assert_eq!(private_memory.last_checkpoint, 32);
        assert_eq!(private_memory.len(), 0);

        unsafe { private_memory.buffer.set_len(96) };
        assert_eq!(private_memory.len(), 64);
        private_memory.new_context();

        assert_eq!(private_memory.buffer.len(), 96);
        assert_eq!(private_memory.checkpoints.len(), 3);
        assert_eq!(private_memory.last_checkpoint, 96);
        assert_eq!(private_memory.len(), 0);

        // Free contexts
        private_memory.free_context();
        assert_eq!(private_memory.buffer.len(), 96);
        assert_eq!(private_memory.checkpoints.len(), 2);
        assert_eq!(private_memory.last_checkpoint, 32);
        assert_eq!(private_memory.len(), 64);

        private_memory.free_context();
        assert_eq!(private_memory.buffer.len(), 32);
        assert_eq!(private_memory.checkpoints.len(), 1);
        assert_eq!(private_memory.last_checkpoint, 0);
        assert_eq!(private_memory.len(), 32);

        private_memory.free_context();
        assert_eq!(private_memory.buffer.len(), 0);
        assert_eq!(private_memory.checkpoints.len(), 0);
        assert_eq!(private_memory.last_checkpoint, 0);
        assert_eq!(private_memory.len(), 0);
    }

    // #[test]
    // fn resize() {
    //     let mut private_memory = PrivateMemory::new();
    //     private_memory.new_context();

    //     private_memory.resize(32);
    //     assert_eq!(private_memory.buffer.len(), 32);
    //     assert_eq!(private_memory.len(), 32);
    //     assert_eq!(private_memory.buffer.get(0..32), Some(&[0_u8; 32] as &[u8]));

    //     private_memory.new_context();
    //     private_memory.resize(96);
    //     assert_eq!(private_memory.buffer.len(), 128);
    //     assert_eq!(private_memory.len(), 96);
    //     assert_eq!(
    //         private_memory.buffer.get(32..128),
    //         Some(&[0_u8; 96] as &[u8])
    //     );

    //     private_memory.free_context();
    //     private_memory.resize(64);
    //     assert_eq!(private_memory.buffer.len(), 64);
    //     assert_eq!(private_memory.len(), 64);
    //     assert_eq!(private_memory.buffer.get(0..64), Some(&[0_u8; 64] as &[u8]));
    // }
}
