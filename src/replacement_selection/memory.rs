use core::panic;
use std::cell::RefCell;

use crate::diskio::constants::DEFAULT_BUFFER_SIZE;

const EXTENT_SIZE: usize = DEFAULT_BUFFER_SIZE;
const ALIGN: usize = 32;
const FREE_LISTS: usize = EXTENT_SIZE / ALIGN;
const HEADER_SIZE: usize = 4;
const FOOTER_SIZE: usize = 4;
const PTR_SIZE: usize = 8;
const MIN_BLOCK_SIZE: usize = ALIGN;
const NONE_U32: u32 = u32::MAX;
const TAG_FLAGS_MASK: u32 = 0x1F;
const TAG_ALLOCATED: u32 = 0x01;
const OFFSET_UNITS: usize = EXTENT_SIZE / ALIGN;
const OFFSET_BITS: u32 = OFFSET_UNITS.trailing_zeros();
const OFFSET_MASK: u32 = (1u32 << OFFSET_BITS) - 1;
const RESERVED_HANDLES: u32 = 3;
const MAX_HANDLE_VALUE: u32 = u32::MAX - RESERVED_HANDLES;
const MAX_EXTENTS: usize = ((MAX_HANDLE_VALUE as u64 + 1) >> OFFSET_BITS) as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocHandle(u32);

impl AllocHandle {
    /// Sentinel value representing no allocation (uses u32::MAX which is never valid).
    pub const NONE: Self = AllocHandle(u32::MAX);
    pub const EARLY: Self = AllocHandle(u32::MAX - 1);
    pub const LATE: Self = AllocHandle(u32::MAX - 2);

    /// Pack extent index and offset (in 32-byte units) into a u32 handle.
    fn new(extent: usize, offset: usize) -> Self {
        debug_assert!(offset % ALIGN == 0);
        debug_assert!(OFFSET_UNITS.is_power_of_two());
        let units = offset / ALIGN;
        debug_assert!(units < (1usize << OFFSET_BITS));
        debug_assert!(extent < MAX_EXTENTS);
        let value = ((extent as u32) << OFFSET_BITS) | units as u32;
        debug_assert!(value <= MAX_HANDLE_VALUE, "handle value reserved");
        AllocHandle(value)
    }

    /// Check if this handle is the sentinel NONE value.
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    pub fn is_early_fence(self) -> bool {
        self.0 == u32::MAX - 1
    }

    pub fn is_late_fence(self) -> bool {
        self.0 == u32::MAX - 2
    }

    pub fn is_sentinel(self) -> bool {
        self.is_early_fence() || self.is_late_fence()
    }

    pub fn is_empty(self) -> bool {
        self.is_none() || self.is_sentinel()
    }

    /// Check if this handle is a valid allocation.
    pub fn is_some(self) -> bool {
        !self.is_empty()
    }

    /// Return the extent index encoded in the handle.
    pub fn extent_index(self) -> usize {
        debug_assert!(!self.is_empty());
        (self.0 >> OFFSET_BITS) as usize
    }

    /// Return the byte offset encoded in the handle.
    pub fn offset_bytes(self) -> usize {
        debug_assert!(!self.is_empty());
        ((self.0 & OFFSET_MASK) as usize) * ALIGN
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlockPtr {
    extent: usize,
    offset: usize,
}

impl From<AllocHandle> for BlockPtr {
    fn from(handle: AllocHandle) -> Self {
        BlockPtr {
            extent: handle.extent_index(),
            offset: handle.offset_bytes(),
        }
    }
}

pub struct MemoryManager {
    extents: Vec<Vec<u8>>,
    free_lists: Vec<Option<BlockPtr>>,
    last_non_null: isize,
    max_bytes: Option<usize>,
    allocated_bytes: usize,
}

impl MemoryManager {
    /// Create a new allocator with one empty extent.
    pub fn new() -> Self {
        let mut manager = Self {
            extents: Vec::new(),
            free_lists: vec![None; FREE_LISTS],
            last_non_null: -1,
            max_bytes: None,
            allocated_bytes: 0,
        };
        let _ = manager.add_extent();
        manager
    }

    /// Create a new allocator with a maximum total byte capacity.
    /// The actual limit will be rounded down to a multiple of EXTENT_SIZE (64KB).
    pub fn with_limit(max_bytes: usize) -> Self {
        // Round down to the nearest extent boundary to ensure max_bytes
        // accurately reflects the actual usable memory
        let max_extents = max_bytes / EXTENT_SIZE;
        let adjusted_max_bytes = if max_extents > 0 {
            max_extents * EXTENT_SIZE
        } else {
            panic!("MemoryManager limit too small to allocate any extents");
        };

        let mut manager = Self {
            extents: Vec::new(),
            free_lists: vec![None; FREE_LISTS],
            last_non_null: -1,
            max_bytes: Some(adjusted_max_bytes),
            allocated_bytes: 0,
        };
        let _ = manager.add_extent();
        manager
    }

    /// Allocate a block with at least `size` bytes of payload.
    pub fn alloc(&mut self, size: usize) -> Option<AllocHandle> {
        let required = self.required_block_size(size)?;
        if let Some(ptr) = self.find_free_block(required) {
            let handle = self.allocate_from(ptr, required);
            self.allocated_bytes = self
                .allocated_bytes
                .saturating_add(self.allocation_size(handle));
            return Some(handle);
        }
        if self.add_extent().is_none() {
            return None;
        }
        let ptr = self.find_free_block(required)?;
        let handle = self.allocate_from(ptr, required);
        self.allocated_bytes = self
            .allocated_bytes
            .saturating_add(self.allocation_size(handle));
        Some(handle)
    }

    /// Free a previously allocated block and coalesce neighbors.
    pub fn free(&mut self, handle: AllocHandle) {
        let alloc_size = self.allocation_size(handle);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(alloc_size);
        let mut ptr = BlockPtr::from(handle);
        let (size, allocated) = self.read_block_tag(ptr);
        debug_assert!(allocated, "double free detected");

        // Mark this block as free before attempting any coalescing.
        self.write_block_tags(ptr, size, false);

        let mut new_offset = ptr.offset;
        let mut new_size = size;

        // Check the left neighbor using the footer immediately before this block.
        if ptr.offset >= FOOTER_SIZE {
            let left_footer = ptr.offset - FOOTER_SIZE;
            let (left_size, left_allocated) = self.read_block_tag_at(ptr.extent, left_footer);
            if !left_allocated {
                // Left block is free: unlink it and merge into this block.
                let left_offset = ptr.offset - left_size;
                let left_ptr = BlockPtr {
                    extent: ptr.extent,
                    offset: left_offset,
                };
                self.remove_free(left_ptr);
                new_offset = left_offset;
                new_size += left_size;
            }
        }

        // Check the right neighbor using the header immediately after this block.
        let right_offset = ptr.offset + size;
        if right_offset + HEADER_SIZE <= EXTENT_SIZE {
            let (right_size, right_allocated) = self.read_block_tag_at(ptr.extent, right_offset);
            if !right_allocated {
                // Right block is free: unlink it and merge into this block.
                let right_ptr = BlockPtr {
                    extent: ptr.extent,
                    offset: right_offset,
                };
                self.remove_free(right_ptr);
                new_size += right_size;
            }
        }

        // Write the merged block's tags and reinsert it into the free lists.
        ptr.offset = new_offset;
        self.write_block_tags(ptr, new_size, false);
        self.insert_free(ptr);
    }

    /// Return the usable payload capacity of an allocated block.
    #[allow(dead_code)]
    pub fn payload_capacity(&self, handle: AllocHandle) -> usize {
        let ptr = BlockPtr::from(handle);
        let (size, allocated) = self.read_block_tag(ptr);
        debug_assert!(allocated, "payload_capacity on free block");
        size - HEADER_SIZE - FOOTER_SIZE
    }

    /// Return the total block size (including header/footer) for an allocation.
    pub fn allocation_size(&self, handle: AllocHandle) -> usize {
        let ptr = BlockPtr::from(handle);
        let (size, allocated) = self.read_block_tag(ptr);
        debug_assert!(allocated, "allocation_size on free block");
        size
    }

    #[allow(dead_code)]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    pub fn total_bytes(&self) -> usize {
        if let Some(max_bytes) = self.max_bytes {
            max_bytes
        } else {
            self.extents.len().saturating_mul(EXTENT_SIZE)
        }
    }

    pub fn has_headroom(&self) -> bool {
        let total = self.total_bytes();
        if total == 0 {
            return false;
        }
        self.allocated_bytes < total.saturating_mul(95) / 100
    }

    /// Raise the maximum byte capacity.  Only allows future extent growth;
    /// existing extents and live allocations are unaffected.
    /// Panics in debug builds if `new_max` is smaller than the current limit.
    pub fn grow_limit(&mut self, new_max: usize) {
        let new_extents = new_max / EXTENT_SIZE;
        debug_assert!(
            new_extents >= self.max_bytes.map_or(0, |m| m / EXTENT_SIZE),
            "grow_limit called with a smaller limit"
        );
        if new_extents > 0 {
            self.max_bytes = Some(new_extents * EXTENT_SIZE);
        }
    }

    /// Print a summary of free slots grouped by size.
    #[allow(dead_code)]
    pub fn print_free_slots(&self) {
        let mut total_blocks = 0;
        let mut total_bytes = 0;
        let mut size_groups: Vec<(usize, usize)> = Vec::new();

        for (index, &head) in self.free_lists.iter().enumerate() {
            if head.is_none() {
                continue;
            }

            let size = (index + 1) * ALIGN;
            let mut count = 0;
            let mut current = head;

            // Walk the linked list to count blocks
            while let Some(ptr) = current {
                count += 1;
                current = self.read_free_next(ptr);
            }

            total_blocks += count;
            total_bytes += size * count;
            size_groups.push((size, count));
        }

        println!(
            "Free slots: {} blocks, {} bytes total",
            total_blocks, total_bytes
        );
        for (size, count) in size_groups {
            let size_str = if size >= 1024 {
                format!("{}KB", size / 1024)
            } else {
                format!("{}B", size)
            };
            println!(
                "  {}: {} block{}",
                size_str,
                count,
                if count != 1 { "s" } else { "" }
            );
        }
    }

    /// Borrow the payload of an allocated block.
    pub fn payload(&self, handle: AllocHandle) -> &[u8] {
        let ptr = BlockPtr::from(handle);
        let (size, allocated) = self.read_block_tag(ptr);
        debug_assert!(allocated, "payload on free block");
        let payload_offset = ptr.offset + HEADER_SIZE;
        let payload_len = size - HEADER_SIZE - FOOTER_SIZE;
        &self.extents[ptr.extent][payload_offset..payload_offset + payload_len]
    }

    /// Mutably borrow the payload of an allocated block.
    pub fn payload_mut(&mut self, handle: AllocHandle) -> &mut [u8] {
        let ptr = BlockPtr::from(handle);
        let (size, allocated) = self.read_block_tag(ptr);
        debug_assert!(allocated, "payload_mut on free block");
        let payload_offset = ptr.offset + HEADER_SIZE;
        let payload_len = size - HEADER_SIZE - FOOTER_SIZE;
        &mut self.extents[ptr.extent][payload_offset..payload_offset + payload_len]
    }

    /// Compute the full block size (including tags) for a payload request.
    fn required_block_size(&self, request: usize) -> Option<usize> {
        let total = request.checked_add(HEADER_SIZE + FOOTER_SIZE)?;
        let aligned = align_up(total, ALIGN).max(MIN_BLOCK_SIZE);
        if aligned > EXTENT_SIZE {
            None
        } else {
            Some(aligned)
        }
    }

    /// Append a new extent and insert it as a single free block.
    fn add_extent(&mut self) -> Option<()> {
        if let Some(max_bytes) = self.max_bytes {
            let next_total = (self.extents.len() + 1).saturating_mul(EXTENT_SIZE);
            if next_total > max_bytes {
                return None;
            }
        }
        if self.extents.len() >= MAX_EXTENTS {
            return None;
        }
        let extent_id = self.extents.len();
        self.extents.push(vec![0u8; EXTENT_SIZE]);
        let ptr = BlockPtr {
            extent: extent_id,
            offset: 0,
        };
        self.write_block_tags(ptr, EXTENT_SIZE, false);
        self.write_free_prev(ptr, None);
        self.write_free_next(ptr, None);
        self.insert_free(ptr);
        Some(())
    }

    /// Allocate from a chosen free block, splitting if needed.
    fn allocate_from(&mut self, ptr: BlockPtr, required: usize) -> AllocHandle {
        let (size, _) = self.read_block_tag(ptr);
        self.remove_free(ptr);

        let remainder = size.saturating_sub(required);
        if remainder >= MIN_BLOCK_SIZE {
            self.write_block_tags(ptr, required, true);

            let rem_ptr = BlockPtr {
                extent: ptr.extent,
                offset: ptr.offset + required,
            };
            self.write_block_tags(rem_ptr, remainder, false);
            self.write_free_prev(rem_ptr, None);
            self.write_free_next(rem_ptr, None);
            self.insert_free(rem_ptr);
        } else {
            self.write_block_tags(ptr, size, true);
        }

        AllocHandle::new(ptr.extent, ptr.offset)
    }

    /// Find the first free block in the smallest suitable size class.
    fn find_free_block(&mut self, required: usize) -> Option<BlockPtr> {
        let start = list_index_for_size(required);
        if self.last_non_null < 0 || start as isize > self.last_non_null {
            return None;
        }
        for i in start..=self.last_non_null as usize {
            if let Some(ptr) = self.free_lists[i] {
                return Some(ptr);
            }
        }
        self.last_non_null = start as isize - 1;
        None
    }

    /// Insert a free block at the head of its size class list.
    fn insert_free(&mut self, ptr: BlockPtr) {
        let (size, allocated) = self.read_block_tag(ptr);
        debug_assert!(!allocated, "insert_free on allocated block");
        debug_assert!(size >= MIN_BLOCK_SIZE, "block too small for free list");
        let index = list_index_for_size(size);

        let head = self.free_lists[index];
        self.write_free_prev(ptr, None);
        self.write_free_next(ptr, head);
        if let Some(head_ptr) = head {
            self.write_free_prev(head_ptr, Some(ptr));
        }
        self.free_lists[index] = Some(ptr);
        if index as isize > self.last_non_null {
            self.last_non_null = index as isize;
        }
    }

    /// Remove a free block from its size class list.
    fn remove_free(&mut self, ptr: BlockPtr) {
        let (size, allocated) = self.read_block_tag(ptr);
        debug_assert!(!allocated, "remove_free on allocated block");
        let index = list_index_for_size(size);

        let prev = self.read_free_prev(ptr);
        let next = self.read_free_next(ptr);

        if let Some(prev_ptr) = prev {
            self.write_free_next(prev_ptr, next);
        } else {
            self.free_lists[index] = next;
        }
        if let Some(next_ptr) = next {
            self.write_free_prev(next_ptr, prev);
        }

        self.write_free_prev(ptr, None);
        self.write_free_next(ptr, None);

        if self.free_lists[index].is_none() && index as isize == self.last_non_null {
            self.update_last_non_null();
        }
    }

    /// Recompute the highest non-empty free list index.
    fn update_last_non_null(&mut self) {
        let mut idx = self.last_non_null;
        while idx >= 0 {
            if self.free_lists[idx as usize].is_some() {
                self.last_non_null = idx;
                return;
            }
            idx -= 1;
        }
        self.last_non_null = -1;
    }

    /// Write header/footer tags for a block.
    fn write_block_tags(&mut self, ptr: BlockPtr, size: usize, allocated: bool) {
        debug_assert!(size % ALIGN == 0);
        self.write_block_tag(ptr, size, allocated);
        let footer_offset = ptr.offset + size - FOOTER_SIZE;
        self.write_block_tag_at(ptr.extent, footer_offset, size, allocated);
    }

    /// Read the block tag at a block pointer.
    fn read_block_tag(&self, ptr: BlockPtr) -> (usize, bool) {
        self.read_block_tag_at(ptr.extent, ptr.offset)
    }

    /// Read a block tag at an arbitrary offset within an extent.
    fn read_block_tag_at(&self, extent: usize, offset: usize) -> (usize, bool) {
        let tag = self.read_u32(extent, offset);
        let size = (tag & !TAG_FLAGS_MASK) as usize;
        let allocated = (tag & TAG_ALLOCATED) != 0;
        (size, allocated)
    }

    /// Write the block tag for a block header.
    fn write_block_tag(&mut self, ptr: BlockPtr, size: usize, allocated: bool) {
        self.write_block_tag_at(ptr.extent, ptr.offset, size, allocated);
    }

    /// Write a block tag at an arbitrary offset within an extent.
    fn write_block_tag_at(&mut self, extent: usize, offset: usize, size: usize, allocated: bool) {
        let tag = (size as u32) | if allocated { TAG_ALLOCATED } else { 0 };
        self.write_u32(extent, offset, tag);
    }

    /// Read the prev pointer from a free block's payload.
    fn read_free_prev(&self, ptr: BlockPtr) -> Option<BlockPtr> {
        self.read_ptr(ptr.extent, ptr.offset + HEADER_SIZE)
    }

    /// Read the next pointer from a free block's payload.
    fn read_free_next(&self, ptr: BlockPtr) -> Option<BlockPtr> {
        self.read_ptr(ptr.extent, ptr.offset + HEADER_SIZE + PTR_SIZE)
    }

    /// Write the prev pointer in a free block's payload.
    fn write_free_prev(&mut self, ptr: BlockPtr, value: Option<BlockPtr>) {
        self.write_ptr(ptr.extent, ptr.offset + HEADER_SIZE, value);
    }

    /// Write the next pointer in a free block's payload.
    fn write_free_next(&mut self, ptr: BlockPtr, value: Option<BlockPtr>) {
        self.write_ptr(ptr.extent, ptr.offset + HEADER_SIZE + PTR_SIZE, value);
    }

    /// Read an encoded BlockPtr stored as two u32s.
    fn read_ptr(&self, extent: usize, offset: usize) -> Option<BlockPtr> {
        let raw_extent = self.read_u32(extent, offset);
        let raw_offset = self.read_u32(extent, offset + 4);
        if raw_extent == NONE_U32 {
            None
        } else {
            Some(BlockPtr {
                extent: raw_extent as usize,
                offset: raw_offset as usize,
            })
        }
    }

    /// Write an encoded BlockPtr stored as two u32s.
    fn write_ptr(&mut self, extent: usize, offset: usize, ptr: Option<BlockPtr>) {
        match ptr {
            Some(value) => {
                self.write_u32(extent, offset, value.extent as u32);
                self.write_u32(extent, offset + 4, value.offset as u32);
            }
            None => {
                self.write_u32(extent, offset, NONE_U32);
                self.write_u32(extent, offset + 4, NONE_U32);
            }
        }
    }

    /// Read a little-endian u32 from the extent.
    fn read_u32(&self, extent: usize, offset: usize) -> u32 {
        let bytes = &self.extents[extent][offset..offset + 4];
        u32::from_le_bytes(bytes.try_into().expect("u32 slice"))
    }

    /// Write a little-endian u32 to the extent.
    fn write_u32(&mut self, extent: usize, offset: usize, value: u32) {
        let bytes = value.to_le_bytes();
        self.extents[extent][offset..offset + 4].copy_from_slice(&bytes);
    }
}

/// Map a block size to its free list index.
fn list_index_for_size(size: usize) -> usize {
    debug_assert!(size % ALIGN == 0);
    debug_assert!(size >= MIN_BLOCK_SIZE);
    (size / ALIGN) - 1
}

/// Round up to the next multiple of `align`.
fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

thread_local! {
    static TLS_MEMORY_MANAGER: RefCell<MemoryManager> =
        RefCell::new(MemoryManager::new());
}

pub(crate) fn reset_thread_local_manager() {
    TLS_MEMORY_MANAGER.with(|manager| {
        *manager.borrow_mut() = MemoryManager::new();
    });
}

pub(crate) fn reset_thread_local_manager_with_limit(max_bytes: usize) {
    TLS_MEMORY_MANAGER.with(|manager| {
        *manager.borrow_mut() = MemoryManager::with_limit(max_bytes);
    });
}

pub(crate) fn tls_has_headroom() -> bool {
    TLS_MEMORY_MANAGER.with(|manager| manager.borrow().has_headroom())
}

pub(crate) fn tls_grow_limit(new_max: usize) {
    TLS_MEMORY_MANAGER.with(|manager| manager.borrow_mut().grow_limit(new_max));
}

pub(crate) fn tls_allocated_bytes() -> usize {
    TLS_MEMORY_MANAGER.with(|manager| manager.borrow().allocated_bytes())
}

pub(crate) struct ManagedSlice {
    handle: AllocHandle,
}

impl ManagedSlice {
    pub(crate) fn empty() -> Self {
        Self {
            handle: AllocHandle::NONE,
        }
    }

    pub(crate) fn early_fence() -> Self {
        Self {
            handle: AllocHandle::EARLY,
        }
    }

    pub(crate) fn late_fence() -> Self {
        Self {
            handle: AllocHandle::LATE,
        }
    }

    pub(crate) fn alloc(key: &[u8], value: &[u8]) -> Option<Self> {
        let key_len = u16::try_from(key.len()).ok()?;
        let value_len = u16::try_from(value.len()).ok()?;
        let total = key.len().checked_add(value.len())?.checked_add(4)?;

        let key_len_bytes = key_len.to_le_bytes();
        let value_len_bytes = value_len.to_le_bytes();

        let handle = TLS_MEMORY_MANAGER.with(|manager| {
            let mut manager = manager.borrow_mut();
            let handle = manager.alloc(total)?;
            let payload = manager.payload_mut(handle);
            payload[..2].copy_from_slice(&key_len_bytes);
            payload[2..2 + key.len()].copy_from_slice(key);
            let value_len_offset = 2 + key.len();
            payload[value_len_offset..value_len_offset + 2].copy_from_slice(&value_len_bytes);
            payload[value_len_offset + 2..total].copy_from_slice(value);
            Some(handle)
        })?;

        Some(Self { handle })
    }

    pub(crate) fn payload_info(&self) -> Option<(*const u8, usize, usize, usize)> {
        if self.handle.is_empty() {
            return None;
        }
        TLS_MEMORY_MANAGER.with(|manager| {
            let manager = manager.borrow();
            let payload = manager.payload(self.handle);
            if payload.len() < 4 {
                return None;
            }
            let key_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
            let value_len_offset = 2usize.saturating_add(key_len);
            if payload.len() < value_len_offset + 2 {
                return None;
            }
            let value_len =
                u16::from_le_bytes([payload[value_len_offset], payload[value_len_offset + 1]])
                    as usize;
            let total = value_len_offset + 2 + value_len;
            debug_assert!(payload.len() >= total, "payload smaller than key+value");
            Some((payload.as_ptr(), key_len, value_len, total))
        })
    }

    pub(crate) fn allocation_size(&self) -> usize {
        if self.handle.is_empty() {
            return 0;
        }
        TLS_MEMORY_MANAGER.with(|manager| manager.borrow().allocation_size(self.handle))
    }

    pub(crate) fn release(&mut self) {
        if self.handle.is_some() {
            TLS_MEMORY_MANAGER.with(|manager| manager.borrow_mut().free(self.handle));
            self.handle = AllocHandle::NONE;
        }
    }

    pub(crate) fn is_early_fence(&self) -> bool {
        self.handle.is_early_fence()
    }

    pub(crate) fn is_late_fence(&self) -> bool {
        self.handle.is_late_fence()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_size(request: usize) -> usize {
        align_up(request + HEADER_SIZE + FOOTER_SIZE, ALIGN).max(MIN_BLOCK_SIZE)
    }

    #[test]
    fn alloc_and_free_roundtrip() {
        let mut mem = MemoryManager::new();
        let handle = mem.alloc(100).expect("alloc");
        assert!(mem.payload_capacity(handle) >= 100);
        assert!(mem.payload(handle).len() >= 100);
        mem.free(handle);
    }

    #[test]
    fn coalesces_adjacent_free_blocks() {
        let mut mem = MemoryManager::new();
        let a = mem.alloc(512).expect("alloc a");
        let b = mem.alloc(512).expect("alloc b");
        mem.free(a);
        mem.free(b);
        let almost_full = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE - ALIGN;
        let big = mem.alloc(almost_full).expect("alloc big after coalesce");
        mem.free(big);
    }

    #[test]
    fn alloc_rejects_oversize_request() {
        let mut mem = MemoryManager::new();
        assert!(mem.alloc(EXTENT_SIZE).is_none());
    }

    #[test]
    fn splits_block_and_tracks_remainder() {
        let mut mem = MemoryManager::new();
        let request = 64;
        let required = required_size(request);
        let handle = mem.alloc(request).expect("alloc");
        assert_eq!(handle.offset_bytes(), 0);
        let remainder_size = EXTENT_SIZE - required;
        let remainder_index = list_index_for_size(remainder_size);
        let remainder_ptr = mem.free_lists[remainder_index].expect("remainder");
        assert_eq!(remainder_ptr.extent, 0);
        assert_eq!(remainder_ptr.offset, required);
        mem.free(handle);
    }

    #[test]
    fn coalesces_left_and_right_neighbors() {
        let mut mem = MemoryManager::new();
        let a = mem.alloc(200).expect("alloc a");
        let b = mem.alloc(200).expect("alloc b");
        let c = mem.alloc(200).expect("alloc c");
        mem.free(a);
        mem.free(c);
        mem.free(b);
        let index = list_index_for_size(EXTENT_SIZE);
        let head = mem.free_lists[index].expect("full free block");
        assert_eq!(head.offset, 0);
        assert_eq!(head.extent, 0);
        for (i, entry) in mem.free_lists.iter().enumerate() {
            if i == index {
                continue;
            }
            assert!(entry.is_none());
        }
    }

    #[test]
    fn last_non_null_updates_on_alloc_and_free() {
        let mut mem = MemoryManager::new();
        let first = EXTENT_SIZE - ALIGN - HEADER_SIZE - FOOTER_SIZE;
        let _handle = mem.alloc(first).expect("alloc large");
        assert_eq!(mem.last_non_null, 0);
        let _handle2 = mem.alloc(1).expect("alloc remainder");
        assert_eq!(mem.last_non_null, -1);
    }

    #[test]
    fn allocates_from_best_fit_block() {
        let mut mem = MemoryManager::new();
        let a_req = 32;
        let b_req = 70;
        let c_req = 200;
        let a = mem.alloc(a_req).expect("alloc a");
        let b = mem.alloc(b_req).expect("alloc b");
        let c = mem.alloc(c_req).expect("alloc c");

        let a_size = required_size(a_req);
        let b_size = required_size(b_req);
        assert_eq!(a.offset_bytes(), 0);
        assert_eq!(b.offset_bytes(), a_size);
        assert_eq!(c.offset_bytes(), a_size + b_size);

        mem.free(b);
        mem.free(c);

        let pick = mem.alloc(60).expect("alloc pick");
        assert_eq!(pick.offset_bytes(), a_size);
        mem.free(a);
        mem.free(pick);
    }

    #[test]
    fn grows_with_additional_extents() {
        let mut mem = MemoryManager::new();
        let first = EXTENT_SIZE - ALIGN - HEADER_SIZE - FOOTER_SIZE;
        let _a = mem.alloc(first).expect("alloc large");
        let _b = mem.alloc(128).expect("alloc new extent");
        assert_eq!(mem.extents.len(), 2);
    }

    // ==================== Handle Encoding Tests ====================

    #[test]
    fn alloc_handle_roundtrip() {
        for extent in [0, 1, 10, 100, MAX_EXTENTS - 1] {
            for offset in [0, ALIGN, ALIGN * 10, EXTENT_SIZE - ALIGN] {
                let handle = AllocHandle::new(extent, offset);
                assert_eq!(handle.extent_index(), extent);
                assert_eq!(handle.offset_bytes(), offset);
            }
        }
    }

    #[test]
    fn alloc_handle_boundary_values() {
        // Minimum values
        let min = AllocHandle::new(0, 0);
        assert_eq!(min.extent_index(), 0);
        assert_eq!(min.offset_bytes(), 0);

        // Maximum extent index
        let max_extent = AllocHandle::new(MAX_EXTENTS - 1, 0);
        assert_eq!(max_extent.extent_index(), MAX_EXTENTS - 1);

        // Maximum offset
        let max_offset = AllocHandle::new(0, EXTENT_SIZE - ALIGN);
        assert_eq!(max_offset.offset_bytes(), EXTENT_SIZE - ALIGN);
    }

    // ==================== Zero and Minimum Size Tests ====================

    #[test]
    fn alloc_zero_size() {
        let mut mem = MemoryManager::new();
        let handle = mem.alloc(0).expect("alloc zero");
        // Should still get minimum block size with usable payload
        let capacity = mem.payload_capacity(handle);
        assert!(
            capacity > 0,
            "zero-size alloc should have positive capacity"
        );
        mem.free(handle);
    }

    #[test]
    fn alloc_one_byte() {
        let mut mem = MemoryManager::new();
        let handle = mem.alloc(1).expect("alloc one byte");
        assert!(mem.payload_capacity(handle) >= 1);
        let payload = mem.payload_mut(handle);
        payload[0] = 0xAB;
        assert_eq!(mem.payload(handle)[0], 0xAB);
        mem.free(handle);
    }

    #[test]
    fn alloc_minimum_payload_size() {
        let mut mem = MemoryManager::new();
        // Minimum block is ALIGN bytes, minus header and footer
        let min_payload = MIN_BLOCK_SIZE.saturating_sub(HEADER_SIZE + FOOTER_SIZE);
        let handle = mem.alloc(min_payload).expect("alloc min");
        assert!(mem.payload_capacity(handle) >= min_payload);
        mem.free(handle);
    }

    // ==================== Maximum Size Tests ====================

    #[test]
    fn alloc_maximum_valid_size() {
        let mut mem = MemoryManager::new();
        let max_payload = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let handle = mem.alloc(max_payload).expect("alloc max");
        assert!(mem.payload_capacity(handle) >= max_payload);
        mem.free(handle);
    }

    #[test]
    fn alloc_just_over_max_fails() {
        let mut mem = MemoryManager::new();
        let too_large = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE + 1;
        assert!(mem.alloc(too_large).is_none());
    }

    #[test]
    fn alloc_huge_size_fails() {
        let mut mem = MemoryManager::new();
        assert!(mem.alloc(usize::MAX).is_none());
        assert!(mem.alloc(usize::MAX / 2).is_none());
    }

    // ==================== Payload Read/Write Tests ====================

    #[test]
    fn payload_write_and_read() {
        let mut mem = MemoryManager::new();
        let handle = mem.alloc(256).expect("alloc");
        let payload = mem.payload_mut(handle);
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let payload_read = mem.payload(handle);
        for (i, byte) in payload_read.iter().enumerate() {
            assert_eq!(*byte, (i % 256) as u8);
        }
        mem.free(handle);
    }

    #[test]
    fn payload_persists_across_other_allocations() {
        let mut mem = MemoryManager::new();
        let h1 = mem.alloc(64).expect("alloc h1");
        mem.payload_mut(h1).fill(0xAA);

        let h2 = mem.alloc(64).expect("alloc h2");
        mem.payload_mut(h2).fill(0xBB);

        let h3 = mem.alloc(64).expect("alloc h3");
        mem.payload_mut(h3).fill(0xCC);

        // Verify data persists
        assert!(mem.payload(h1).iter().all(|&b| b == 0xAA));
        assert!(mem.payload(h2).iter().all(|&b| b == 0xBB));
        assert!(mem.payload(h3).iter().all(|&b| b == 0xCC));

        mem.free(h2);

        // h1 and h3 should still be intact
        assert!(mem.payload(h1).iter().all(|&b| b == 0xAA));
        assert!(mem.payload(h3).iter().all(|&b| b == 0xCC));

        mem.free(h1);
        mem.free(h3);
    }

    #[test]
    fn payload_capacity_matches_or_exceeds_request() {
        let mut mem = MemoryManager::new();
        for size in [1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 255, 256, 1000] {
            let handle = mem.alloc(size).expect("alloc");
            assert!(
                mem.payload_capacity(handle) >= size,
                "capacity {} < request {}",
                mem.payload_capacity(handle),
                size
            );
            mem.free(handle);
        }
    }

    // ==================== Coalescing Tests ====================

    #[test]
    fn coalesce_left_only() {
        let mut mem = MemoryManager::new();
        let a = mem.alloc(128).expect("alloc a");
        let b = mem.alloc(128).expect("alloc b");
        let _c = mem.alloc(128).expect("alloc c"); // Keep c allocated

        mem.free(a);
        mem.free(b); // Should coalesce with a

        // Allocate something that fits in merged a+b
        let a_size = required_size(128);
        let merged_payload = 2 * a_size - HEADER_SIZE - FOOTER_SIZE;
        let merged = mem.alloc(merged_payload).expect("alloc merged");
        assert_eq!(merged.offset_bytes(), 0);
        mem.free(merged);
    }

    #[test]
    fn coalesce_right_only() {
        let mut mem = MemoryManager::new();
        let _a = mem.alloc(128).expect("alloc a"); // Keep a allocated
        let b = mem.alloc(128).expect("alloc b");
        let c = mem.alloc(128).expect("alloc c");

        mem.free(c);
        mem.free(b); // Should coalesce with c

        let b_size = required_size(128);
        let merged_payload = 2 * b_size - HEADER_SIZE - FOOTER_SIZE;
        let merged = mem.alloc(merged_payload).expect("alloc merged");
        assert_eq!(merged.offset_bytes(), b_size); // Should start at b's position
        mem.free(merged);
    }

    #[test]
    fn coalesce_both_sides() {
        let mut mem = MemoryManager::new();
        let a = mem.alloc(128).expect("alloc a");
        let b = mem.alloc(128).expect("alloc b");
        let c = mem.alloc(128).expect("alloc c");

        mem.free(a);
        mem.free(c);
        mem.free(b); // Should coalesce with both a and c

        let block_size = required_size(128);
        let merged_payload = 3 * block_size - HEADER_SIZE - FOOTER_SIZE;
        let merged = mem.alloc(merged_payload).expect("alloc merged");
        assert_eq!(merged.offset_bytes(), 0);
        mem.free(merged);
    }

    #[test]
    fn no_coalesce_with_allocated_neighbors() {
        let mut mem = MemoryManager::new();
        let a = mem.alloc(128).expect("alloc a");
        let b = mem.alloc(128).expect("alloc b");
        let c = mem.alloc(128).expect("alloc c");

        mem.free(b); // a and c are still allocated, no coalescing

        let b_size = required_size(128);
        let b_payload = b_size - HEADER_SIZE - FOOTER_SIZE;
        let realloc = mem.alloc(b_payload).expect("realloc b slot");
        assert_eq!(realloc.offset_bytes(), b.offset_bytes());

        mem.free(a);
        mem.free(c);
        mem.free(realloc);
    }

    #[test]
    fn coalesce_full_extent_from_many_blocks() {
        let mut mem = MemoryManager::new();
        let block_size = required_size(100);
        let count = EXTENT_SIZE / block_size;
        let mut handles = Vec::new();

        for _ in 0..count {
            if let Some(h) = mem.alloc(100) {
                handles.push(h);
            }
        }

        // Free all in reverse order
        for h in handles.into_iter().rev() {
            mem.free(h);
        }

        // Should have coalesced back to full extent
        let index = list_index_for_size(EXTENT_SIZE);
        assert!(mem.free_lists[index].is_some());
    }

    #[test]
    fn coalesce_interleaved_free() {
        let mut mem = MemoryManager::new();
        let a = mem.alloc(64).expect("a");
        let b = mem.alloc(64).expect("b");
        let c = mem.alloc(64).expect("c");
        let d = mem.alloc(64).expect("d");
        let e = mem.alloc(64).expect("e");

        // Free odd positions first
        mem.free(a);
        mem.free(c);
        mem.free(e);

        // Free even positions - this causes coalescing
        mem.free(b);
        mem.free(d);

        // After all frees, the blocks should have coalesced. The exact final
        // size depends on whether there was a remainder block after the 5 allocations.
        // Verify we can allocate a large block that spans the freed space.
        let block_size = required_size(64);
        let expected_payload = 5 * block_size - HEADER_SIZE - FOOTER_SIZE;
        let big = mem.alloc(expected_payload).expect("alloc coalesced space");
        assert_eq!(big.offset_bytes(), 0);
        mem.free(big);
    }

    // ==================== Multi-Extent Tests ====================

    #[test]
    fn allocate_across_multiple_extents() {
        let mut mem = MemoryManager::new();
        let large = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let h1 = mem.alloc(large).expect("alloc extent 1");
        assert_eq!(h1.extent_index(), 0);

        let h2 = mem.alloc(large).expect("alloc extent 2");
        assert_eq!(h2.extent_index(), 1);

        let h3 = mem.alloc(large).expect("alloc extent 3");
        assert_eq!(h3.extent_index(), 2);

        assert_eq!(mem.extents.len(), 3);

        mem.free(h2);
        mem.free(h1);
        mem.free(h3);
    }

    #[test]
    fn free_and_reuse_across_extents() {
        let mut mem = MemoryManager::new();
        let large = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE;

        let h1 = mem.alloc(large).expect("h1");
        let h2 = mem.alloc(large).expect("h2");

        mem.free(h1);

        // New allocation should reuse extent 0
        let h3 = mem.alloc(large).expect("h3");
        assert_eq!(h3.extent_index(), 0);

        mem.free(h2);
        mem.free(h3);
    }

    #[test]
    fn mixed_sizes_across_extents() {
        let mut mem = MemoryManager::new();
        let mut handles = Vec::new();

        // Fill first extent with small blocks
        let small_size = required_size(64);
        let per_extent = EXTENT_SIZE / small_size;

        for _ in 0..(per_extent * 2) {
            if let Some(h) = mem.alloc(64) {
                handles.push(h);
            }
        }

        assert!(mem.extents.len() >= 2);

        for h in handles {
            mem.free(h);
        }
    }

    // ==================== Free List Integrity Tests ====================

    #[test]
    fn free_list_empty_after_full_allocation() {
        let mut mem = MemoryManager::new();
        let max_payload = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let _h = mem.alloc(max_payload).expect("alloc full");

        // All free lists should be empty
        for entry in &mem.free_lists {
            assert!(entry.is_none());
        }
        assert_eq!(mem.last_non_null, -1);
    }

    #[test]
    fn free_list_head_updated_correctly() {
        let mut mem = MemoryManager::new();
        // Fill the extent to avoid remainder blocks that could coalesce
        let block_size = required_size(64);
        let blocks_needed = EXTENT_SIZE / block_size;

        let mut handles = Vec::new();
        for _ in 0..blocks_needed {
            if let Some(h) = mem.alloc(64) {
                handles.push(h);
            }
        }

        // Now free first block - it becomes head of its size class
        let a = handles[0];
        mem.free(a);
        let index = list_index_for_size(block_size);
        assert_eq!(mem.free_lists[index].unwrap().offset, a.offset_bytes());

        // Free a non-adjacent block (skip one) - it becomes new head
        let c = handles[2];
        mem.free(c);
        assert_eq!(mem.free_lists[index].unwrap().offset, c.offset_bytes());

        // Free remaining (will cause coalescing but that's ok)
        for (i, h) in handles.into_iter().enumerate() {
            if i != 0 && i != 2 {
                mem.free(h);
            }
        }
    }

    #[test]
    fn last_non_null_tracks_correctly() {
        let mut mem = MemoryManager::new();

        // Initially should point to full extent size class
        let initial_index = list_index_for_size(EXTENT_SIZE);
        assert_eq!(mem.last_non_null, initial_index as isize);

        // Allocate everything
        let max_payload = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let _h = mem.alloc(max_payload).expect("alloc");
        assert_eq!(mem.last_non_null, -1);
    }

    // ==================== Stress Tests ====================

    #[test]
    fn stress_many_small_allocations() {
        let mut mem = MemoryManager::new();
        let mut handles = Vec::new();

        for i in 0..1000 {
            if let Some(h) = mem.alloc(32) {
                let payload = mem.payload_mut(h);
                payload[0] = (i % 256) as u8;
                handles.push((h, (i % 256) as u8));
            }
        }

        // Verify all data
        for (h, expected) in &handles {
            assert_eq!(mem.payload(*h)[0], *expected);
        }

        // Free all
        for (h, _) in handles {
            mem.free(h);
        }
    }

    #[test]
    fn stress_varied_sizes() {
        let mut mem = MemoryManager::new();
        let sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
        let mut handles = Vec::new();

        for &size in sizes.iter().cycle().take(200) {
            if let Some(h) = mem.alloc(size) {
                handles.push(h);
            }
        }

        for h in handles {
            mem.free(h);
        }
    }

    #[test]
    fn stress_alloc_free_interleaved() {
        let mut mem = MemoryManager::new();
        let mut handles = Vec::new();

        for i in 0..500 {
            let h = mem.alloc(64).expect("alloc");
            handles.push(h);

            // Free every third allocation
            if i % 3 == 0 && !handles.is_empty() {
                let to_free = handles.remove(0);
                mem.free(to_free);
            }
        }

        for h in handles {
            mem.free(h);
        }
    }

    #[test]
    fn stress_random_pattern() {
        let mut mem = MemoryManager::new();
        let mut handles = Vec::new();

        // Pseudo-random using linear congruential generator
        let mut seed: u32 = 12345;
        let next_rand = |s: &mut u32| {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *s
        };

        for _ in 0..1000 {
            let r = next_rand(&mut seed);
            if r % 3 != 0 || handles.is_empty() {
                let size = (r % 512 + 1) as usize;
                if let Some(h) = mem.alloc(size) {
                    handles.push(h);
                }
            } else {
                let idx = (r as usize) % handles.len();
                let h = handles.swap_remove(idx);
                mem.free(h);
            }
        }

        for h in handles {
            mem.free(h);
        }
    }

    // ==================== Edge Cases ====================

    #[test]
    fn alloc_exact_block_sizes() {
        let mut mem = MemoryManager::new();
        // Allocate sizes that result in exact block sizes (no wasted space)
        for mult in 1..10 {
            let block_size = mult * ALIGN;
            let payload_size = block_size - HEADER_SIZE - FOOTER_SIZE;
            if payload_size > 0 {
                let h = mem.alloc(payload_size).expect("alloc exact");
                assert_eq!(mem.payload_capacity(h), payload_size);
                mem.free(h);
            }
        }
    }

    #[test]
    fn alloc_fills_exact_remainder() {
        let mut mem = MemoryManager::new();
        // Allocate so remainder is exactly MIN_BLOCK_SIZE
        let first_payload = EXTENT_SIZE - MIN_BLOCK_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let h1 = mem.alloc(first_payload).expect("first alloc");

        // Remainder should be MIN_BLOCK_SIZE
        let min_payload = MIN_BLOCK_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let h2 = mem.alloc(min_payload).expect("remainder alloc");

        mem.free(h1);
        mem.free(h2);
    }

    #[test]
    fn alloc_no_remainder() {
        let mut mem = MemoryManager::new();
        // Allocate exactly the full extent
        let full_payload = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let h = mem.alloc(full_payload).expect("full alloc");

        // No free space should remain
        assert_eq!(mem.last_non_null, -1);

        mem.free(h);
    }

    #[test]
    fn block_ptr_from_handle() {
        let handle = AllocHandle::new(5, 128);
        let ptr: BlockPtr = handle.into();
        assert_eq!(ptr.extent, 5);
        assert_eq!(ptr.offset, 128);
    }

    #[test]
    fn align_up_function() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(64, 32), 64);
    }

    #[test]
    fn list_index_mapping() {
        assert_eq!(list_index_for_size(MIN_BLOCK_SIZE), 0);
        assert_eq!(list_index_for_size(2 * ALIGN), 1);
        assert_eq!(list_index_for_size(EXTENT_SIZE), FREE_LISTS - 1);
    }

    // ==================== Fragmentation Tests ====================

    #[test]
    fn fragmentation_and_defragmentation() {
        let mut mem = MemoryManager::new();
        let mut handles = Vec::new();

        // Create fragmented memory by allocating many small blocks
        for _ in 0..50 {
            if let Some(h) = mem.alloc(64) {
                handles.push(h);
            }
        }

        // Free every other block to create fragmentation
        let to_free: Vec<_> = handles
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, h)| *h)
            .collect();

        for h in to_free {
            mem.free(h);
            handles.retain(|&x| x != h);
        }

        // Free remaining blocks - should coalesce
        for h in handles {
            mem.free(h);
        }

        // Should be able to allocate large block now
        let large = EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE - ALIGN;
        let h = mem.alloc(large).expect("large after defrag");
        mem.free(h);
    }

    #[test]
    fn worst_case_fragmentation() {
        let mut mem = MemoryManager::new();
        let mut handles = Vec::new();

        // Allocate alternating sizes
        for i in 0..20 {
            let size = if i % 2 == 0 { 64 } else { 256 };
            if let Some(h) = mem.alloc(size) {
                handles.push(h);
            }
        }

        // Free only the larger blocks
        let to_free: Vec<_> = handles
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, h)| *h)
            .collect();

        for h in &to_free {
            mem.free(*h);
        }
        handles.retain(|h| !to_free.contains(h));

        // Free remaining
        for h in handles {
            mem.free(h);
        }
    }

    // ==================== Reallocation Pattern Tests ====================

    #[test]
    fn reuse_freed_block_same_size() {
        let mut mem = MemoryManager::new();
        let h1 = mem.alloc(128).expect("first");
        let offset1 = h1.offset_bytes();
        mem.free(h1);

        let h2 = mem.alloc(128).expect("second");
        // Should reuse the same block
        assert_eq!(h2.offset_bytes(), offset1);
        mem.free(h2);
    }

    #[test]
    fn fifo_vs_lifo_reuse() {
        let mut mem = MemoryManager::new();
        // Fill extent to avoid remainder coalescing issues
        let block_size = required_size(64);
        let blocks_needed = EXTENT_SIZE / block_size;

        let mut handles = Vec::new();
        for _ in 0..blocks_needed {
            if let Some(h) = mem.alloc(64) {
                handles.push(h);
            }
        }

        let a = handles[0];
        let b = handles[1];
        let c = handles[2];

        let offset_a = a.offset_bytes();
        let offset_c = c.offset_bytes();

        mem.free(a);
        mem.free(c);

        // Free list is LIFO, so c should be reused first (most recently freed)
        let d = mem.alloc(64).expect("d");
        assert_eq!(d.offset_bytes(), offset_c);

        let e = mem.alloc(64).expect("e");
        assert_eq!(e.offset_bytes(), offset_a);

        // Free remaining handles
        mem.free(b);
        mem.free(d);
        mem.free(e);
        for h in handles.into_iter().skip(3) {
            mem.free(h);
        }
    }

    // ==================== Boundary Tests ====================

    #[test]
    fn allocate_at_extent_boundary() {
        let mut mem = MemoryManager::new();

        // Fill extent leaving exactly MIN_BLOCK_SIZE
        let first_size = EXTENT_SIZE - MIN_BLOCK_SIZE;
        let first_payload = first_size - HEADER_SIZE - FOOTER_SIZE;
        let h1 = mem.alloc(first_payload).expect("first");

        // Allocate the minimum block
        let min_payload = MIN_BLOCK_SIZE - HEADER_SIZE - FOOTER_SIZE;
        let h2 = mem.alloc(min_payload).expect("min at boundary");

        // Next allocation should go to new extent
        let h3 = mem.alloc(64).expect("new extent");
        assert_eq!(h3.extent_index(), 1);

        mem.free(h1);
        mem.free(h2);
        mem.free(h3);
    }

    #[test]
    fn consecutive_extent_allocations() {
        let mut mem = MemoryManager::new();
        let mut handles = Vec::new();

        // Allocate 10 full extents
        for i in 0..10 {
            let h = mem
                .alloc(EXTENT_SIZE - HEADER_SIZE - FOOTER_SIZE)
                .expect("full extent");
            assert_eq!(h.extent_index(), i);
            handles.push(h);
        }

        assert_eq!(mem.extents.len(), 10);

        for h in handles {
            mem.free(h);
        }
    }
}
