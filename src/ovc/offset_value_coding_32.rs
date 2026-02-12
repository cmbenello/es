use std::cmp::Ordering;
use std::mem;

use super::offset_value_coding_64::{OVCFlag, SentinelValue};
use crate::replacement_selection::RecordSize;

pub const OVC32_CHUNK_SIZE: usize = 2;

pub fn encode_run_with_ovc32(normalized_run: &[Vec<u8>]) -> Vec<OVCEntry32> {
    let mut result = Vec::with_capacity(normalized_run.len());

    for i in 0..normalized_run.len() {
        let mut new_entry = OVCEntry32::new(normalized_run[i].clone());

        if i > 0 {
            new_entry.derive_ovc_from(&result[i - 1]);
        } else {
            // The first entry is always an initial value
            new_entry.ovc = OVCU32::initial_value();
        }

        result.push(new_entry);
    }

    result
}

pub fn encode_runs_with_ovc32(normalized_runs: &[Vec<Vec<u8>>]) -> Vec<Vec<OVCEntry32>> {
    normalized_runs
        .iter()
        .map(|run| encode_run_with_ovc32(run))
        .collect::<Vec<_>>()
}

pub fn is_ovc_consistent(run: &[OVCEntry32]) -> bool {
    if run.is_empty() {
        return true;
    }

    for i in 1..run.len() {
        let prev_key = run[i - 1].get_key();
        let curr_key = run[i].get_key();
        let ovc = run[i].get_ovc();

        match ovc.flag() {
            OVCFlag::EarlyFence | OVCFlag::LateFence => {
                return false;
            }
            OVCFlag::DuplicateValue => {
                if prev_key != curr_key {
                    return false;
                }
                continue;
            }
            OVCFlag::InitialValue => {
                return false;
            }
            OVCFlag::NormalValue => {
                let stored_offset = ovc.offset();

                if stored_offset > prev_key.len() || stored_offset > curr_key.len() {
                    return false;
                }

                if prev_key[..stored_offset] != curr_key[..stored_offset] {
                    return false;
                }

                let stored_value = ovc.value();
                let actual_suffix = &curr_key[stored_offset..];
                let check_len = std::cmp::min(stored_value.len(), actual_suffix.len());

                if stored_value[..check_len] != actual_suffix[..check_len] {
                    return false;
                }

                if stored_value.len() > check_len
                    && stored_value[check_len..].iter().any(|&b| b != 0)
                {
                    return false;
                }
            }
        }
    }

    true
}

pub fn is_optimal_ovc(run: &[OVCEntry32]) -> bool {
    for i in 1..run.len() {
        let mut test_entry = OVCEntry32::new(run[i].get_key().to_vec());
        test_entry.derive_ovc_from(&run[i - 1]);
        if test_entry.get_ovc() != run[i].get_ovc() {
            return false;
        }
    }
    true
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct OVCU32(u32);

impl OVCU32 {
    // Memory layout (32 bits total):
    // ┌───────┬─────────────────────┬──────────────────┐
    // │ flags │ arity_minus_offset  │      value       │
    // └───────┴─────────────────────┴──────────────────┘
    //  bits 29-31      bits 16-28        bits 0-15
    //  (3 bits)        (13 bits)         (16 bits)
    //
    // Fields:
    // • value (bits 0-15): The actual value stored at this offset (2 bytes)
    //
    // • arity_minus_offset (bits 16-28): Encodes the distance from a reference point
    //   Calculated as: 0x1FFF - (actual_index_difference)
    //   This inverted encoding may be used for efficient comparisons
    //
    // • flags (bits 29-31): Identifies the type of this entry
    //   - 0: EARLY_FENCE
    //   - 1: DUPLICATE_VALUE
    //   - 2: NORMAL_VALUE
    //   - 3: INITIAL_VALUE
    //   - 4: LATE_FENCE

    const VALUE_MASK: u32 = 0x0000_FFFF; // 16 bits
    const ARITY_MASK: u32 = 0x1FFF; // 13 bits
    const ARITY_MAX: u16 = 0x1FFF; // 8191
    const ARITY_SHIFT: u32 = 16; // shift for arity_minus_offset (bits 16-28)
    const FLAG_MASK: u32 = 0x7; // 3 bits
    const FLAG_SHIFT: u32 = 29; // shift for flags (bits 29-31)

    pub fn new(value: &[u8], offset: usize, flag: OVCFlag) -> Self {
        debug_assert!(value.len() <= 2, "Value must be at most 2 bytes long");
        let mut value_u32 = 0_u32;
        for i in 0..value.len() {
            value_u32 |= (value[i] as u32) << (8 * (1 - i));
        }

        let arity_minus_offset = (Self::ARITY_MAX as usize)
            .saturating_sub(offset)
            .min(Self::ARITY_MAX as usize) as u32;

        let packed = value_u32
            | (arity_minus_offset << Self::ARITY_SHIFT)
            | ((flag.as_u8() as u32) << Self::FLAG_SHIFT);
        Self(packed)
    }

    pub fn early_fence() -> Self {
        let packed = (OVCFlag::EarlyFence.as_u8() as u32) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_early_fence(&self) -> bool {
        self.flag() == OVCFlag::EarlyFence
    }

    pub fn late_fence() -> Self {
        let packed = (OVCFlag::LateFence.as_u8() as u32) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_late_fence(&self) -> bool {
        self.flag() == OVCFlag::LateFence
    }

    pub fn duplicate_value() -> Self {
        let packed = (OVCFlag::DuplicateValue.as_u8() as u32) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_duplicate_value(&self) -> bool {
        self.flag() == OVCFlag::DuplicateValue
    }

    pub fn normal_value(value: &[u8], offset: usize) -> Self {
        let packed = Self::new(value, offset, OVCFlag::NormalValue);
        packed
    }

    pub fn is_normal_value(&self) -> bool {
        self.flag() == OVCFlag::NormalValue
    }

    pub fn initial_value() -> Self {
        let packed = (OVCFlag::InitialValue.as_u8() as u32) << Self::FLAG_SHIFT;
        Self(packed)
    }

    pub fn is_initial_value(&self) -> bool {
        self.flag() == OVCFlag::InitialValue
    }

    pub fn value(&self) -> [u8; 2] {
        let val = self.0.to_be_bytes();
        val[2..4].try_into().unwrap()
    }

    pub fn value_as_u32(&self) -> u32 {
        self.0 & Self::VALUE_MASK
    }

    pub fn arity_minus_offset(&self) -> u16 {
        ((self.0 >> Self::ARITY_SHIFT) & Self::ARITY_MASK) as u16
    }

    pub fn offset(&self) -> usize {
        Self::ARITY_MAX as usize - self.arity_minus_offset() as usize
    }

    pub fn flag(&self) -> OVCFlag {
        let flag_bits = ((self.0 >> Self::FLAG_SHIFT) & Self::FLAG_MASK) as u8;
        OVCFlag::from_u8(flag_bits).unwrap_or(OVCFlag::EarlyFence)
    }

    pub fn to_le_bytes(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    pub fn from_le_bytes(bytes: [u8; 4]) -> Self {
        let packed = u32::from_le_bytes(bytes);
        Self(packed)
    }

    pub fn to_be_bytes(&self) -> [u8; 4] {
        self.0.to_be_bytes()
    }

    pub fn from_be_bytes(bytes: [u8; 4]) -> Self {
        let packed = u32::from_be_bytes(bytes);
        Self(packed)
    }
}

impl std::fmt::Debug for OVCU32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.value();
        let offset = self.offset();
        let flag = self.flag();
        let flag_str = match flag {
            OVCFlag::EarlyFence => "EarlyFence",
            OVCFlag::DuplicateValue => "DuplicateValue",
            OVCFlag::NormalValue => "NormalValue",
            OVCFlag::InitialValue => "InitialValue",
            OVCFlag::LateFence => "LateFence",
        };

        write!(
            f,
            "OVC {{ value: {:?}, offset: {}, flag: {} }}",
            value, offset, flag_str
        )
    }
}

impl std::fmt::Display for OVCU32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.value();
        let value_as_u32 = self.value_as_u32();
        let offset = self.offset();
        let flag = self.flag();

        let flag_str = match flag {
            OVCFlag::EarlyFence => "[E]".to_string(),
            OVCFlag::DuplicateValue => format!("[D @ {}]", offset),
            OVCFlag::NormalValue => format!("[V:{:?}({}) @ {}]", value, value_as_u32, offset),
            OVCFlag::LateFence => "[L]".to_string(),
            OVCFlag::InitialValue => "[I]".to_string(),
        };

        write!(f, "{}", flag_str)
    }
}

pub trait OVC32Trait: SentinelValue {
    fn ovc(&self) -> &OVCU32;
    fn ovc_mut(&mut self) -> &mut OVCU32;
    fn key(&self) -> &[u8];

    fn derive_ovc_from(&mut self, prev: &Self) -> bool {
        let min_len = self.key().len().min(prev.key().len());
        // Hybrid approach: byte-by-byte comparison with chunk-aligned OVC creation
        for i in 0..min_len {
            match self.key()[i].cmp(&prev.key()[i]) {
                Ordering::Less => {
                    return false;
                }
                Ordering::Greater => {
                    // Found difference at byte i, align OVC to chunk boundary
                    let aligned_offset = (i / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                    let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(self.key().len());
                    *self.ovc_mut() = OVCU32::normal_value(
                        &self.key()[aligned_offset..chunk_end],
                        aligned_offset,
                    );
                    return true;
                }
                Ordering::Equal => continue,
            }
        }

        // If we reach here, prev is a prefix of self or they're equal
        match self.key().len().cmp(&prev.key().len()) {
            Ordering::Greater => {
                // If self is longer, align the offset to chunk boundary
                let aligned_offset = (prev.key().len() / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(self.key().len());
                *self.ovc_mut() =
                    OVCU32::normal_value(&self.key()[aligned_offset..chunk_end], aligned_offset);
            }
            Ordering::Equal => {
                *self.ovc_mut() = OVCU32::duplicate_value();
            }
            Ordering::Less => {
                return false;
            }
        }
        true
    }

    fn update(&mut self, prev: &Self) {
        let _ = self.derive_ovc_from(prev);
    }

    fn max_ovc(&mut self, other: &Self) {
        if *self.ovc_mut() < *other.ovc() {
            *self.ovc_mut() = *other.ovc();
        }
    }

    fn compare_and_update(&mut self, other: &mut Self) -> Ordering {
        let val_a = self.key();
        let val_b = other.key();
        let ovc_a = self.ovc();
        let ovc_b = other.ovc();

        match ovc_a.cmp(&ovc_b) {
            Ordering::Equal => {
                if ovc_a.is_early_fence() || ovc_a.is_late_fence() || ovc_a.is_duplicate_value() {
                    return Ordering::Equal;
                }

                let offset = ovc_a.offset();
                let start_index =
                    (!ovc_a.is_initial_value() as usize) * (offset + OVC32_CHUNK_SIZE);

                let min_len = val_a.len().min(val_b.len());

                for i in start_index..min_len {
                    match val_a[i].cmp(&val_b[i]) {
                        Ordering::Equal => continue,
                        Ordering::Less => {
                            // Found difference at byte i, align OVC to chunk boundary
                            let aligned_offset = (i / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                            let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_b.len());
                            *other.ovc_mut() = OVCU32::normal_value(
                                &val_b[aligned_offset..chunk_end],
                                aligned_offset,
                            );
                            return Ordering::Less;
                        }
                        Ordering::Greater => {
                            // Found difference at byte i, align OVC to chunk boundary
                            let aligned_offset = (i / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                            let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_a.len());
                            *self.ovc_mut() = OVCU32::normal_value(
                                &val_a[aligned_offset..chunk_end],
                                aligned_offset,
                            );
                            return Ordering::Greater;
                        }
                    }
                }

                // If we reach here, the prefixes are equal
                match val_a.len().cmp(&val_b.len()) {
                    Ordering::Greater => {
                        // This is longer than the other
                        // Align to chunk boundary where the difference starts
                        let aligned_offset = (val_b.len() / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                        let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_a.len());
                        *self.ovc_mut() =
                            OVCU32::normal_value(&val_a[aligned_offset..chunk_end], aligned_offset);
                        Ordering::Greater
                    }
                    Ordering::Less => {
                        // The other is longer than this
                        // Align to chunk boundary where the difference starts
                        let aligned_offset = (val_a.len() / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                        let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_b.len());
                        *other.ovc_mut() =
                            OVCU32::normal_value(&val_b[aligned_offset..chunk_end], aligned_offset);
                        Ordering::Less
                    }
                    Ordering::Equal => {
                        *self.ovc_mut() = OVCU32::duplicate_value();
                        Ordering::Equal
                    }
                }
            }
            ord => ord,
        }
    }

    fn compare_and_update_with_mode(&mut self, other: &mut Self, full_comp: bool) -> Ordering {
        if full_comp {
            // Handle fences: early_fence < everything < late_fence
            if self.is_early_fence() {
                return Ordering::Less; // Self is smallest
            }
            if self.is_late_fence() {
                return Ordering::Greater; // Self is largest
            }
            if other.is_early_fence() {
                return Ordering::Greater; // Other is smallest, so self is larger
            }
            if other.is_late_fence() {
                return Ordering::Less; // Other is largest, so self is smaller
            }
            let val_a = self.key();
            let val_b = other.key();

            let start_index = 0;
            let min_len = val_a.len().min(val_b.len());

            for i in start_index..min_len {
                match val_a[i].cmp(&val_b[i]) {
                    Ordering::Equal => continue,
                    Ordering::Less => {
                        let aligned_offset = (i / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                        let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_b.len());
                        *other.ovc_mut() =
                            OVCU32::normal_value(&val_b[aligned_offset..chunk_end], aligned_offset);
                        return Ordering::Less;
                    }
                    Ordering::Greater => {
                        let aligned_offset = (i / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                        let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_a.len());
                        *self.ovc_mut() =
                            OVCU32::normal_value(&val_a[aligned_offset..chunk_end], aligned_offset);
                        return Ordering::Greater;
                    }
                }
            }

            match val_a.len().cmp(&val_b.len()) {
                Ordering::Greater => {
                    let aligned_offset = (val_b.len() / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                    let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_a.len());
                    *self.ovc_mut() =
                        OVCU32::normal_value(&val_a[aligned_offset..chunk_end], aligned_offset);
                    Ordering::Greater
                }
                Ordering::Less => {
                    let aligned_offset = (val_a.len() / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                    let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(val_b.len());
                    *other.ovc_mut() =
                        OVCU32::normal_value(&val_b[aligned_offset..chunk_end], aligned_offset);
                    Ordering::Less
                }
                Ordering::Equal => {
                    *self.ovc_mut() = OVCU32::duplicate_value();
                    Ordering::Equal
                }
            }
        } else {
            self.compare_and_update(other)
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
// Do not implement Ord or PartialOrd for OVCEntry
// Use `compare_and_update` method for ordering
pub struct OVCEntry32 {
    ovc: OVCU32,
    val: Vec<u8>,
}

impl std::fmt::Debug for OVCEntry32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(f, "{} -> {:?}", self.ovc, self.val)
        }
    }
}

impl std::fmt::Display for OVCEntry32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(f, "{} -> {:?}", self.ovc, self.val)
        }
    }
}

impl From<Vec<u8>> for OVCEntry32 {
    fn from(val: Vec<u8>) -> Self {
        let ovc = OVCU32::initial_value();
        Self { ovc, val }
    }
}

impl OVCEntry32 {
    pub fn new(val: Vec<u8>) -> Self {
        let ovc = OVCU32::initial_value();
        Self { ovc, val }
    }

    pub fn get_ovc(&self) -> OVCU32 {
        self.ovc
    }

    pub fn get_key(&self) -> &[u8] {
        &self.val
    }

    pub fn get_data(&self) -> &[u8] {
        &self.val
    }
}

impl OVC32Trait for OVCEntry32 {
    fn ovc(&self) -> &OVCU32 {
        &self.ovc
    }

    fn ovc_mut(&mut self) -> &mut OVCU32 {
        &mut self.ovc
    }

    fn key(&self) -> &[u8] {
        &self.val
    }
}

impl SentinelValue for OVCEntry32 {
    fn early_fence() -> Self {
        Self {
            ovc: OVCU32::early_fence(),
            val: Vec::new(),
        }
    }

    fn late_fence() -> Self {
        Self {
            ovc: OVCU32::late_fence(),
            val: Vec::new(),
        }
    }

    fn is_early_fence(&self) -> bool {
        self.ovc.is_early_fence()
    }

    fn is_late_fence(&self) -> bool {
        self.ovc.is_late_fence()
    }
}

impl RecordSize for OVCEntry32 {
    fn size(&self) -> usize {
        mem::size_of::<Self>() + self.val.len()
    }
}

#[derive(Clone, PartialEq, Eq)]
// Do not implement Ord or PartialOrd for OVCKeyValue32
// Use `compare_and_update` method for ordering
pub struct OVCKeyValue32 {
    ovc: OVCU32,
    key: Vec<u8>,
    value: Vec<u8>,
}

impl OVCKeyValue32 {
    pub fn new(key: Vec<u8>, value: Vec<u8>) -> Self {
        let ovc = OVCU32::initial_value();
        Self { ovc, key, value }
    }

    pub fn new_with_ovc(ovc: OVCU32, key: Vec<u8>, value: Vec<u8>) -> Self {
        Self { ovc, key, value }
    }

    pub fn value(&self) -> &[u8] {
        &self.value
    }

    pub fn take(self) -> (OVCU32, Vec<u8>, Vec<u8>) {
        (self.ovc, self.key, self.value)
    }
}

impl OVC32Trait for OVCKeyValue32 {
    fn ovc(&self) -> &OVCU32 {
        &self.ovc
    }

    fn ovc_mut(&mut self) -> &mut OVCU32 {
        &mut self.ovc
    }

    fn key(&self) -> &[u8] {
        &self.key
    }
}

impl SentinelValue for OVCKeyValue32 {
    fn early_fence() -> Self {
        Self {
            ovc: OVCU32::early_fence(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    fn late_fence() -> Self {
        Self {
            ovc: OVCU32::late_fence(),
            key: Vec::new(),
            value: Vec::new(),
        }
    }

    fn is_early_fence(&self) -> bool {
        self.ovc.is_early_fence()
    }

    fn is_late_fence(&self) -> bool {
        self.ovc.is_late_fence()
    }
}

impl std::fmt::Debug for OVCKeyValue32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(f, "{} -> {:?}", self.ovc, self.key)
        }
    }
}

impl std::fmt::Display for OVCKeyValue32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(f, "{} -> {:?}", self.ovc, self.key)
        }
    }
}

impl From<(Vec<u8>, Vec<u8>)> for OVCKeyValue32 {
    fn from((key, value): (Vec<u8>, Vec<u8>)) -> Self {
        let ovc = OVCU32::initial_value();
        Self { ovc, key, value }
    }
}

impl From<(OVCU32, Vec<u8>, Vec<u8>)> for OVCKeyValue32 {
    fn from((ovc, key, value): (OVCU32, Vec<u8>, Vec<u8>)) -> Self {
        Self { ovc, key, value }
    }
}

impl RecordSize for OVCKeyValue32 {
    fn size(&self) -> usize {
        mem::size_of::<Self>() + self.key.len() + self.value.len()
    }
}

/// Key-only tree entry for zero-copy merge.
/// Stores the key and value_len (but NOT value bytes).
/// Values flow directly from source reader to output writer.
#[derive(Clone, PartialEq, Eq)]
pub struct OVCKey32 {
    ovc: OVCU32,
    key: Vec<u8>,
    pub value_len: usize,
}

impl OVCKey32 {
    pub fn new(ovc: OVCU32, key: Vec<u8>, value_len: usize) -> Self {
        Self {
            ovc,
            key,
            value_len,
        }
    }

    /// Consume self and return the key buffer for reuse.
    pub fn take_key(self) -> Vec<u8> {
        self.key
    }
}

impl OVC32Trait for OVCKey32 {
    fn ovc(&self) -> &OVCU32 {
        &self.ovc
    }

    fn ovc_mut(&mut self) -> &mut OVCU32 {
        &mut self.ovc
    }

    fn key(&self) -> &[u8] {
        &self.key
    }
}

impl SentinelValue for OVCKey32 {
    fn early_fence() -> Self {
        Self {
            ovc: OVCU32::early_fence(),
            key: Vec::new(),
            value_len: 0,
        }
    }

    fn late_fence() -> Self {
        Self {
            ovc: OVCU32::late_fence(),
            key: Vec::new(),
            value_len: 0,
        }
    }

    fn is_early_fence(&self) -> bool {
        self.ovc.is_early_fence()
    }

    fn is_late_fence(&self) -> bool {
        self.ovc.is_late_fence()
    }
}

impl std::fmt::Debug for OVCKey32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(
                f,
                "{} -> {:?} (vlen={})",
                self.ovc, self.key, self.value_len
            )
        }
    }
}

impl std::fmt::Display for OVCKey32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_early_fence() {
            write!(f, "[EarlyFence]")
        } else if self.is_late_fence() {
            write!(f, "[LateFence]")
        } else {
            write!(
                f,
                "{} -> {:?} (vlen={})",
                self.ovc, self.key, self.value_len
            )
        }
    }
}

#[allow(dead_code)]
pub(crate) fn compute_ovc_delta(prev_key: Option<&[u8]>, key: &[u8]) -> OVCU32 {
    let Some(prev) = prev_key else {
        return OVCU32::initial_value();
    };

    let min_len = prev.len().min(key.len());
    for i in 0..min_len {
        match key[i].cmp(&prev[i]) {
            Ordering::Less => {
                panic!("Input must be non-decreasing for OVC encoding");
            }
            Ordering::Greater => {
                let aligned_offset = (i / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
                let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(key.len());
                return OVCU32::normal_value(&key[aligned_offset..chunk_end], aligned_offset);
            }
            Ordering::Equal => continue,
        }
    }

    match key.len().cmp(&prev.len()) {
        Ordering::Greater => {
            let aligned_offset = (prev.len() / OVC32_CHUNK_SIZE) * OVC32_CHUNK_SIZE;
            let chunk_end = (aligned_offset + OVC32_CHUNK_SIZE).min(key.len());
            OVCU32::normal_value(&key[aligned_offset..chunk_end], aligned_offset)
        }
        Ordering::Equal => OVCU32::duplicate_value(),
        Ordering::Less => {
            panic!("Input must be non-decreasing for OVC encoding");
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::ovc::utils::random_string;

    use super::*;

    #[test]
    fn test_simple_ovc() {
        let base = vec![1, 2, 3, 4];
        let duplicate = vec![1, 2, 3, 4];
        let longer = vec![1, 2, 3, 4, 0];

        let mut ovc_base = OVCEntry32::new(base);
        let mut ovc_duplicate = OVCEntry32::new(duplicate);
        let mut ovc_longer = OVCEntry32::new(longer);

        assert_eq!(
            ovc_duplicate.compare_and_update(&mut ovc_base),
            Ordering::Equal
        );
        assert_eq!(
            ovc_longer.compare_and_update(&mut ovc_base),
            Ordering::Greater
        );

        println!("Base OVC: {:?}", ovc_base);
        println!(
            "Duplicate OVC: {:?}, code: {:?}",
            ovc_duplicate,
            ovc_duplicate.get_ovc()
        );
        println!("Duplicate 4: {:?}", OVCU32::duplicate_value());
        println!("Longer OVC: {:?}", ovc_longer);
        assert!(ovc_duplicate.get_ovc() < ovc_longer.get_ovc());
        assert_eq!(ovc_duplicate.get_ovc(), OVCU32::duplicate_value());
    }

    #[test]
    fn test_run_encoding() {
        let run = vec![
            vec![1, 2, 3],
            vec![1, 2, 3],
            vec![1, 2, 3, 0],
            vec![1, 2, 3, 0, 0],
        ];

        let ovc_run = encode_run_with_ovc32(&run);
        assert_eq!(ovc_run.len(), 4);
        assert!(ovc_run[0].get_ovc().is_initial_value());
        assert!(ovc_run[1].get_ovc().is_duplicate_value());
        assert!(ovc_run[2].get_ovc().offset() == 2 && ovc_run[2].get_ovc().value() == [3, 0]);
        assert!(ovc_run[3].get_ovc().offset() == 4 && ovc_run[3].get_ovc().value() == [0, 0]);

        println!("OvC Run");
        for entry in &ovc_run {
            println!("{:?}", entry);
        }
    }

    #[test]
    fn test_update() {
        let base = vec![9, 1, 3];
        let this = vec![9, 1, 4]; // Different from the copy at index 2 with value 4
        let ovc_base = OVCEntry32::new(base);
        let mut ovc = OVCEntry32::new(this);

        ovc.update(&ovc_base);
        assert!(ovc.get_ovc().offset() == 2 && ovc.get_ovc().value() == [4, 0]);
    }

    #[test]
    fn test_order_different_ovcs() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry32::new(base);

        // [NOTE] Value with sort order smaller than base does not work!!!
        // let val1 = vec![1, 1, 1, 1, 1];
        // let mut ovc1 = OVCEntry::new(val1);
        // ovc1.update(&ovc_base);

        let val2 = vec![1, 2, 3, 4, 5];
        let mut ovc2 = OVCEntry32::new(val2);
        ovc2.update(&ovc_base);

        let val3 = vec![1, 2, 3, 5, 5];
        let mut ovc3 = OVCEntry32::new(val3);
        ovc3.update(&ovc_base);

        let val4 = vec![1, 2, 7, 5, 5];
        let mut ovc4 = OVCEntry32::new(val4);
        ovc4.update(&ovc_base);

        let val5 = vec![1, 3, 5, 5, 5];
        let mut ovc5 = OVCEntry32::new(val5);
        ovc5.update(&ovc_base);

        let val6 = vec![2, 5, 5, 5, 5];
        let mut ovc6 = OVCEntry32::new(val6);
        ovc6.update(&ovc_base);

        let mut vec = vec![
            ovc6.clone(),
            ovc5.clone(),
            ovc4.clone(),
            ovc3.clone(),
            ovc2.clone(),
        ];
        vec.sort_by_key(|entry| entry.get_ovc());

        println!("Sorted OVCs:");
        for entry in &vec {
            println!("{:?}, code: {:?}", entry, entry.get_ovc());
        }

        let expected = vec![ovc2, ovc3, ovc4, ovc5, ovc6];

        assert_eq!(vec, expected);
    }

    #[test]
    fn test_order_same_ovcs_different_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry32::new(base);

        let val1 = vec![1, 2, 3, 6, 1];
        let mut ovc1 = OVCEntry32::new(val1);
        ovc1.update(&ovc_base);

        let val2 = vec![1, 2, 3, 6, 2];
        let mut ovc2 = OVCEntry32::new(val2);
        ovc2.update(&ovc_base);

        let val3 = vec![1, 2, 3, 6, 3];
        let mut ovc3 = OVCEntry32::new(val3);
        ovc3.update(&ovc_base);

        let val4 = vec![1, 2, 3, 6, 4];
        let mut ovc4 = OVCEntry32::new(val4);
        ovc4.update(&ovc_base);

        let val5 = vec![1, 2, 3, 6, 5];
        let mut ovc5 = OVCEntry32::new(val5);
        ovc5.update(&ovc_base);

        let mut vec = vec![
            ovc5.clone(),
            ovc4.clone(),
            ovc3.clone(),
            ovc2.clone(),
            ovc1.clone(),
        ];

        assert!(vec.iter().all(|ovc| {
            let ovc = ovc.get_ovc();
            ovc.offset() == 2 && ovc.value() == [3, 6]
        }));

        vec.sort_by(|a, b| {
            let a_ovc = a.get_ovc();
            let b_ovc = b.get_ovc();
            match a_ovc.cmp(&b_ovc) {
                Ordering::Equal => {
                    // If the OVCs are equal, compare the values
                    let a_val = a.get_data();
                    let b_val = b.get_data();
                    a_val.cmp(b_val)
                }
                ord => ord,
            }
        });

        let expected = vec![ovc1, ovc2, ovc3, ovc4, ovc5];

        assert_eq!(vec, expected);
    }

    #[test]
    fn test_order_same_ovcs_same_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry32::new(base);

        let val1 = vec![1, 2, 3, 4, 6];
        let mut ovc1 = OVCEntry32::new(val1);
        ovc1.update(&ovc_base);

        let val2 = vec![1, 2, 3, 4, 6];
        let mut ovc2 = OVCEntry32::new(val2);
        ovc2.update(&ovc_base);

        assert_eq!(ovc1, ovc2);
    }

    #[test]
    fn test_order_different_lengths() {
        let base = vec![1, 2, 3];
        let ovc_base = OVCEntry32::new(base);

        let test_vals = [
            vec![1, 2, 3],
            vec![1, 2, 3, 0],
            vec![1, 2, 3, 0, 0],
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 3, 4, 5, 6],
            vec![1, 2, 3, 4, 5, 6, 0],
        ];

        let mut ovcs: Vec<OVCEntry32> = test_vals
            .iter()
            .map(|val| {
                let mut ovc = OVCEntry32::new(val.clone());
                ovc.update(&ovc_base);
                ovc
            })
            .collect();

        ovcs.sort_by(|a, b| {
            let a_ovc = a.get_ovc();
            let b_ovc = b.get_ovc();
            match a_ovc.cmp(&b_ovc) {
                Ordering::Equal => {
                    // If the OVCs are equal, compare the values
                    let a_val = a.get_data();
                    let b_val = b.get_data();
                    a_val.cmp(b_val)
                }
                ord => ord,
            }
        });

        println!("Sorted OVCs:");
        for entry in &ovcs {
            println!("{:?}", entry);
        }

        // Check that the OVCs are sorted correctly
        for i in 0..test_vals.len() {
            assert_eq!(ovcs[i].get_data(), test_vals[i]);
        }
    }

    #[test]
    fn test_ovc_with_normalize() {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();

        let run_length = 10000;
        let original_run = (0..run_length)
            .map(|_| random_string(rng.random_range(1..=100)))
            .collect::<Vec<_>>();

        // Normalize the run. This is sorted in the same order as the original run.
        let mut normalized_run = original_run
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect::<Vec<_>>();

        // Make sure that all elements are unique
        normalized_run.sort();

        // Create the offset value codings with base as the smallest element
        let ovc_base = OVCEntry32::new(normalized_run[0].clone());
        let mut ovcs = normalized_run
            .iter()
            .map(|s| OVCEntry32::new(s.clone()))
            .collect::<Vec<_>>();
        ovcs.iter_mut().for_each(|ovc| ovc.update(&ovc_base));

        // Randomize the order of the offset value codings
        ovcs.shuffle(&mut rng);

        // Sort the offset value codings
        ovcs.sort_by(|a, b| {
            let a_ovc = a.get_ovc();
            let b_ovc = b.get_ovc();
            match a_ovc.cmp(&b_ovc) {
                Ordering::Equal => {
                    // If the OVCs are equal, compare the values
                    let a_val = a.get_data();
                    let b_val = b.get_data();
                    a_val.cmp(b_val)
                }
                ord => ord,
            }
        });

        // Print the sorted OVCs
        // println!("Sorted OVCs:");
        // for entry in &ovcs {
        //     println!("{:?}", entry);
        // }

        // Check that the order of the offset value codings is the same as the normalized run
        for (ovc, expected) in ovcs.iter().zip(normalized_run.iter()) {
            // Dereference the pointer to get the value
            assert_eq!(ovc.get_data(), expected);
        }
    }

    #[test]
    fn test_compare_and_update_different_ovcs() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry32::new(base);

        // Different OVCs relative to base, different keys
        let val1 = vec![1, 2, 3, 6, 1];
        let mut ovc1 = OVCEntry32::new(val1);
        ovc1.update(&ovc_base);
        let val2 = vec![1, 2, 4, 6, 2];
        let mut ovc2 = OVCEntry32::new(val2);
        ovc2.update(&ovc_base);

        assert!(ovc1.get_ovc().offset() == 2 && ovc1.get_ovc().value() == [3, 6]);
        assert!(ovc2.get_ovc().offset() == 2 && ovc2.get_ovc().value() == [4, 6]);

        let res = OVCEntry32::compare_and_update(&mut ovc1, &mut ovc2);
        assert_eq!(res, Ordering::Less);
        assert!(ovc1.get_ovc().offset() == 2 && ovc1.get_ovc().value() == [3, 6]);
        assert!(ovc2.get_ovc().offset() == 2 && ovc2.get_ovc().value() == [4, 6]);

        let res = OVCEntry32::compare_and_update(&mut ovc2, &mut ovc1);
        assert_eq!(res, Ordering::Greater);
        assert!(ovc1.get_ovc().offset() == 2 && ovc1.get_ovc().value() == [3, 6]);
        assert!(ovc2.get_ovc().offset() == 2 && ovc2.get_ovc().value() == [4, 6]);
    }

    #[test]
    fn test_compare_and_update_same_ovcs_same_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry32::new(base);

        // Same OVCs relative to base, same keys
        let val1 = vec![1, 2, 3, 4, 6];
        let mut ovc1 = OVCEntry32::new(val1);
        ovc1.update(&ovc_base);
        let val2 = vec![1, 2, 3, 4, 6];
        let mut ovc2 = OVCEntry32::new(val2);
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 4 && ovc.value() == [6, 0]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 4 && ovc.value() == [6, 0]);

        let res = ovc1.compare_and_update(&mut ovc2);
        assert!(res == Ordering::Equal);

        // The OVCs should remain the same
        let ovc = ovc1.get_ovc();
        assert!(ovc.is_duplicate_value());
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 4 && ovc.value() == [6, 0]);
    }

    #[test]
    fn test_compare_and_update_same_ovcs_different_keys() {
        let base = vec![1, 2, 3, 4, 5];
        let ovc_base = OVCEntry32::new(base);

        // Same OVCs relative to base but different values
        let val1 = vec![1, 2, 3, 6, 1];
        let mut ovc1 = OVCEntry32::new(val1);
        ovc1.update(&ovc_base);
        let val2 = vec![1, 2, 3, 6, 2];
        let mut ovc2 = OVCEntry32::new(val2);
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 2 && ovc.value() == [3, 6]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 2 && ovc.value() == [3, 6]);

        let res = ovc1.compare_and_update(&mut ovc2);
        assert!(res == Ordering::Less);
        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 2 && ovc.value() == [3, 6]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 4 && ovc.value() == [2, 0]);

        // ovc1.reset();
        ovc1.update(&ovc_base);
        // ovc2.reset();
        ovc2.update(&ovc_base);

        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 2 && ovc.value() == [3, 6]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 2 && ovc.value() == [3, 6]);

        let res = OVCEntry32::compare_and_update(&mut ovc2, &mut ovc1);
        assert!(res == Ordering::Greater);
        let ovc = ovc1.get_ovc();
        assert!(ovc.offset() == 2 && ovc.value() == [3, 6]);
        let ovc = ovc2.get_ovc();
        assert!(ovc.offset() == 4 && ovc.value() == [2, 0]);
    }
}
