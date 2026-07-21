//! zlib-compatible CRC32 (IEEE 802.3, reflected, poly 0xEDB88320, init/xorout
//! 0xFFFFFFFF) -- matches Python's `zlib.crc32()` and `c/mode15_reader.c`'s
//! `m15_crc32()` bit-for-bit. Bit-shift table-free implementation (no
//! lazily-initialized static, no unsafe) so it's trivially safe to call from
//! many fuzz iterations without any shared mutable state.
//!
//! Ported by hand from mode15_reader.c's m15_crc32_update() -- same
//! reflected-poly bit loop, just table-free (8 shifts/byte instead of a
//! precomputed 256-entry table): this crate's fuzz campaigns hash at most a
//! few KB per iteration, so the table's setup cost this file's C sibling
//! explicitly avoids paying (see mode15_reader.c's own comment on why it
//! recomputes its table per-call rather than caching it) isn't worth
//! re-deriving here either; a plain bit loop is simpler and just as correct.

/// CRC32 of `data`, matching `zlib.crc32(data)` and `c/mode15_reader.c`'s
/// `m15_crc32()`. Pure, panics never (no indexing, no arithmetic that can
/// overflow: everything here is bounded 32-bit shifts/xors).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            // branchless: mask is all-ones if the low bit was set, else 0
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Streaming variant: same algorithm, split init/update/final so callers can
/// hash disjoint ranges (e.g. mode15's "reconstructed index bytes" followed
/// by the borrowed payload) without concatenating them into one buffer --
/// mirrors mode15_reader.c's own m15_crc32_init_state/_update/_final split.
#[derive(Clone, Copy, Debug)]
pub struct Crc32State(u32);

impl Default for Crc32State {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32State {
    pub fn new() -> Self {
        Crc32State(0xFFFF_FFFF)
    }
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.0;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                let mask = 0u32.wrapping_sub(crc & 1);
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        self.0 = crc;
    }
    pub fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard CRC32 "check value" -- also literally asserted in
    /// mode15_reader.h's own doc comment ("CRC32(\"123456789\") ==
    /// 0xCBF43926, the standard check value, verified in
    /// tests/test_mode15_reader.c"). If this doesn't match, nothing else in
    /// this crate's Mode-1.5 verification can be trusted.
    #[test]
    fn known_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn empty_input() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn streaming_matches_oneshot() {
        let data = b"the quick brown fox jumps over the lazy dog, 0123456789";
        let one_shot = crc32(data);
        let (a, b) = data.split_at(17);
        let mut st = Crc32State::new();
        st.update(a);
        st.update(b);
        assert_eq!(st.finish(), one_shot);
    }
}
