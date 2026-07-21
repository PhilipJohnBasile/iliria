//! Minimal, panic-free, std-only JSON parser -- just enough of RFC 8259 to
//! parse a safetensors header (`c/st.h`'s `json_parse`/`json_get` role, for
//! this Rust side; `c/json.h` itself is out of scope -- this crate was not
//! asked to audit it and does not claim to replicate its internals bit for
//! bit). Written by hand instead of pulling in `serde_json` because this
//! crate has a hard std-only/offline-build constraint.
//!
//! PANIC-SAFETY DESIGN (this is the one property the whole exercise leans
//! on, so it gets its own paragraph): every function here returns
//! `Result`/`Option`, never indexes a slice/string without a preceding
//! length check, and never performs arithmetic that can overflow on
//! attacker-sized input --
//!   - the parser scans the input as raw `&[u8]` (never slices a `&str` by
//!     byte offset, which is exactly the pattern that panics on a
//!     non-char-boundary index in real Rust code that mixes byte positions
//!     with string slicing);
//!   - string content is *built up* into an owned `Vec<u8>` / `String` via
//!     push operations and a single final `String::from_utf8` (which
//!     returns `Result`, never panics) rather than sliced out of the
//!     source;
//!   - nesting depth is capped (`MAX_DEPTH`) so adversarial input like
//!     `"[[[[[[...".repeat(1_000_000)` cannot recurse the parser into a
//!     stack overflow -- note a stack overflow is a hard process abort in
//!     Rust, NOT a catchable panic, so `std::panic::catch_unwind` in the
//!     fuzz harness would NOT save us here; this cap is the actual mitigation.

pub const MAX_DEPTH: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    /// Every JSON number (safetensors only ever needs non-negative integers
    /// out of this, but we parse the full grammar so a fuzzer mutating a
    /// number into "-1.5e10" is rejected by *our* semantic layer, not by
    /// this parser silently mangling it).
    Num(f64),
    Str(String),
    Arr(Vec<JsonValue>),
    /// Order-preserving (Vec of pairs, not a HashMap): safetensors headers
    /// are small (a few dozen keys at most per tensor entry) and duplicate
    /// detection / deterministic iteration order both matter more here than
    /// O(1) lookup.
    Obj(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }
    pub fn as_obj(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Obj(o) => Some(o.as_slice()),
            _ => None,
        }
    }
    pub fn obj_get(&self, key: &str) -> Option<&JsonValue> {
        self.as_obj()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    /// Exact non-negative integer (matches safetensors `shape`/`data_offsets`
    /// semantics: a `3.5` or a negative number here is a MALFORMED header,
    /// not silently truncated -- see quant_format.rs / safetensors.rs for
    /// where the C-equivalent LENIENT mode instead truncates, mirroring a C
    /// `(int64_t)double_value` cast).
    pub fn as_u64_exact(&self) -> Option<u64> {
        match self {
            JsonValue::Num(n) => {
                if !n.is_finite() || *n < 0.0 || n.fract() != 0.0 {
                    return None;
                }
                if *n > u64::MAX as f64 {
                    return None;
                }
                Some(*n as u64) // f64->u64 `as` is saturating/defined, never panics
            }
            _ => None,
        }
    }
    /// Lenient companion to as_u64_exact: truncates toward zero and clamps
    /// into i64 range, mirroring what a C `(int64_t)jval->num` cast actually
    /// does to a JSON number with no validation at all (this is used ONLY
    /// by the "C-equivalent lenient" mimic in safetensors.rs, never by the
    /// strict validator).
    pub fn as_i64_c_cast(&self) -> Option<i64> {
        match self {
            JsonValue::Num(n) => {
                if n.is_nan() {
                    return Some(0); // (int64_t)NAN is UB in C; we just need *a* deterministic value
                }
                let t = n.trunc();
                if t <= i64::MIN as f64 {
                    Some(i64::MIN)
                } else if t >= i64::MAX as f64 {
                    Some(i64::MAX)
                } else {
                    Some(t as i64)
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub pos: usize,
    pub msg: &'static str,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cursor { b, pos: 0, depth: 0 }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }
    fn err(&self, msg: &'static str) -> JsonError {
        JsonError { pos: self.pos, msg }
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn expect(&mut self, want: u8) -> Result<(), JsonError> {
        match self.bump() {
            Some(c) if c == want => Ok(()),
            _ => Err(self.err("unexpected byte")),
        }
    }
    fn literal(&mut self, lit: &'static [u8]) -> Result<(), JsonError> {
        if self.pos + lit.len() > self.b.len() {
            return Err(self.err("truncated literal"));
        }
        if &self.b[self.pos..self.pos + lit.len()] == lit {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(self.err("unknown literal"))
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, JsonError> {
        self.skip_ws();
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            // MUST check before recursing further -- see module doc: this is
            // the actual stack-overflow mitigation, not decorative.
            return Err(self.err("max nesting depth exceeded"));
        }
        let v = match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonValue::Str),
            Some(b't') => self.literal(b"true").map(|_| JsonValue::Bool(true)),
            Some(b'f') => self.literal(b"false").map(|_| JsonValue::Bool(false)),
            Some(b'n') => self.literal(b"null").map(|_| JsonValue::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            _ => Err(self.err("expected a value")),
        };
        self.depth -= 1;
        v
    }

    fn parse_object(&mut self) -> Result<JsonValue, JsonError> {
        self.expect(b'{')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Obj(out));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let val = self.parse_value()?;
            out.push((key, val));
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        Ok(JsonValue::Obj(out))
    }

    fn parse_array(&mut self) -> Result<JsonValue, JsonError> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Arr(out));
        }
        loop {
            let val = self.parse_value()?;
            out.push(val);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        Ok(JsonValue::Arr(out))
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut bytes: Vec<u8> = Vec::new();
        loop {
            let c = self.bump().ok_or_else(|| self.err("unterminated string"))?;
            match c {
                b'"' => break,
                b'\\' => {
                    let esc = self.bump().ok_or_else(|| self.err("truncated escape"))?;
                    match esc {
                        b'"' => bytes.push(b'"'),
                        b'\\' => bytes.push(b'\\'),
                        b'/' => bytes.push(b'/'),
                        b'b' => bytes.push(0x08),
                        b'f' => bytes.push(0x0C),
                        b'n' => bytes.push(b'\n'),
                        b'r' => bytes.push(b'\r'),
                        b't' => bytes.push(b'\t'),
                        b'u' => {
                            let cp = self.parse_hex4()?;
                            // Minimal surrogate-pair handling; anything that
                            // doesn't form a valid scalar value is rejected
                            // rather than guessed at.
                            let scalar = if (0xD800..=0xDBFF).contains(&cp) {
                                self.expect(b'\\')?;
                                self.expect(b'u')?;
                                let lo = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                0x10000u32
                                    + ((cp as u32 - 0xD800) << 10)
                                    + (lo as u32 - 0xDC00)
                            } else {
                                cp as u32
                            };
                            let ch = char::from_u32(scalar)
                                .ok_or_else(|| self.err("invalid unicode escape"))?;
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                _ => bytes.push(c),
            }
        }
        String::from_utf8(bytes).map_err(|_| self.err("invalid utf-8 in string"))
    }

    fn parse_hex4(&mut self) -> Result<u16, JsonError> {
        if self.pos + 4 > self.b.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let mut v: u16 = 0;
        for _ in 0..4 {
            let c = self.bump().unwrap(); // safe: length checked above
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(self.err("invalid hex digit")),
            };
            v = v.wrapping_mul(16).wrapping_add(d as u16);
        }
        Ok(v)
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let mut saw_digit = false;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
            saw_digit = true;
        }
        if !saw_digit {
            return Err(self.err("invalid number"));
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let mut saw_frac = false;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
                saw_frac = true;
            }
            if !saw_frac {
                return Err(self.err("invalid number: empty fraction"));
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            let mut saw_exp = false;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
                saw_exp = true;
            }
            if !saw_exp {
                return Err(self.err("invalid number: empty exponent"));
            }
        }
        // `start..self.pos` only ever spans bytes we ourselves advanced over
        // one-at-a-time above, all ASCII -- always a valid UTF-8 str slice
        // and always in-bounds, so from_utf8/indexing here cannot panic.
        let text = std::str::from_utf8(&self.b[start..self.pos])
            .map_err(|_| self.err("unreachable: non-utf8 number"))?;
        text.parse::<f64>()
            .map(JsonValue::Num)
            .map_err(|_| self.err("number out of range"))
    }
}

/// Parse `bytes` as a single JSON value; trailing non-whitespace bytes after
/// the value are an error (safetensors headers are exactly one JSON object,
/// nothing after).
pub fn parse(bytes: &[u8]) -> Result<JsonValue, JsonError> {
    let mut cur = Cursor::new(bytes);
    let v = cur.parse_value()?;
    cur.skip_ws();
    if cur.pos != cur.b.len() {
        return Err(cur.err("trailing data after JSON value"));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_safetensors_shaped_object() {
        let src = br#"{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":{"x":"y"}}"#;
        let v = parse(src).expect("valid json");
        let obj = v.as_obj().expect("top-level object");
        assert_eq!(obj.len(), 2);
        let a = v.obj_get("a").unwrap();
        assert_eq!(a.obj_get("dtype").unwrap().as_str(), Some("F32"));
        let shape = a.obj_get("shape").unwrap().as_arr().unwrap();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0].as_u64_exact(), Some(2));
        assert_eq!(shape[1].as_u64_exact(), Some(3));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(parse(br#"{}garbage"#).is_err());
    }

    #[test]
    fn rejects_unterminated_string() {
        assert!(parse(br#"{"a":"b"#).is_err());
    }

    #[test]
    fn rejects_deep_nesting_without_stack_overflow() {
        let mut s = Vec::new();
        s.resize(200_000, b'[');
        s.extend(std::iter::repeat(b']').take(200_000));
        // Must return an Err (depth cap), not overflow the stack.
        assert!(parse(&s).is_err());
    }

    #[test]
    fn empty_input_is_error_not_panic() {
        assert!(parse(b"").is_err());
    }

    #[test]
    fn truncated_unicode_escape_is_error() {
        assert!(parse(br#"{"a":"\u12"}"#).is_err());
    }

    #[test]
    fn non_utf8_input_is_error_not_panic() {
        let bad = [b'"', 0xFF, 0xFE, b'"'];
        assert!(parse(&bad).is_err());
    }
}
