//! GGUF, the file format llama.cpp and most published quantized models use.
//!
//! Layout: magic `GGUF`, version, tensor count, metadata count; metadata as typed
//! key/values; tensor descriptions (name, shape, type, offset); then, aligned, the tensor
//! data. Everything a device needs to know fits in the header, so a device reads the
//! header and then only the byte ranges of its own tensors: a stage holding 20 of 80
//! layers downloads a quarter of the file.

use std::collections::HashMap;
use std::ops::Range;

use anyhow::{bail, Context, Result};

use crate::quant::GgmlType;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F64(v) => Some(v),
            Value::U64(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the row length.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte range in the file.
    pub range: Range<u64>,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
    /// Row length and row count, treating any extra dimensions as more rows.
    pub fn shape2(&self) -> (usize, usize) {
        let cols = self.dims.first().copied().unwrap_or(1) as usize;
        (cols, (self.elements() as usize) / cols.max(1))
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Header bytes; tensor data starts here.
    pub data_start: u64,
}

/// Raised when the header runs past the bytes given: read more and try again.
#[derive(Debug)]
pub struct NeedMore;

impl std::fmt::Display for NeedMore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the header continues past the bytes read")
    }
}
impl std::error::Error for NeedMore {}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.at + n > self.b.len() {
            return Err(NeedMore.into());
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > 1 << 24 {
            bail!("a {n}-byte string in a GGUF header");
        }
        Ok(String::from_utf8_lossy(self.take(n as usize)?).into_owned())
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        let le = |s: &[u8]| -> [u8; 8] {
            let mut b = [0u8; 8];
            b[..s.len()].copy_from_slice(s);
            b
        };
        Ok(match ty {
            0 => Value::U64(self.take(1)?[0] as u64),
            1 => Value::I64(self.take(1)?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 2 {
                    bail!("nested arrays in a GGUF header");
                }
                let item = self.u32()?;
                let n = self.u64()?;
                if n > 1 << 24 {
                    bail!("an array of {n} items in a GGUF header");
                }
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    items.push(self.value(item, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(le(self.take(8)?))),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => bail!("unknown GGUF value type {other}"),
        })
    }
}

/// Parses the header from the start of a file. Fails with [`NeedMore`] when `bytes` ends
/// inside the header.
pub fn parse(bytes: &[u8]) -> Result<Header> {
    let mut c = Cursor { b: bytes, at: 0 };
    if c.take(4)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        bail!("GGUF version {version} is not supported");
    }
    let n_tensors = c.u64()?;
    let n_meta = c.u64()?;
    if n_tensors > 1 << 20 || n_meta > 1 << 20 {
        bail!("implausible GGUF header ({n_tensors} tensors, {n_meta} keys)");
    }
    let mut meta = HashMap::new();
    for _ in 0..n_meta {
        let key = c.string()?;
        let ty = c.u32()?;
        let value = c.value(ty, 0)?;
        meta.insert(key, value);
    }
    let mut raw = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()?;
        if nd > 4 {
            bail!("tensor {name} has {nd} dimensions");
        }
        let dims: Vec<u64> = (0..nd).map(|_| c.u64()).collect::<Result<_>>()?;
        let ty = c.u32()?;
        let offset = c.u64()?;
        raw.push((name, dims, ty, offset));
    }
    let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32).max(1);
    let data_start = (c.at as u64).div_ceil(align) * align;
    let mut tensors = Vec::with_capacity(raw.len());
    for (name, dims, ty, offset) in raw {
        let ty = GgmlType::from_u32(ty).with_context(|| format!("tensor {name}"))?;
        let elements: u64 = dims.iter().product();
        let len = ty.bytes_for(elements as usize).with_context(|| format!("tensor {name}"))? as u64;
        let start = data_start + offset;
        tensors.push(TensorInfo { name, dims, ty, range: start..start + len });
    }
    Ok(Header { version, meta, tensors, data_start })
}

impl Header {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn arch(&self) -> &str {
        self.get("general.architecture").and_then(Value::as_str).unwrap_or("llama")
    }
    /// `<arch>.<key>`, as a number.
    pub fn arch_u64(&self, key: &str) -> Option<u64> {
        self.get(&format!("{}.{key}", self.arch())).and_then(Value::as_u64)
    }
    pub fn arch_f64(&self, key: &str) -> Option<f64> {
        self.get(&format!("{}.{key}", self.arch())).and_then(Value::as_f64)
    }
}

/// Writes a GGUF file: for test fixtures (a small random model) and nothing else.
pub mod write {
    pub enum Meta {
        U32(u32),
        F32(f32),
        Str(String),
        Strs(Vec<String>),
        F32s(Vec<f32>),
        I32s(Vec<i32>),
    }

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    pub fn gguf(meta: &[(&str, Meta)], tensors: &[(&str, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((meta.len() as u64).to_le_bytes());
        for (k, v) in meta {
            string(&mut out, k);
            match v {
                Meta::U32(x) => {
                    out.extend(4u32.to_le_bytes());
                    out.extend(x.to_le_bytes());
                }
                Meta::F32(x) => {
                    out.extend(6u32.to_le_bytes());
                    out.extend(x.to_le_bytes());
                }
                Meta::Str(s) => {
                    out.extend(8u32.to_le_bytes());
                    string(&mut out, s);
                }
                Meta::Strs(items) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(8u32.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    items.iter().for_each(|s| string(&mut out, s));
                }
                Meta::F32s(items) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(6u32.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    items.iter().for_each(|x| out.extend(x.to_le_bytes()));
                }
                Meta::I32s(items) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(5u32.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    items.iter().for_each(|x| out.extend(x.to_le_bytes()));
                }
            }
        }
        let mut offset = 0u64;
        let mut offsets = Vec::new();
        for (name, dims, ty, data) in tensors {
            string(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|d| out.extend(d.to_le_bytes()));
            out.extend(ty.to_le_bytes());
            out.extend(offset.to_le_bytes());
            offsets.push(offset);
            offset = (offset + data.len() as u64).div_ceil(32) * 32;
        }
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        let base = out.len();
        for ((_, _, _, data), off) in tensors.iter().zip(offsets) {
            out.resize(base + off as usize, 0);
            out.extend(data);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::write::{gguf, Meta};
    use super::{parse, NeedMore};

    #[test]
    fn round_trips_and_asks_for_more() {
        let file = gguf(
            &[("general.architecture", Meta::Str("llama".into())), ("llama.block_count", Meta::U32(2)), ("tokenizer.ggml.tokens", Meta::Strs(vec!["a".into(), "b".into()]))],
            &[("w", vec![32, 2], 0, vec![1u8; 256]), ("v", vec![32], 8, vec![2u8; 34])],
        );
        let h = parse(&file).unwrap();
        assert_eq!(h.arch(), "llama");
        assert_eq!(h.arch_u64("block_count"), Some(2));
        let w = h.tensor("w").unwrap();
        assert_eq!(w.shape2(), (32, 2));
        assert_eq!(&file[w.range.start as usize..w.range.end as usize], &[1u8; 256][..]);
        let v = h.tensor("v").unwrap();
        assert_eq!(&file[v.range.start as usize..v.range.end as usize], &[2u8; 34][..]);
        let short = parse(&file[..40]).unwrap_err();
        assert!(short.downcast_ref::<NeedMore>().is_some());
        assert!(parse(b"NOPE0000").is_err());
    }
}
