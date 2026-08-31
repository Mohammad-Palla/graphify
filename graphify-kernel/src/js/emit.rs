//! Result rows and their conversion to Python dicts.
//!
//! # Why the fields are an ordered list and not a struct
//!
//! Graphify's extractors build each node/edge as a literal `dict`, and CPython
//! dicts preserve insertion order -- which `json.dump` then writes out verbatim,
//! because the export does not sort keys. So "byte-identical output" includes the
//! *order* of the keys, not just their values, and the six edge shapes JS/TS emits
//! do not agree on one order: `add_edge` puts `context` last, while the import,
//! call and dynamic-import literals put it third and interleave
//! `confidence_score` / `deferred` in between.
//!
//! A struct with `Option` fields would force one order on all of them and be
//! silently wrong for five of the six. The parity harness would not catch it
//! either: it canonicalizes with `sort_keys=True`, so key order is exactly the
//! kind of difference that survives every per-file check and only shows up as a
//! whole-graph byte diff much later. Carrying the pairs in emission order makes
//! the order a thing each call site states explicitly.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A field value in a result dict.
///
/// `None` is a variant rather than an absent key because the two are different
/// dicts: a `raw_calls` entry always carries `"receiver"`, and its value is
/// Python `None` when no receiver was captured. Dropping the key instead would
/// canonicalize differently and read as a divergence.
pub enum Val {
    S(String),
    Static(&'static str),
    F(f64),
    /// A Python INT, which `F` is not. `extract_elixir`'s result carries
    /// `input_tokens: 0` / `output_tokens: 0` as ints, and `0` and `0.0` are
    /// different values to `json.dumps` -- so emitting them through `F` would
    /// report every Elixir file DIVERGENT.
    I(i64),
    B(bool),
    None,
    /// A JSON list. Only `scope_chain` uses it: C#'s `add_node` stamps
    /// `list(scope_stack)` on every node declared inside a namespace.
    List(Vec<Val>),
    /// A nested dict, in insertion order -- the `metadata` block. Keys are
    /// owned `String` rather than `&'static str` because C# metadata keys come
    /// from `sanitize_metadata`, which rewrites them.
    Meta(Vec<(String, Val)>),
}

impl Val {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Val::S(s) => s.into_pyobject(py)?.into_any(),
            Val::Static(s) => (*s).into_pyobject(py)?.into_any(),
            Val::F(f) => f.into_pyobject(py)?.into_any(),
            Val::I(i) => i.into_pyobject(py)?.into_any(),
            // `bool` converts to the SINGLETON `Py_True`/`Py_False`, whose
            // `into_pyobject` yields a `Borrowed`; `.to_owned()` makes the two
            // match arms one type.
            Val::B(b) => b.into_pyobject(py)?.to_owned().into_any(),
            Val::None => py.None().into_bound(py),
            Val::List(items) => {
                let l = pyo3::types::PyList::empty(py);
                for it in items {
                    l.append(it.to_py(py)?)?;
                }
                l.into_any()
            }
            Val::Meta(pairs) => {
                let d = PyDict::new(py);
                for (k, v) in pairs {
                    d.set_item(k, v.to_py(py)?)?;
                }
                d.into_any()
            }
        })
    }

    fn set(&self, d: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
        d.set_item(key, self.to_py(d.py())?)
    }

    /// The same, public: `engine::extract` writes `result_extra` values into the
    /// result dict without going through a `NodeRow` or an `EdgeRow`.
    pub fn set_on(&self, d: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
        self.set(d, key)
    }
}

pub struct NodeRow {
    pub id: String,
    /// Everything after `id`, in insertion order.
    pub fields: Vec<(&'static str, Val)>,
}

pub struct EdgeRow {
    pub source: String,
    pub target: String,
    pub relation: &'static str,
    /// Everything after `relation`, in insertion order.
    pub fields: Vec<(&'static str, Val)>,
}

pub type RawCall = Vec<(&'static str, Val)>;

pub fn node_to_py<'py>(
    py: Python<'py>,
    n: &NodeRow,
    callable_def: bool,
    callable_class: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", &n.id)?;
    for (k, v) in &n.fields {
        v.set(&d, k)?;
    }
    // Stamped after the walk in Python too (`for n in nodes: if n["id"] in
    // callable_def_nids`), so these are the last keys in the dict.
    if callable_def {
        d.set_item("_callable", true)?;
        if callable_class {
            d.set_item("_callable_class", true)?;
        }
    }
    Ok(d)
}

pub fn edge_to_py<'py>(py: Python<'py>, e: &EdgeRow) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("source", &e.source)?;
    d.set_item("target", &e.target)?;
    // `relation` is `&'static str` for every edge but PHP's framework ones,
    // whose relation is BUILT (`uses_{helper}`). Those carry it as the first
    // field under `__relation`, which is written into the same slot so the key
    // order is identical either way.
    if let Some((_, Val::S(r))) = e.fields.first().filter(|(k, _)| *k == "__relation") {
        d.set_item("relation", r)?;
    } else {
        d.set_item("relation", e.relation)?;
    }
    for (k, v) in &e.fields {
        if *k == "__relation" {
            continue;
        }
        v.set(&d, k)?;
    }
    Ok(d)
}

pub fn raw_call_to_py<'py>(py: Python<'py>, c: &RawCall) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in c {
        v.set(&d, k)?;
    }
    Ok(d)
}

