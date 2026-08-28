//! Native per-file AST extraction core for Graphify.
//!
//! # The contract
//!
//! [`extract_file`] takes a file's path, its bytes and its language, and returns
//! either a result dict in *exactly* the shape `graphify.extract._safe_extract`
//! produces, or `None` to mean **defer**: "I do not handle this, run the Python
//! extractor instead."
//!
//! Deferral is the load-bearing idea. A native walker is a large surface, and
//! the failure mode measured across this project's earlier pooling work was
//! never an exception -- it was silent wrong output. So the kernel never guesses:
//! anything it does not fully understand (an unsupported language, a node type it
//! has no rule for, a parse error, a construct whose Python semantics it cannot
//! reproduce byte-for-byte) is deferred, and the deferral is *counted* so a
//! rising rate is visible rather than hidden.
//!
//! # Why not columnar buffers
//!
//! The obvious design -- flat struct-of-arrays across the FFI boundary, as
//! CodeGraph's kernel does -- solves a problem we do not have. Their kernel
//! ships results between `worker_threads`, where every object pays a
//! structured-clone; ours is called *inside* a `ProcessPoolExecutor` worker that
//! already pickles the result once. Measured payloads are ~43 items per file
//! (~830k for all of Bun), so building dicts directly costs well under a second
//! against a 61s parse-and-walk. Columns would buy nothing and cost a decode
//! layer that could itself be wrong.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod ids;
mod languages;

/// Why a file was deferred to the Python extractor. Returned to Python so the
/// parity harness can report *which* gap dominates, not just that one exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Defer {
    /// No native walker is registered for this language at all.
    UnsupportedLanguage,
    /// The grammar reported an error node; Python's recovery is authoritative.
    ParseError,
    /// A node type this walker has no byte-identical rule for.
    UnhandledConstruct,
}

impl Defer {
    pub fn as_str(self) -> &'static str {
        match self {
            Defer::UnsupportedLanguage => "unsupported_language",
            Defer::ParseError => "parse_error",
            Defer::UnhandledConstruct => "unhandled_construct",
        }
    }
}

/// The kernel's own version, independent of Graphify's. Bumped whenever the
/// output of any walker changes, so a stale cached extraction can be detected.
const KERNEL_VERSION: &str = "0.1.0";

#[pyfunction]
fn version() -> &'static str {
    KERNEL_VERSION
}

/// Languages with a registered native walker. Empty in 0.1.0: the seam ships and
/// is proven safe before any language semantics are ported, so that a parity
/// failure later can only implicate the walker and never the plumbing.
#[pyfunction]
fn supported_languages() -> Vec<&'static str> {
    languages::supported()
}

/// Extract one file natively, or return `None` to defer to Python.
///
/// `source` is the file's raw bytes -- not a `str`. Graphify's Python extractors
/// read bytes and hand them to tree-sitter unchanged, and a lossy decode would
/// shift every byte offset in the file, so the boundary is bytes on both sides.
#[pyfunction]
#[pyo3(signature = (path, source, language))]
fn extract_file<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    language: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    match languages::walker_for(language) {
        None => Ok(None),
        Some(walker) => match walker(py, path, source) {
            Ok(Some(result)) => Ok(Some(result)),
            Ok(None) => Ok(None),
            Err(err) => Err(err),
        },
    }
}

/// Round-trip self-check callable from Python, so `kernel.py` can verify that the
/// loaded binary actually works before routing a single real file through it. A
/// kernel that imports but is subtly broken (ABI mismatch, wrong grammar
/// version) is worse than one that fails to load.
#[pyfunction]
fn selftest<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("version", KERNEL_VERSION)?;
    out.set_item("languages", PyList::new(py, languages::supported())?)?;
    // Prove the tree-sitter link is live, not just that the module loaded.
    out.set_item("tree_sitter_ok", languages::grammar_smoke_test())?;
    Ok(out)
}

/// Differential-test hooks. Exposed so the id primitives can be compared against
/// `graphify.ids.make_id` over real corpora rather than trusted from unit tests
/// -- these are the foundation every node and edge id is built on, so a
/// disagreement here would corrupt everything downstream silently.
#[pyfunction]
fn debug_make_id(parts: Vec<String>) -> Option<String> {
    let refs: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
    ids::make_id_ascii(&refs)
}

#[pyfunction]
fn debug_file_stem(path: &str) -> Option<String> {
    ids::file_stem(path)
}

#[pymodule]
fn graphify_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(supported_languages, m)?)?;
    m.add_function(wrap_pyfunction!(extract_file, m)?)?;
    m.add_function(wrap_pyfunction!(selftest, m)?)?;
    m.add_function(wrap_pyfunction!(debug_make_id, m)?)?;
    m.add_function(wrap_pyfunction!(debug_file_stem, m)?)?;
    Ok(())
}
