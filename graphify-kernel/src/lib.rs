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
mod js;
mod languages;
mod py;

/// A walker's verdict on one file: a finished result, or a deferral *with the
/// reason*.
///
/// The reason is the whole point. A walker that defers is correct but buys
/// nothing, and without a reason the only visible signal is a percentage --
/// which tells you that a gap exists but never which one, so the next construct
/// to implement has to be guessed. `&'static str` because every reason is either
/// a literal or a tree-sitter node kind, which the grammar already owns for the
/// life of the process: naming a gap costs no allocation, so the production path
/// pays nothing for being explainable.
pub enum Outcome<'py> {
    Native(Bound<'py, PyDict>),
    Defer(&'static str),
}

/// Shorthand for the deferral arm, so a walker reads `return defer("kind:x")`.
pub fn defer<'py>(reason: &'static str) -> PyResult<Outcome<'py>> {
    Ok(Outcome::Defer(reason))
}

/// Every language's Python-side resolver, handed to whichever walker runs.
///
/// One bundle rather than a per-language parameter because [`languages::Walker`]
/// is a single fn-pointer type: making the resolver type vary per language would
/// need either an enum at the dispatch point or a registry per language, and the
/// registry being ONE table is what makes "which languages go native" answerable
/// by reading one function. Construction is cheap -- each holds an `Option`
/// callable and an empty memo -- so a walker paying for a bundle it half-ignores
/// costs nothing measurable against a parse.
pub struct Resolvers<'py> {
    pub js: js::imports::Resolver<'py>,
    pub py: py::imports::Resolver<'py>,
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

/// Extract one file natively. Returns `(result, defer_reason)` -- exactly one of
/// which is non-None.
///
/// The reason travels with the deferral rather than being derivable from it,
/// because the two callers need different things from the same call and neither
/// should have to re-derive it: `kernel.try_extract` tallies reasons so a pooled
/// build can print a breakdown, and the parity harness ranks them to pick the
/// next construct to implement. Returning a bare `None` made both of those
/// guesswork.
///
/// `source` is the file's raw bytes -- not a `str`. Graphify's Python extractors
/// read bytes and hand them to tree-sitter unchanged, and a lossy decode would
/// shift every byte offset in the file, so the boundary is bytes on both sides.
///
/// `resolve_import` is a callable `(specifier) -> tuple | None` wrapping
/// `_resolve_js_import_target`. Omitting it does not disable imports -- it makes
/// any file containing one defer, since resolution has no safe default.
///
/// `resolve_module` is the same idea for `_resolve_js_module_path`, used only by
/// the symbol-fact collector. Omitting it drops `js_symbol_facts` from the result
/// (phase 3 then collects them in Python) but leaves nodes/edges native.
///
/// `resolve_py_import` is Python's RELATIVE-import resolution -- the `Path.parent`
/// walk plus `_probe_python_module_candidate`'s filesystem probes. It is a THIRD
/// parameter rather than a reuse of `resolve_import`, deliberately: the two
/// callables answer different questions and return different shapes, and a slot
/// whose meaning depends on `language` is exactly the kind of thing that goes
/// silently wrong. Omitting it makes any Python file containing a relative import
/// defer, since there is no safe default for where it points.
#[pyfunction]
#[pyo3(signature = (path, source, language, resolve_import=None, resolve_module=None, resolve_py_import=None))]
fn extract_file<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    language: &str,
    resolve_import: Option<Bound<'py, PyAny>>,
    resolve_module: Option<Bound<'py, PyAny>>,
    resolve_py_import: Option<Bound<'py, PyAny>>,
) -> PyResult<(Option<Bound<'py, PyDict>>, Option<&'static str>)> {
    let res = Resolvers {
        js: js::imports::Resolver::new(resolve_import, resolve_module),
        py: py::imports::Resolver::new(resolve_py_import),
    };
    match languages::walker_for(language) {
        None => Ok((None, Some("no_walker"))),
        Some(walker) => match walker(py, path, source, &res)? {
            Outcome::Native(result) => Ok((Some(result), None)),
            Outcome::Defer(reason) => Ok((None, Some(reason))),
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
    // language -> (abi_version, node_kind_count, field_count), so `kernel.py` can
    // refuse any language whose grammar does not match the one Python loads. See
    // `languages::grammar_fingerprints` for why this is not optional.
    let fps = PyDict::new(py);
    for (name, abi, kinds, fields) in languages::grammar_fingerprints() {
        fps.set_item(name, (abi, kinds, fields))?;
    }
    out.set_item("grammars", fps)?;
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

/// Parse `source` and run ONE full-tree traversal over it, returning the tree's
/// depth.
///
/// An attribution hook, not a feature. There is no native profiler on this box
/// (`perf` and `valgrind` are absent, and `py-spy` sees a single opaque frame for
/// the whole of `extract_file`), so the only way to find out what the native path
/// spends its time on is to measure a part of it in isolation. This exposes the
/// cheapest possible traversal -- the depth probe, which touches every node and
/// does nothing else -- so its cost can be subtracted from parse and compared
/// against a full extract. The walker performs roughly seven full-tree
/// traversals per file, so one traversal's cost multiplied out is the size of the
/// prize for making them cheaper.
#[pyfunction]
#[pyo3(signature = (source, language))]
fn debug_traversal_cost(source: &[u8], language: &str) -> Option<u32> {
    js::debug_traversal_cost(source, language)
}

/// Panic on purpose, so the seam's crash containment can be tested against a REAL
/// Rust panic rather than a stand-in.
///
/// This matters because of a detail that is easy to get wrong: PyO3 turns a panic
/// into `pyo3_runtime.PanicException`, which derives from `BaseException`, NOT
/// `Exception`. A `try/except Exception` around the call therefore does not catch
/// it, and the panic escapes through `_safe_extract` (also `except Exception`) and
/// out of the pool worker, where `ProcessPoolExecutor` surfaces it as a failure
/// for every file that worker was holding. One malformed file would take out a
/// batch. The contract is that a native failure is a deferral, so this is
/// exercised rather than assumed.
#[pyfunction]
fn debug_panic() -> PyResult<()> {
    panic!("deliberate panic from graphify_kernel::debug_panic");
}

#[pymodule]
fn graphify_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(supported_languages, m)?)?;
    m.add_function(wrap_pyfunction!(extract_file, m)?)?;
    m.add_function(wrap_pyfunction!(selftest, m)?)?;
    m.add_function(wrap_pyfunction!(debug_make_id, m)?)?;
    m.add_function(wrap_pyfunction!(debug_file_stem, m)?)?;
    m.add_function(wrap_pyfunction!(debug_panic, m)?)?;
    m.add_function(wrap_pyfunction!(debug_traversal_cost, m)?)?;
    Ok(())
}
