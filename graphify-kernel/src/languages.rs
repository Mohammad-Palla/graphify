//! Language registry: the single place that decides whether a file goes native.
//!
//! A language appears in [`supported`] only once its walker has passed the parity
//! gate on a real corpus (see `harness/kernel_parity.py`). Until then it is
//! absent, and every file of that language defers to Python. This is deliberately
//! a *whitelist* keyed on the language Graphify already resolved -- never a
//! suffix sniff of our own, which could disagree with `_get_extractor`'s
//! filename special cases (`.blade.php`, MCP configs, ObjC vs MATLAB `.m`,
//! C vs C++ `.h`) and silently route a file to the wrong grammar.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tree_sitter::Parser;

/// A native walker: bytes in, either a Graphify result dict or a deferral.
pub type Walker = for<'py> fn(Python<'py>, &str, &[u8]) -> PyResult<Option<Bound<'py, PyDict>>>;

/// Languages with a parity-gated native walker.
///
/// Empty in 0.1.0 on purpose. Shipping the seam with nothing routed makes the
/// first milestone falsifiable: if the cold equivalence gate passes with the
/// kernel loaded and every file deferring, then the routing, fallback and
/// accounting are correct, and any later parity failure is the walker's fault
/// alone. Starting with a half-finished TypeScript walker would confound the two.
pub fn supported() -> Vec<&'static str> {
    Vec::new()
}

pub fn walker_for(_language: &str) -> Option<Walker> {
    None
}

/// Parse a trivial TypeScript source to prove the grammar is linked and
/// ABI-compatible. Returns false rather than panicking: a failure here must make
/// `kernel.py` disable the kernel, not abort the build.
pub fn grammar_smoke_test() -> bool {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .is_err()
    {
        return false;
    }
    match parser.parse("const x: number = 1;", None) {
        Some(tree) => {
            let root = tree.root_node();
            root.kind() == "program" && !root.has_error()
        }
        None => false,
    }
}
