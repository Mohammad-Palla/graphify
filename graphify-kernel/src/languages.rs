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
use tree_sitter::Parser;

use crate::Outcome;

/// A native walker: bytes in, either a Graphify result dict or a reasoned deferral.
pub type Walker = for<'py> fn(
    Python<'py>,
    &str,
    &[u8],
    &crate::js::imports::Resolver<'py>,
) -> PyResult<Outcome<'py>>;

/// Languages with a parity-gated native walker.
///
/// A language appears here only after `harness/kernel_walker_parity.py` reports
/// **DIVERGENT 0** over every file of that language in the corpora, AND the
/// native rate is high enough for routing to be worth anything. Measured before
/// this list was populated, over 11,834 files in bun / vue / django:
///
/// ```text
/// corpus   language     files  native   deferred  DIVERGENT
/// bun      typescript    3165   99.1%      0.9%          0
/// bun      tsx            176   99.4%      0.6%          0
/// bun      javascript    7867   98.6%      1.4%          0
/// vue      typescript     474   99.6%      0.4%          0
/// vue      javascript      36  100.0%      0.0%          0
/// django   javascript     111   98.2%      1.8%          0
/// ```
///
/// The residual deferrals are, in order: a parse error (Python's recovery is
/// authoritative), a decorator (mints stub nodes through `ensure_named_node`), a
/// non-ASCII identifier (the id recipe's Unicode fixpoint is not reproduced), and
/// one 9 MB fixture too deep to recurse into.
///
/// `python` is deliberately absent: it is routable by `_GRAMMAR_TO_LANGUAGE` but
/// has no walker, so it defers at `walker_for`.
pub fn supported() -> Vec<&'static str> {
    vec!["typescript", "tsx", "javascript"]
}

pub fn walker_for(language: &str) -> Option<Walker> {
    match language {
        "typescript" => Some(crate::js::walk_typescript),
        "tsx" => Some(crate::js::walk_tsx),
        "javascript" => Some(crate::js::walk_javascript),
        _ => None,
    }
}

/// A structural fingerprint of each linked grammar, for the Python side to
/// compare against the grammar IT loads.
///
/// This exists because a version skew between the two sides is invisible and
/// produces exactly the failure mode this whole design is built to avoid. The
/// walker was measured DIVERGENT=0 on 3,165 TypeScript files and DIVERGENT on 8
/// of 7,867 JavaScript files, for one reason: `tree-sitter-typescript` was 0.23.2
/// on both sides, while `tree-sitter-javascript` was 0.25.0 in Python and 0.23.1
/// here. The two grammars parse `await f()` differently and disagree about
/// whether some files contain an error node at all -- so the kernel was faithfully
/// walking a *different tree* than Python would have.
///
/// Nothing in the build would have caught that: the crate pins a semver RANGE, and
/// `pip install --upgrade tree-sitter-javascript` moves the other side with no
/// signal. So the version agreement is checked at load time and a mismatch
/// disables that language.
///
/// `abi_version` alone is far too coarse (many grammar revisions share ABI 15).
/// The node-kind and field counts change with essentially any grammar revision,
/// which is what makes the triple a usable proxy for "same grammar".
pub fn grammar_fingerprints() -> Vec<(&'static str, u32, u32, u32)> {
    let langs: [(&'static str, tree_sitter::Language); 3] = [
        ("typescript", tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        ("tsx", tree_sitter_typescript::LANGUAGE_TSX.into()),
        ("javascript", tree_sitter_javascript::LANGUAGE.into()),
    ];
    langs
        .iter()
        .map(|(name, l)| {
            (
                *name,
                l.abi_version() as u32,
                l.node_kind_count() as u32,
                l.field_count() as u32,
            )
        })
        .collect()
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
