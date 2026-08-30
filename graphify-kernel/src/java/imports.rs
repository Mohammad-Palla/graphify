//! `extract.py::_import_java`, transliterated.
//!
//! Unlike the JS and Python import handlers this one touches no filesystem: a
//! Java import names a package path, and the handler only ever takes its last
//! dotted segment. So there is no resolver callback here and nothing to defer
//! on -- the whole handler is a string walk.

use tree_sitter::Node;

use super::{Ctx, R};
use crate::js::ast::children;

/// `_walk_scoped`: the dotted name under a `scoped_identifier` chain.
///
/// Walks the `scope` field down, collecting `name` on the way, then reverses.
/// A node that is neither `scoped_identifier` nor `identifier` stops the walk,
/// which is how a malformed import yields a short (possibly empty) name rather
/// than an error.
fn walk_scoped<'a>(ctx: &Ctx<'a, '_>, n: Node) -> R<String> {
    let mut parts: Vec<&'a str> = Vec::new();
    let mut cur = Some(n);
    while let Some(c) = cur {
        match c.kind() {
            "scoped_identifier" => {
                if let Some(name_node) = c.child_by_field_name("name") {
                    parts.push(ctx.text(name_node)?);
                }
                cur = c.child_by_field_name("scope");
            }
            "identifier" => {
                parts.push(ctx.text(c)?);
                break;
            }
            _ => break,
        }
    }
    parts.reverse();
    Ok(parts.join("."))
}

/// `_import_java`. Emits at most ONE edge and returns after the first
/// `scoped_identifier`/`identifier` child, matching Python's `break`.
pub fn import_java(ctx: &mut Ctx, node: Node) -> R<()> {
    for child in children(node) {
        if !matches!(child.kind(), "scoped_identifier" | "identifier") {
            continue;
        }
        let path_str = walk_scoped(ctx, child)?;
        // Python:
        //   module_name = path_str.split(".")[-1].strip("*").strip(".") or (
        //       path_str.split(".")[-2] if len(...) > 1 else path_str)
        // `strip("*")` and `strip(".")` strip from BOTH ends, and in that order.
        let segs: Vec<&str> = path_str.split('.').collect();
        let last = segs.last().copied().unwrap_or("");
        let stripped = last.trim_matches('*').trim_matches('.');
        let module_name: &str = if !stripped.is_empty() {
            stripped
        } else if segs.len() > 1 {
            segs[segs.len() - 2]
        } else {
            &path_str
        };
        if !module_name.is_empty() {
            let tgt = ctx.mkid(&[module_name])?;
            let line = node.start_position().row + 1;
            ctx.add_import_edge(&tgt, line);
        }
        // Python `break`s here. With tree-sitter-java it is not OBSERVABLE:
        // `import_declaration` is `'import' 'static'? (identifier |
        // scoped_identifier) ('.' '*')?`, so at most one child ever matches --
        // the `*` is an anonymous node, not an identifier. Replacing this with
        // `continue` changed nothing across the whole sweep. Kept because it
        // mirrors the Python, and because "only one child can match" is a claim
        // about a third-party grammar that a version bump could falsify.
        break;
    }
    Ok(())
}
