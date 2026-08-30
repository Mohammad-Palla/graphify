//! `_import_csharp`: the `using` directive.

use tree_sitter::Node;

use crate::engine::meta;
use crate::engine::{Ctx, R};
use crate::js::emit::{EdgeRow, Val};

use super::helpers::py_strip;

pub fn import_csharp(ctx: &mut Ctx, node: Node) -> R<()> {
    let raw = ctx.text(node)?;
    let mut text = py_strip(raw).trim_end_matches(';');
    // `global using X;` -- C# 10's file-global form. The `global ` prefix is
    // stripped before the `using` test, so it takes the same path.
    if let Some(rest) = text.strip_prefix("global ") {
        text = py_strip(rest);
    }
    if !text.starts_with("using") {
        return Ok(());
    }
    let body = py_strip(&text["using".len()..]);

    let mut using_kind = "namespace";
    let mut alias: Option<&str> = None;
    let mut target_fqn = body;
    if let Some(rest) = body.strip_prefix("static ") {
        using_kind = "static";
        target_fqn = py_strip(rest);
    } else if let Some((lhs, rhs)) = body.split_once('=') {
        using_kind = "alias";
        alias = Some(py_strip(lhs));
        target_fqn = py_strip(rhs);
    }
    if target_fqn.is_empty() {
        return Ok(());
    }

    let target = ctx.mkid(&[target_fqn])?;
    let line = node.start_position().row + 1;
    // The `{...}.items() if v is not None` filter: `alias` is absent unless this
    // is an alias using, and `scope_id` unless the directive sits inside a
    // namespace. `using_kind`, `target_fqn` and `scope_kind` are never None.
    let mut md: Vec<(String, Val)> = vec![("using_kind".to_string(), Val::Static(using_kind))];
    if let Some(a) = alias {
        md.push(("alias".to_string(), Val::S(a.to_string())));
    }
    md.push(("target_fqn".to_string(), Val::S(target_fqn.to_string())));
    md.push((
        "scope_kind".to_string(),
        Val::Static(if ctx.scope_stack.is_empty() { "file" } else { "namespace" }),
    ));
    if let Some(s) = ctx.scope_stack.last() {
        md.push(("scope_id".to_string(), Val::S(s.clone())));
    }

    // Built as a dict literal in Python, NOT through `add_edge`: `context` comes
    // third and `metadata` is always present.
    ctx.edges.push(EdgeRow {
        source: ctx.file_nid.clone(),
        target,
        relation: "imports",
        fields: vec![
            ("context", Val::Static("import")),
            ("confidence", Val::Static("EXTRACTED")),
            ("source_file", Val::S(ctx.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
            ("weight", Val::F(1.0)),
            ("metadata", Val::Meta(meta::sanitize(md))),
        ],
    });
    Ok(())
}
