//! `_import_kotlin`: `import a.b.C`, `import a.b.*`, `import a.b.C as D`.

use tree_sitter::Node;

use crate::engine::meta;
use crate::engine::{Ctx, R};
use crate::js::ast::children;
use crate::js::emit::{EdgeRow, Val};

pub fn import_kotlin(ctx: &mut Ctx, node: Node) -> R<()> {
    // Grammar 1.1.0 emits an `import` node whose children are the keyword and a
    // `qualified_identifier`; older forks emit `import_header` with a `path`
    // field or a bare `identifier` spanning the whole dotted text (#2526).
    let mut path_node = node.child_by_field_name("path");
    if path_node.is_none() {
        path_node = children(node)
            .into_iter()
            .find(|c| c.kind() == "qualified_identifier");
    }
    let raw: String = match path_node {
        Some(p) => ctx.text(p)?.trim().to_string(),
        None => children(node)
            .into_iter()
            .find(|c| c.kind() == "identifier")
            .map(|c| ctx.text(c).map(|t| t.trim().to_string()))
            .transpose()?
            .unwrap_or_default(),
    };
    if raw.is_empty() {
        return Ok(());
    }
    // A wildcard imports a whole PACKAGE, not a symbol: the last segment is a
    // package name, so a symbol-level edge would dangle on -- or collide with --
    // an unrelated node that happens to share it.
    if raw.ends_with(".*") || raw == "*" || children(node).into_iter().any(|c| c.kind() == "*") {
        return Ok(());
    }
    // `import a.b.C as D`: the alias is the identifier child AFTER `as`.
    let mut alias: Option<String> = None;
    let mut saw_as = false;
    for child in children(node) {
        if !saw_as {
            saw_as = child.kind() == "as";
        } else if matches!(child.kind(), "identifier" | "simple_identifier") {
            let a = ctx.text(child)?.trim();
            alias = if a.is_empty() { None } else { Some(a.to_string()) };
            break;
        }
    }
    let module_name = raw.rsplit('.').next().unwrap_or("").trim();
    if module_name.is_empty() {
        return Ok(());
    }
    let target = ctx.mkid(&[module_name])?;
    let line = node.start_position().row + 1;
    // The target is the bare last segment for now;
    // `_resolve_kotlin_import_targets` rewrites it to the real node id via the
    // `target_fqn` stamped here, once the per-file package index exists.
    let mut md: Vec<(String, Val)> = vec![("target_fqn".to_string(), Val::S(raw))];
    if let Some(a) = alias {
        md.push(("alias".to_string(), Val::S(a)));
    }
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
