//! `_import_php`: `use Foo\Bar;`

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

pub fn import_php(ctx: &mut Ctx, node: Node) -> R<()> {
    for child in children(node) {
        if !matches!(child.kind(), "qualified_name" | "name" | "identifier") {
            continue;
        }
        // The unqualified tail: PHP namespaces separate with a backslash.
        let raw = ctx.text(child)?;
        let module_name = raw.rsplit('\\').next().unwrap_or(raw).trim();
        if !module_name.is_empty() {
            let target = ctx.mkid(&[module_name])?;
            let line = node.start_position().row + 1;
            ctx.add_import_edge(&target, line);
        }
        // The FIRST name-ish child only, matched or not.
        return Ok(());
    }
    Ok(())
}
