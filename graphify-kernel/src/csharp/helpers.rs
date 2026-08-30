//! The `_csharp_*` helpers from `engine.py`, one for one.

use std::collections::HashSet;

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// The `role` half of a `_csharp_collect_type_refs` tuple.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Role {
    Type,
    GenericArg,
}

/// One collected reference: `(name, role, qualified, qualifier)`.
pub type TypeRef = (String, Role, bool, String);

/// `str.strip()` / `str.rstrip(...)`, Python's whitespace set.
///
/// Python's `str.isspace()` is Unicode White_Space PLUS `\x1c`-`\x1f`, which
/// Rust's `char::is_whitespace` excludes. The four extra characters are
/// vanishingly rare in C# source, which is exactly why the difference would
/// never show up in a corpus run and would be wrong forever.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `str.rpartition(sep)`: `(head, sep, tail)`, and when `sep` is absent the
/// EMPTY string comes first -- `"Foo".rpartition(".") == ("", "", "Foo")`. Only
/// the head and tail are used here.
pub fn rpartition<'a>(s: &'a str, sep: char) -> (&'a str, &'a str) {
    match s.rfind(sep) {
        Some(i) => (&s[..i], &s[i + sep.len_utf8()..]),
        None => ("", s),
    }
}

/// Python's `name[:1].isupper()`.
///
/// Non-ASCII defers rather than guessing: `str.isupper()` is the Unicode Cased +
/// Uppercase rule and Rust's `char::is_uppercase` is the Uppercase property
/// alone, so a titlecase or otherwise cased-but-not-uppercase leading character
/// would silently classify differently -- and unlike an id, this value never
/// reaches `make_id`, so nothing else would catch it.
pub fn first_is_upper(name: &str) -> R<bool> {
    if !name.is_ascii() {
        return Err("non_ascii_id");
    }
    Ok(name.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false))
}

const TYPE_PARAMETER_SCOPE_DECLARATIONS: &[&str] = &[
    "class_declaration",
    "interface_declaration",
    "record_declaration",
    "struct_declaration",
    "method_declaration",
];

/// `_csharp_type_parameters_in_scope`: every type-parameter name visible from
/// `node`, walking up through enclosing declarations.
pub fn type_parameters_in_scope(ctx: &Ctx, node: Node) -> R<HashSet<String>> {
    let mut names = HashSet::new();
    let mut scope = Some(node);
    while let Some(s) = scope {
        if TYPE_PARAMETER_SCOPE_DECLARATIONS.contains(&s.kind()) {
            for child in children(s) {
                if child.kind() != "type_parameter_list" {
                    continue;
                }
                for param in children(child) {
                    if param.kind() == "type_parameter" {
                        if let Some(n) = children(param).into_iter().find(|c| c.kind() == "identifier")
                        {
                            let name = ctx.text(n)?;
                            if !name.is_empty() {
                                names.insert(name.to_string());
                            }
                        }
                    } else if param.kind() == "identifier" {
                        let name = ctx.text(param)?;
                        if !name.is_empty() {
                            names.insert(name.to_string());
                        }
                    }
                }
            }
        }
        scope = s.parent();
    }
    Ok(names)
}

/// `_read_csharp_type_name`: `(name, qualified, qualifier)`.
pub fn read_type_name(ctx: &Ctx, node: Option<Node>) -> R<Option<(String, bool, String)>> {
    let node = match node {
        Some(n) => n,
        None => return Ok(None),
    };
    match node.kind() {
        "identifier" | "predefined_type" => {
            return Ok(Some((ctx.text(node)?.to_string(), false, String::new())))
        }
        "qualified_name" => {
            let text = ctx.text(node)?;
            let (prefix, tail) = rpartition(text, '.');
            let tail = tail.split_once('<').map(|(a, _)| a).unwrap_or(tail);
            return Ok(Some((tail.to_string(), true, prefix.to_string())));
        }
        "generic_name" => {
            // Only when the `name` field is present; otherwise the Python falls
            // THROUGH to the child scan below rather than returning None.
            if let Some(name_node) = node.child_by_field_name("name") {
                let qualified = name_node.kind() == "qualified_name";
                let (prefix, tail) = rpartition(ctx.text(name_node)?, '.');
                return Ok(Some((
                    tail.to_string(),
                    qualified,
                    if qualified { prefix.to_string() } else { String::new() },
                )));
            }
        }
        _ => {}
    }
    for child in children(node) {
        if !child.is_named() {
            continue;
        }
        if let Some(result) = read_type_name(ctx, Some(child))? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// `_csharp_collect_type_refs`: walk a type expression, appending every named
/// type it mentions with the role it plays.
///
/// `skip` is the type-parameter set. `None` means "compute it from `node`",
/// which is what the Python's `skip is None` default does -- and two call sites
/// (the property block, and the class-level attribute scan) rely on it.
pub fn collect_type_refs(
    ctx: &Ctx,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<TypeRef>,
    skip: Option<&HashSet<String>>,
) -> R<()> {
    let node = match node {
        Some(n) => n,
        None => return Ok(()),
    };
    let owned;
    let skip = match skip {
        Some(s) => s,
        None => {
            owned = type_parameters_in_scope(ctx, node)?;
            &owned
        }
    };
    let role = if generic { Role::GenericArg } else { Role::Type };
    match node.kind() {
        "predefined_type" => return Ok(()),
        "identifier" => {
            let name = ctx.text(node)?;
            if !name.is_empty() && !skip.contains(name) {
                out.push((name.to_string(), role, false, String::new()));
            }
            return Ok(());
        }
        "qualified_name" => {
            let (prefix, tail) = rpartition(ctx.text(node)?, '.');
            let tail = tail.split_once('<').map(|(a, _)| a).unwrap_or(tail);
            if !tail.is_empty() && !skip.contains(tail) {
                out.push((tail.to_string(), role, true, prefix.to_string()));
            }
            return Ok(());
        }
        "generic_name" => {
            let name_child = node
                .child_by_field_name("name")
                .or_else(|| children(node).into_iter().find(|c| c.kind() == "identifier"));
            if let Some(name_child) = name_child {
                let qualified = name_child.kind() == "qualified_name";
                let (prefix, name) = rpartition(ctx.text(name_child)?, '.');
                if !name.is_empty() && !skip.contains(name) {
                    out.push((
                        name.to_string(),
                        role,
                        qualified,
                        if qualified { prefix.to_string() } else { String::new() },
                    ));
                }
            }
            for sub in children(node) {
                if sub.kind() != "type_argument_list" {
                    continue;
                }
                for arg in children(sub) {
                    if arg.is_named() {
                        collect_type_refs(ctx, Some(arg), true, out, Some(skip))?;
                    }
                }
            }
            return Ok(());
        }
        "nullable_type" | "array_type" | "pointer_type" | "ref_type" => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out, Some(skip))?;
                }
            }
            return Ok(());
        }
        _ => {}
    }
    if node.is_named() {
        for c in children(node) {
            if c.is_named() {
                collect_type_refs(ctx, Some(c), generic, out, Some(skip))?;
            }
        }
    }
    Ok(())
}

/// `_csharp_attribute_names`: `(name, qualified, qualifier)` per `[Attribute]`.
pub fn attribute_names(ctx: &Ctx, decl: Node) -> R<Vec<(String, bool, String)>> {
    let mut names = Vec::new();
    let skip = type_parameters_in_scope(ctx, decl)?;
    for child in children(decl) {
        if child.kind() != "attribute_list" {
            continue;
        }
        for attr in children(child) {
            if attr.kind() != "attribute" {
                continue;
            }
            let name_node = attr.child_by_field_name("name").or_else(|| {
                children(attr)
                    .into_iter()
                    .find(|c| matches!(c.kind(), "identifier" | "qualified_name"))
            });
            if let Some(name_node) = name_node {
                let qualified = name_node.kind() == "qualified_name";
                let (prefix, text) = rpartition(ctx.text(name_node)?, '.');
                if !text.is_empty() && !skip.contains(text) {
                    names.push((
                        text.to_string(),
                        qualified,
                        if qualified { prefix.to_string() } else { String::new() },
                    ));
                }
            }
        }
    }
    Ok(names)
}

/// `_csharp_receiver_type_name`: a declared type reduced to a receiver-typable
/// class name, or None. Pascal-case only -- a primitive owns no resolvable
/// method, and `var` (`implicit_type`) carries no name at all.
pub fn receiver_type_name(ctx: &Ctx, type_node: Option<Node>) -> R<Option<String>> {
    let info = match read_type_name(ctx, type_node)? {
        Some(i) => i,
        None => return Ok(None),
    };
    let name = info.0;
    if !name.is_empty() && first_is_upper(&name)? {
        Ok(Some(name))
    } else {
        Ok(None)
    }
}

/// `_csharp_classify_base`: `implements` when the base is a declared interface
/// or follows the `IFoo` convention, else `inherits`.
pub fn classify_base(name: &str, interface_names: &HashSet<String>) -> R<&'static str> {
    if interface_names.contains(name) {
        return Ok("implements");
    }
    if !name.is_ascii() {
        return Err("non_ascii_id");
    }
    let b = name.as_bytes();
    if b.len() >= 2 && b[0] == b'I' && b[1].is_ascii_uppercase() {
        return Ok("implements");
    }
    Ok("inherits")
}

/// `_csharp_pre_scan_interfaces`.
pub fn pre_scan_interfaces(ctx: &Ctx, root: Node) -> R<HashSet<String>> {
    let mut out = HashSet::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if n.kind() == "interface_declaration" {
            if let Some(name_node) = n.child_by_field_name("name") {
                let text = ctx.text(name_node)?;
                if !text.is_empty() {
                    out.insert(text.to_string());
                }
            }
        }
        stack.extend(children(n));
    }
    Ok(out)
}

/// `_csharp_namespace_name`.
pub fn namespace_name(ctx: &Ctx, node: Node) -> R<String> {
    if let Some(name_node) = node.child_by_field_name("name") {
        return Ok(py_strip(ctx.text(name_node)?).to_string());
    }
    for child in children(node) {
        if matches!(child.kind(), "identifier" | "qualified_name") {
            return Ok(py_strip(ctx.text(child)?).to_string());
        }
    }
    Ok(String::new())
}

/// `_csharp_namespace_id`: `csharp_namespace:<sha1(dotted)[:16]>`.
///
/// NOT run through `make_id` -- the colon survives into the node id, which is
/// what distinguishes a namespace node from every symbol node.
pub fn namespace_id(dotted_name: &str) -> String {
    use sha1::{Digest, Sha1};
    let digest = Sha1::digest(dotted_name.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("csharp_namespace:{}", &hex[..16])
}
