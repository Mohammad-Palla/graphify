//! The `_swift_*` free functions from `engine.py`, reproduced.
//!
//! Kept in their own module because Swift has more of them than any other
//! language on this engine -- eleven, against Scala's one -- and because four of
//! them (`constructor_type`, `factory_call`, `attribute_type_name`,
//! `receiver_name`) are deliberately NARROW: they recognise one exact shape and
//! return `None` for anything deeper. Precision over recall is the documented
//! choice at each site (#2561/#1604), so widening any of them "to catch more"
//! fabricates edges rather than finding missed ones.

use std::collections::{HashMap, HashSet};

use tree_sitter::Node;

use crate::engine::{Ctx, R};
use crate::js::ast::children;

/// `text[:1].isupper()`.
///
/// Python's `str.isupper()` on a one-character slice is true only for a CASED
/// uppercase character, which is Rust's `char::is_uppercase` (titlecase `Lt` is
/// neither, in both languages). An empty string is false in both.
fn starts_upper(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_uppercase())
}

/// `_swift_declaration_keyword`: the leading UNNAMED kind token of a
/// `class_declaration`.
///
/// tree-sitter-swift parses `class`, `struct`, `enum`, `extension` and `actor`
/// all as `class_declaration`; this token is the only thing that tells them
/// apart, and `classify_base` needs it to know whether a base name can possibly
/// be a superclass.
pub fn declaration_keyword(node: Node) -> Option<&'static str> {
    for c in children(node) {
        if c.is_named() {
            continue;
        }
        match c.kind() {
            "class" => return Some("class"),
            "struct" => return Some("struct"),
            "enum" => return Some("enum"),
            "extension" => return Some("extension"),
            "actor" => return Some("actor"),
            _ => {}
        }
    }
    None
}

/// `_swift_pre_scan`: `(protocol names, class-like names)` for the whole file.
///
/// The walk order is Python's LIFO stack exactly (`stack.pop()` after
/// `stack.extend(children)`). Both outputs are sets, so order cannot reach the
/// result here -- unlike `local_var_types` below, where it decides which binding
/// wins.
pub fn pre_scan<'tree>(
    ctx: &Ctx<'_, 'tree>,
    root: Node<'tree>,
) -> R<(HashSet<String>, HashSet<String>)> {
    let mut protocols = HashSet::new();
    let mut classes = HashSet::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        match n.kind() {
            "protocol_declaration" => {
                let name_node = n
                    .child_by_field_name("name")
                    .or_else(|| children(n).into_iter().find(|c| c.kind() == "type_identifier"));
                if let Some(nn) = name_node {
                    let text = ctx.text(nn)?;
                    if !text.is_empty() {
                        protocols.insert(text.to_string());
                    }
                }
            }
            "class_declaration" => {
                // `extension` is deliberately absent: an extension declares no
                // new type, so its name must not join the class set or a
                // conformance on it would be classified `inherits`.
                if matches!(
                    declaration_keyword(n),
                    Some("class") | Some("struct") | Some("enum") | Some("actor")
                ) {
                    // The `name` FIELD only -- no `type_identifier` fallback
                    // here, unlike the protocol branch above.
                    if let Some(nn) = n.child_by_field_name("name") {
                        let text = ctx.text(nn)?;
                        if !text.is_empty() {
                            classes.insert(text.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
        stack.extend(children(n));
    }
    Ok((protocols, classes))
}

/// `_swift_classify_base`: is this `inheritance_specifier` entry a superclass or
/// a protocol conformance?
///
/// The order of the four rules is the whole point. A name DECLARED in this file
/// answers definitively, so both sets are consulted first. Only then does the
/// declaring keyword decide: a `struct`, `enum`, `extension` or `actor` cannot
/// inherit a class at all, so every entry on one is a conformance. A `class`
/// falls back to Swift's own convention -- first entry is the base class, the
/// rest are protocols -- which is a guess, and the only guess here.
pub fn classify_base(
    name: &str,
    kind: Option<&str>,
    is_first: bool,
    protocols: &HashSet<String>,
    classes: &HashSet<String>,
) -> &'static str {
    if protocols.contains(name) {
        return "implements";
    }
    if classes.contains(name) {
        return "inherits";
    }
    if matches!(kind, Some("struct") | Some("enum") | Some("extension") | Some("actor")) {
        return "implements";
    }
    if is_first {
        "inherits"
    } else {
        "implements"
    }
}

/// `_swift_user_type_name`: the head `type_identifier` of a `user_type`, without
/// its generic arguments.
pub fn user_type_name(ctx: &Ctx, node: Node) -> R<Option<String>> {
    for c in children(node) {
        if c.kind() == "type_identifier" {
            let text = ctx.text(c)?;
            return Ok(if text.is_empty() {
                None
            } else {
                Some(text.to_string())
            });
        }
    }
    Ok(None)
}

/// `_swift_collect_type_refs`: walk a type expression, appending `(name, role)`.
///
/// `role` is `"generic_arg"` inside a `type_arguments` list, `"type"` otherwise.
/// Five arms, each of which RETURNS in the Python, then a catch-all that
/// recurses only through NAMED children -- so the arms are exclusive and the
/// order between them does not matter, but the early return inside `user_type`
/// does: only the FIRST `type_identifier` child is the head, any later one
/// belongs to a nested position.
pub fn collect_type_refs(
    ctx: &Ctx,
    node: Option<Node>,
    generic: bool,
    out: &mut Vec<(String, &'static str)>,
) -> R<()> {
    let node = match node {
        Some(n) => n,
        None => return Ok(()),
    };
    let role = if generic { "generic_arg" } else { "type" };
    match node.kind() {
        "type_annotation" => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out)?;
                }
            }
        }
        "user_type" => {
            for c in children(node) {
                if c.kind() == "type_identifier" {
                    let text = ctx.text(c)?;
                    if !text.is_empty() {
                        out.push((text.to_string(), role));
                    }
                    break;
                }
            }
            for c in children(node) {
                if c.kind() != "type_arguments" {
                    continue;
                }
                for arg in children(c) {
                    if arg.is_named() {
                        collect_type_refs(ctx, Some(arg), true, out)?;
                    }
                }
            }
        }
        "type_identifier" => {
            let text = ctx.text(node)?;
            if !text.is_empty() {
                out.push((text.to_string(), role));
            }
        }
        "optional_type" | "implicitly_unwrapped_optional_type" | "array_type"
        | "dictionary_type" | "tuple_type" => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out)?;
                }
            }
        }
        _ => {
            // The catch-all is guarded on the NODE being named, not just its
            // children: an unnamed token has no type inside it to find.
            if node.is_named() {
                for c in children(node) {
                    if c.is_named() {
                        collect_type_refs(ctx, Some(c), generic, out)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// `_swift_property_type_node`: the `type_annotation` child, if the property has
/// one.
pub fn property_type_node<'tree>(node: Node<'tree>) -> Option<Node<'tree>> {
    children(node).into_iter().find(|c| c.kind() == "type_annotation")
}

/// `_swift_attribute_type_name`: the type named by `@Environment(Type.self)`.
///
/// Whitelist-gated to `Environment` on purpose (#2561). `@Query(Item.self)`
/// holds a COLLECTION of the argument type, so typing the property as the
/// element type fabricates member-call edges -- a measured false edge, not a
/// hypothetical one. The shape must be exactly
/// `[simple_identifier (uppercase), navigation_suffix ".self"]`, so the keypath
/// form (`@Environment(\.dismiss)`) and the module-dotted form
/// (`@Environment(My.Store.self)`) are skipped: a missed edge, never a wrong one.
pub fn attribute_type_name(ctx: &Ctx, property_node: Node) -> R<Option<String>> {
    for c in children(property_node) {
        if c.kind() != "modifiers" {
            continue;
        }
        for attr in children(c) {
            if attr.kind() != "attribute" {
                continue;
            }
            let head = children(attr).into_iter().find(|a| a.kind() == "user_type");
            match head {
                Some(h) if ctx.text(h)? == "Environment" => {}
                _ => continue,
            }
            let arg = children(attr)
                .into_iter()
                .find(|a| a.kind() == "navigation_expression");
            let arg = match arg {
                Some(a) => a,
                None => continue,
            };
            let named: Vec<Node> = children(arg).into_iter().filter(|a| a.is_named()).collect();
            if named.len() != 2 {
                continue;
            }
            let (ident, suffix) = (named[0], named[1]);
            if ident.kind() != "simple_identifier" || suffix.kind() != "navigation_suffix" {
                continue;
            }
            if ctx.text(suffix)? != ".self" {
                continue;
            }
            let name = ctx.text(ident)?;
            if !name.is_empty() && starts_upper(name) {
                return Ok(Some(name.to_string()));
            }
        }
    }
    Ok(None)
}

/// `_swift_factory_call`: `Factory.make()` -> `(Factory, make)`.
///
/// Depth-1 only. `A.B.make()` and `Singleton.shared.make()` stay untyped
/// because the resolver would have to guess the intermediate hop.
pub fn factory_call(ctx: &Ctx, call_node: Node) -> R<Option<(String, String)>> {
    let first = match call_node.child(0) {
        Some(f) if f.kind() == "navigation_expression" => f,
        _ => return Ok(None),
    };
    let named: Vec<Node> = children(first).into_iter().filter(|c| c.is_named()).collect();
    if named.len() != 2 {
        return Ok(None);
    }
    let (head, suffix) = (named[0], named[1]);
    if head.kind() != "simple_identifier" || suffix.kind() != "navigation_suffix" {
        return Ok(None);
    }
    let htext = ctx.text(head)?;
    if htext.is_empty() || !starts_upper(htext) {
        return Ok(None);
    }
    let mut mname: Option<&str> = None;
    for sc in children(suffix) {
        if sc.kind() == "simple_identifier" {
            // `next(...)` -- the FIRST match, not the last.
            mname = Some(ctx.text(sc)?);
            break;
        }
    }
    match mname {
        Some(m) if !m.is_empty() => Ok(Some((htext.to_string(), m.to_string()))),
        _ => Ok(None),
    }
}

/// `_swift_property_name`: the bound name of `let x` / `var x = ...`.
///
/// The two `if`s are NOT an if/elif: a `pattern` child with no
/// `simple_identifier` inside it falls through and the loop keeps going, rather
/// than returning `None`.
pub fn property_name(ctx: &Ctx, property_node: Node) -> R<Option<String>> {
    for c in children(property_node) {
        if c.kind() == "pattern" {
            for sc in children(c) {
                if sc.kind() == "simple_identifier" {
                    return Ok(Some(ctx.text(sc)?.to_string()));
                }
            }
        }
        if c.kind() == "simple_identifier" {
            return Ok(Some(ctx.text(c)?.to_string()));
        }
    }
    Ok(None)
}

/// `_swift_constructor_type`: `Foo()` -> `Foo`.
///
/// Upper-cased callees only, so a free-function call like `configure()` in an
/// initializer is not read as a constructor.
pub fn constructor_type(ctx: &Ctx, call_node: Node) -> R<Option<String>> {
    if let Some(first) = call_node.child(0) {
        if first.kind() == "simple_identifier" {
            let text = ctx.text(first)?;
            if !text.is_empty() && starts_upper(text) {
                return Ok(Some(text.to_string()));
            }
        }
    }
    Ok(None)
}

/// `_swift_receiver_name`: the depth-1 receiver of `recv.method()`.
///
/// `vm.update()` -> `vm`; `Type.staticMethod()` -> `Type`;
/// `Singleton.shared.method()` -> `Singleton` (the HEAD of the chain);
/// `self.svc.fetch()` -> `svc` (the property the call is reached through).
/// Anything deeper is `None`, keeping resolution depth-1.
pub fn receiver_name(ctx: &Ctx, recv: Option<Node>) -> R<Option<String>> {
    let recv = match recv {
        Some(n) => n,
        None => return Ok(None),
    };
    if recv.kind() == "simple_identifier" {
        return Ok(Some(ctx.text(recv)?.to_string()));
    }
    if recv.kind() == "navigation_expression" {
        let head = recv.child(0);
        if let Some(h) = head {
            if h.kind() == "simple_identifier" {
                return Ok(Some(ctx.text(h)?.to_string()));
            }
            if h.kind() == "self_expression" {
                // `self.svc.fetch()`: the receiver is the PROPERTY named in the
                // suffix, not `self`. The Python RETURNS from inside the nested
                // loop, so the FIRST suffix identifier wins and a longer chain
                // (`self.a.b.fetch()`) still yields `a`.
                for child in children(recv) {
                    if child.kind() != "navigation_suffix" {
                        continue;
                    }
                    for sc in children(child) {
                        if sc.kind() == "simple_identifier" {
                            return Ok(Some(ctx.text(sc)?.to_string()));
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

/// `_swift_local_var_types`: `var -> Type` from `let`/`var` bindings inside a
/// function body, so `x.method()` on a later line resolves (#1604).
///
/// Two initializer shapes, PRECISION over recall: a constructor call
/// (`let x = Type()`) and a static-member access (`let x = Type.shared`, the
/// singleton-cached-into-a-local idiom). A factory call has no in-file type, so
/// its binding is stashed in `factory` for corpus-side resolution instead.
///
/// **The stack order is load-bearing here.** First binding for a name wins, so
/// which node is visited first decides the answer. Python pushes children in
/// order and pops from the end, visiting the LAST child first; this reproduces
/// that exactly rather than using a natural pre-order.
pub fn local_var_types<'tree>(
    ctx: &Ctx<'_, 'tree>,
    body: Node<'tree>,
    table: &mut HashMap<String, String>,
    factory: &mut HashMap<String, (String, String)>,
) -> R<()> {
    let mut stack = vec![body];
    while let Some(n) = stack.pop() {
        // A nested function's locals are scoped away from this body.
        if n.kind() == "function_declaration" && n.id() != body.id() {
            continue;
        }
        if n.kind() == "property_declaration" {
            let mut prop_type: Option<String> = None;
            let mut factory_bind: Option<(String, String)> = None;
            for child in children(n) {
                if child.kind() == "call_expression" {
                    prop_type = constructor_type(ctx, child)?;
                    if prop_type.is_none() {
                        factory_bind = factory_call(ctx, child)?;
                    }
                    break;
                }
                if child.kind() == "navigation_expression" {
                    if let Some(head) = child.child(0) {
                        if head.kind() == "simple_identifier" {
                            let htext = ctx.text(head)?;
                            if !htext.is_empty() && starts_upper(htext) {
                                prop_type = Some(htext.to_string());
                            }
                        }
                    }
                    break;
                }
            }
            // `if name and ...` -- an EMPTY bound name is falsy in the
            // Python and must not key the table.
            let name = property_name(ctx, n)?.filter(|s| !s.is_empty());
            if let Some(name) = name {
                if let Some(pt) = prop_type {
                    if !table.contains_key(&name) {
                        table.insert(name, pt);
                    }
                } else if let Some(fb) = factory_bind {
                    if !table.contains_key(&name) && !factory.contains_key(&name) {
                        factory.insert(name, fb);
                    }
                }
            }
        }
        stack.extend(children(n));
    }
    Ok(())
}
