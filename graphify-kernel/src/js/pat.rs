//! The JS/TS shadow-set and pattern helpers, ported 1:1 from `engine.py`.
//!
//! These decide whether an identifier used as a value is a *reference to a
//! module-level callable* (which mints an `indirect_call` edge) or a *local
//! binding that merely shares its name* (which must mint nothing). Every one of
//! the singular/plural and destructuring cases below is in the Python because it
//! was once missing and fabricated a wrong edge, so the port keeps the same
//! shape -- including the parts that look redundant.

use std::collections::HashSet;
use tree_sitter::Node;

use super::ast::{named_children, text};

/// `_JS_SCOPE_BOUNDARY`: an inner scope whose bindings are not the outer scope's.
pub fn is_scope_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration"
            | "function_expression"
            | "function"
            | "arrow_function"
            | "method_definition"
            | "class_declaration"
            | "class"
            | "generator_function"
            | "generator_function_declaration"
    )
}

/// `_JS_FUNCTION_VALUE_TYPES`.
pub fn is_function_value(kind: &str) -> bool {
    matches!(
        kind,
        "arrow_function" | "function_expression" | "function" | "generator_function"
    )
}

/// `_JS_DESCEND_TYPES` = the two closure kinds plus the nested NAMED
/// declarations (#2575).
///
/// `generator_function` -- an anonymous `function*` EXPRESSION -- belongs here and
/// not only in [`is_function_value`]: it is a `walk_calls` boundary, so leaving it
/// out silently drops every call in its body. That is what it did, on
/// `new Response(async function* yo() { await Bun.sleep(30) })`: five of seven
/// raw_calls in the file vanished with no error, and the file still produced
/// identical nodes and edges.
pub fn is_descend_type(kind: &str) -> bool {
    matches!(
        kind,
        "arrow_function"
            | "function_expression"
            | "function_declaration"
            | "generator_function_declaration"
            | "generator_function"
    )
}

/// `_js_collect_pattern_idents`.
pub fn collect_pattern_idents(node: Node, src: &[u8], bound: &mut HashSet<String>) {
    match node.kind() {
        "identifier" | "shorthand_property_identifier_pattern" => {
            bound.insert(text(node, src).to_string());
        }
        // `(h: Handler)` -- Handler is a type, not a bound name.
        "type_annotation" => {}
        // `x = default` -- only x is bound.
        "assignment_pattern" => {
            if let Some(left) = node.child_by_field_name("left") {
                collect_pattern_idents(left, src, bound);
            }
        }
        // `{ a: localName }` -- localName is bound.
        "pair_pattern" => {
            if let Some(val) = node.child_by_field_name("value") {
                collect_pattern_idents(val, src, bound);
            }
        }
        _ => {
            for c in named_children(node) {
                collect_pattern_idents(c, src, bound);
            }
        }
    }
}

/// `_js_local_bound_names`: parameters plus `const`/`let`/`var` targets, not
/// descending into nested scopes.
pub fn local_bound_names(func_node: Node, src: &[u8]) -> HashSet<String> {
    let mut bound = HashSet::new();
    if let Some(params) = func_node.child_by_field_name("parameters") {
        collect_pattern_idents(params, src, &mut bound);
    }
    // `x => f(x)` exposes its parameter as `parameter` (singular): there is no
    // `parameters` list node at all, so the branch above sees nothing.
    if let Some(solo) = func_node.child_by_field_name("parameter") {
        collect_pattern_idents(solo, src, &mut bound);
    }
    if let Some(body) = func_node.child_by_field_name("body") {
        bound_walk(body, src, &mut bound);
    }
    bound
}

fn bound_walk(n: Node, src: &[u8], bound: &mut HashSet<String>) {
    let mut cur = n.walk();
    for c in n.children(&mut cur) {
        if is_scope_boundary(c.kind()) {
            continue;
        }
        if c.kind() == "variable_declarator" {
            if let Some(name) = c.child_by_field_name("name") {
                collect_pattern_idents(name, src, bound);
            }
        } else if c.kind() == "for_in_statement" {
            // `for (const entry of xs)`: the binding is the `left` pattern and is
            // NOT wrapped in a variable_declarator, so the branch above misses it.
            if let Some(left) = c.child_by_field_name("left") {
                collect_pattern_idents(left, src, bound);
            }
        }
        bound_walk(c, src, bound);
    }
}

/// `_js_module_bound_names`: module-scope names rebound to NON-function data.
/// A declarator whose value is itself a function is excluded -- that name IS a
/// callable dispatch tables should resolve to, not a data shadow.
pub fn module_bound_names(root: Node, src: &[u8]) -> HashSet<String> {
    let mut bound = HashSet::new();
    fn walk(n: Node, src: &[u8], bound: &mut HashSet<String>) {
        let mut cur = n.walk();
        for c in n.children(&mut cur) {
            if is_scope_boundary(c.kind()) {
                continue;
            }
            if c.kind() == "variable_declarator" {
                let value = c.child_by_field_name("value");
                if value.map_or(true, |v| !is_function_value(v.kind())) {
                    if let Some(name) = c.child_by_field_name("name") {
                        collect_pattern_idents(name, src, bound);
                    }
                }
            }
            walk(c, src, bound);
        }
    }
    walk(root, src, &mut bound);
    bound
}

/// `_js_dispatch_value_idents`: identifier VALUES of an object/array literal.
/// Keys and inline methods are not references.
pub fn dispatch_value_idents<'tree>(coll: Node<'tree>) -> Vec<Node<'tree>> {
    let mut out = Vec::new();
    let mut cur = coll.walk();
    if coll.kind() == "object" {
        for c in coll.children(&mut cur) {
            if c.kind() == "pair" {
                if let Some(val) = c.child_by_field_name("value") {
                    if val.kind() == "identifier" {
                        out.push(val);
                    }
                }
            } else if c.kind() == "shorthand_property_identifier" {
                out.push(c);
            }
        }
    } else {
        for el in coll.children(&mut cur) {
            if el.kind() == "identifier" {
                out.push(el);
            }
        }
    }
    out
}

/// `_js_topmost_closures`: the outermost closures under `node`, not descending
/// into one that was found -- its nested closures belong to it.
pub fn topmost_closures<'tree>(node: Node<'tree>, out: &mut Vec<Node<'tree>>) {
    let mut cur = node.walk();
    for c in node.children(&mut cur) {
        if is_function_value(c.kind()) {
            out.push(c);
        } else {
            topmost_closures(c, out);
        }
    }
}

/// What an `assignment_expression` LHS defines, when its RHS is a function.
/// Mirrors `_js_member_assignment_target`.
pub enum AssignTarget {
    This(String),
    Exports(String),
    Prototype { owner: String, member: String },
    Object { owner: String, member: String },
}

pub fn member_assignment_target(left: Option<Node>, src: &[u8]) -> Option<AssignTarget> {
    let left = left?;
    if left.kind() != "member_expression" {
        return None;
    }
    let prop = left.child_by_field_name("property")?;
    let member_name = text(prop, src).to_string();
    if member_name.is_empty() {
        return None;
    }
    let obj = left.child_by_field_name("object")?;
    match obj.kind() {
        "this" => Some(AssignTarget::This(member_name)),
        "identifier" => {
            let name = text(obj, src);
            if name == "exports" {
                Some(AssignTarget::Exports(member_name))
            } else {
                Some(AssignTarget::Object {
                    owner: name.to_string(),
                    member: member_name,
                })
            }
        }
        "member_expression" => {
            // `module.exports.X` or `Foo.prototype.X`
            let inner_obj = obj.child_by_field_name("object")?;
            let inner_prop = obj.child_by_field_name("property")?;
            let inner_prop_name = text(inner_prop, src);
            if inner_obj.kind() == "identifier" {
                let inner_obj_name = text(inner_obj, src);
                if inner_obj_name == "module" && inner_prop_name == "exports" {
                    return Some(AssignTarget::Exports(member_name));
                }
                if inner_prop_name == "prototype" {
                    return Some(AssignTarget::Prototype {
                        owner: inner_obj_name.to_string(),
                        member: member_name,
                    });
                }
            }
            None
        }
        _ => None,
    }
}

/// `_LANGUAGE_BUILTIN_GLOBALS`, the JS/TS-reachable part.
///
/// The Python set is shared across every language, so it also lists Python and
/// Swift builtins. Those names are only ever *tested* against a JS/TS callee
/// here, and a JS file that calls a function named `sorted` or `Data` really is
/// suppressed by the Python set too -- so the whole set is reproduced rather
/// than the JS slice, or the two would disagree on exactly those names.
pub fn is_builtin_global(name: &str) -> bool {
    const BUILTINS: &[&str] = &[
        "String", "Number", "Boolean", "Object", "Array", "Symbol", "BigInt",
        "Date", "RegExp", "Error", "TypeError", "RangeError", "SyntaxError",
        "ReferenceError", "EvalError", "URIError",
        "Promise", "Map", "Set", "WeakMap", "WeakSet", "JSON", "Math",
        "Reflect", "Proxy", "Intl",
        "parseInt", "parseFloat", "isNaN", "isFinite",
        "encodeURIComponent", "decodeURIComponent", "encodeURI", "decodeURI",
        "URL", "URLSearchParams", "FormData", "Blob", "File",
        "Headers", "Request", "Response", "AbortController", "AbortSignal",
        "TextEncoder", "TextDecoder", "console",
        "str", "int", "float", "bool", "list", "dict", "set", "tuple", "bytes",
        "len", "range", "enumerate", "zip", "map", "filter", "sum", "min", "max",
        "print", "open", "isinstance", "type", "super", "sorted", "reversed",
        "any", "all", "abs", "round", "next", "iter", "hash", "id", "repr",
        "callable", "getattr", "setattr", "hasattr", "delattr", "vars", "dir",
        "Int", "Int8", "Int16", "Int32", "Int64",
        "UInt", "UInt8", "UInt16", "UInt32", "UInt64",
        "Double", "Float", "Bool", "Character",
        "Sendable", "Codable", "Decodable", "Encodable", "Equatable", "Hashable",
        "Identifiable", "Comparable", "CaseIterable", "RawRepresentable",
        "CustomStringConvertible", "CustomDebugStringConvertible", "AnyObject",
        "LocalizedError",
        "Data", "UUID", "Decimal", "Calendar", "Locale", "TimeZone", "Bundle",
        "IndexPath", "IndexSet", "NotificationCenter", "UserDefaults",
        "FileManager", "URLSession", "URLRequest", "URLComponents",
        "JSONDecoder", "JSONEncoder", "DateFormatter", "NumberFormatter",
        "ISO8601DateFormatter",
        "NSObject", "NSString", "NSError", "NSLock", "NSAttributedString",
        "DispatchQueue", "DispatchGroup", "OperationQueue", "RunLoop",
        "View", "Color", "Font",
    ];
    BUILTINS.contains(&name)
}
