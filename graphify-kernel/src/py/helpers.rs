//! The `_python_*` helpers `_extract_generic` calls, transliterated.
//!
//! These are pure tree walks over already-validated UTF-8, so each is a direct
//! reading of its Python twin. The vocabulary sets below are copied verbatim from
//! `engine.py`; a name missing from one of them does not defer, it changes which
//! edges are emitted, so they are kept as literal lists in the same order as the
//! source rather than reconstructed.

use std::collections::HashSet;

use tree_sitter::Node;

use crate::js::ast::children;
use super::{Ctx, R};

/// `_PYTHON_TYPE_CONTAINERS`: builtin/typing containers that are not themselves
/// emitted as refs, though their nested arguments still count as `generic_arg`.
pub const TYPE_CONTAINERS: &[&str] = &[
    "list", "dict", "set", "tuple", "frozenset", "type",
    "List", "Dict", "Set", "Tuple", "FrozenSet", "Type",
    "Optional", "Union", "Sequence", "Iterable", "Mapping", "MutableMapping",
    "Iterator", "Callable", "Awaitable", "AsyncIterable", "AsyncIterator", "Coroutine",
    "Generator", "AsyncGenerator", "ContextManager", "AsyncContextManager",
    "Annotated", "ClassVar", "Final", "Literal", "Concatenate", "ParamSpec", "TypeVar",
    "None", "Ellipsis",
];

/// `_PYTHON_ANNOTATION_NOISE`: scalar builtins plus `unittest.mock` vocabulary.
pub const ANNOTATION_NOISE: &[&str] = &[
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "object",
    "True", "False",
    "MagicMock", "Mock", "AsyncMock", "NonCallableMock",
    "NonCallableMagicMock", "PropertyMock", "patch", "sentinel",
];

/// `_PYTHON_DECORATOR_NOISE`: builtin/stdlib decorators. Emitting edges for these
/// fabricates sourceless stubs on nearly every class-heavy file, and the
/// unique-function rewire can then collapse them onto an unrelated local
/// definition.
pub const DECORATOR_NOISE: &[&str] = &[
    "property", "staticmethod", "classmethod", "abstractmethod",
    "abstractproperty", "cached_property", "wraps", "lru_cache", "cache",
    "singledispatch", "singledispatchmethod", "total_ordering",
    "contextmanager", "asynccontextmanager", "overload", "override",
    "final", "no_type_check", "runtime_checkable", "dataclass",
];

/// `_LANGUAGE_BUILTIN_GLOBALS`, the whole frozenset -- not just its Python
/// section. `walk_calls` tests `callee_name not in _LANGUAGE_BUILTIN_GLOBALS`
/// against the shared set, so a Python file calling `Object(...)` or `Data(...)`
/// is filtered by the JS and Swift entries too. Narrowing it to the Python block
/// would emit raw_calls Python never emits.
pub const BUILTIN_GLOBALS: &[&str] = &[
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

fn noise(name: &str) -> bool {
    TYPE_CONTAINERS.contains(&name) || ANNOTATION_NOISE.contains(&name)
}

/// The `role` half of `_python_collect_type_refs`' `(name, role)` pairs. Only two
/// values exist and both map to a fixed context string at every call site, so
/// they are a bool rather than a string that could be misspelled.
pub type Refs = Vec<(String, bool)>; // (name, is_generic_arg)

/// `_python_collect_type_refs(node, source, generic, out)`.
pub fn collect_type_refs(ctx: &Ctx, node: Option<Node>, generic: bool, out: &mut Refs) -> R<()> {
    let Some(node) = node else { return Ok(()) };
    match node.kind() {
        "type" => {
            for c in children(node) {
                if c.is_named() {
                    collect_type_refs(ctx, Some(c), generic, out)?;
                }
            }
        }
        "identifier" => {
            let name = ctx.text(node)?;
            if !name.is_empty() && !noise(name) {
                out.push((name.to_string(), generic));
            }
        }
        "attribute" => {
            // `_read_text(node, source).rsplit(".", 1)[-1]`: everything after the
            // LAST dot, or the whole string when there is none.
            let whole = ctx.text(node)?;
            let tail = whole.rsplit('.').next().unwrap_or(whole);
            if !tail.is_empty() && !noise(tail) {
                out.push((tail.to_string(), generic));
            }
        }
        "generic_type" => {
            for c in children(node) {
                if c.kind() == "identifier" {
                    let container = ctx.text(c)?;
                    if !container.is_empty() && !noise(container) {
                        out.push((container.to_string(), generic));
                    }
                } else if c.kind() == "type_parameter" {
                    for sub in children(c) {
                        if sub.is_named() {
                            collect_type_refs(ctx, Some(sub), true, out)?;
                        }
                    }
                }
            }
        }
        "subscript" => {
            let value = node.child_by_field_name("value");
            collect_type_refs(ctx, value, generic, out)?;
            let value_id = value.map(|v| v.id());
            for c in children(node) {
                if Some(c.id()) == value_id || !c.is_named() {
                    continue;
                }
                collect_type_refs(ctx, Some(c), true, out)?;
            }
        }
        _ => {
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

/// `_python_collect_param_refs`: type refs from each typed parameter.
pub fn collect_param_refs(ctx: &Ctx, params_node: Option<Node>) -> R<Refs> {
    let mut out: Refs = Vec::new();
    let Some(params_node) = params_node else { return Ok(out) };
    for child in children(params_node) {
        if matches!(child.kind(), "typed_parameter" | "typed_default_parameter") {
            let type_node = child.child_by_field_name("type");
            collect_type_refs(ctx, type_node, false, &mut out)?;
        }
    }
    Ok(out)
}

/// `_python_param_names`: every local name a `parameters` node binds, including
/// `*args` / `**kwargs` and typed or default forms.
pub fn param_names(ctx: &Ctx, params_node: Option<Node>, out: &mut HashSet<String>) -> R<()> {
    let Some(params_node) = params_node else { return Ok(()) };
    for child in children(params_node) {
        match child.kind() {
            "identifier" => {
                out.insert(ctx.text(child)?.to_string());
            }
            "typed_parameter" | "default_parameter" | "typed_default_parameter"
            | "list_splat_pattern" | "dictionary_splat_pattern" => {
                // The bound name is the `name` field, or failing that the first
                // identifier child (the rest is type / default).
                let name_n = child
                    .child_by_field_name("name")
                    .or_else(|| children(child).into_iter().find(|c| c.kind() == "identifier"));
                if let Some(n) = name_n {
                    out.insert(ctx.text(n)?.to_string());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// `_python_collect_assignment_targets`: identifiers bound as `pattern` targets,
/// recursing through the three unpacking forms so `a, b = ...` binds both.
pub fn collect_assignment_targets(
    ctx: &Ctx,
    node: Option<Node>,
    out: &mut HashSet<String>,
) -> R<()> {
    let Some(node) = node else { return Ok(()) };
    match node.kind() {
        "identifier" => {
            out.insert(ctx.text(node)?.to_string());
        }
        "pattern_list" | "tuple_pattern" | "list_pattern" => {
            for c in children(node) {
                collect_assignment_targets(ctx, Some(c), out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// The binding scan shared by `_python_local_bound_names` and
/// `_python_module_bound_names`.
///
/// The two differ in exactly one way: the function version also handles
/// `with_statement` aliases. Kept as one function with a flag rather than two
/// near-copies, because the shared part is where a divergence would hide -- both
/// stop at `function_definition` / `class_definition` / `lambda`, and both
/// recurse into every other child AFTER handling it, so a `for` inside an `if`
/// inside a `try` still contributes.
fn scan_bindings(
    ctx: &Ctx,
    n: Node,
    out: &mut HashSet<String>,
    with_statements: bool,
) -> R<()> {
    for child in children(n) {
        match child.kind() {
            "function_definition" | "class_definition" | "lambda" => continue,
            "assignment" => {
                collect_assignment_targets(ctx, child.child_by_field_name("left"), out)?;
            }
            "for_statement" | "for_in_clause" => {
                collect_assignment_targets(ctx, child.child_by_field_name("left"), out)?;
            }
            "with_statement" if with_statements => {
                for item in children(child) {
                    if item.kind() != "with_clause" {
                        continue;
                    }
                    for wi in children(item) {
                        if wi.kind() == "with_item" {
                            collect_assignment_targets(
                                ctx,
                                wi.child_by_field_name("alias"),
                                out,
                            )?;
                        }
                    }
                }
            }
            "named_expression" => {
                // walrus `:=`
                collect_assignment_targets(ctx, child.child_by_field_name("name"), out)?;
            }
            _ => {}
        }
        scan_bindings(ctx, child, out, with_statements)?;
    }
    Ok(())
}

/// `_python_local_bound_names`: parameters plus assignment / `for` / `with ... as`
/// / comprehension targets. Nested function and class subtrees are NOT descended
/// into -- their bindings belong to another scope.
pub fn local_bound_names(ctx: &Ctx, func_def_node: Node) -> R<HashSet<String>> {
    let mut bound = HashSet::new();
    param_names(ctx, func_def_node.child_by_field_name("parameters"), &mut bound)?;
    if let Some(body) = func_def_node.child_by_field_name("body") {
        scan_bindings(ctx, body, &mut bound, true)?;
    }
    Ok(bound)
}

/// `_python_module_bound_names`: names rebound by assignment at MODULE scope.
/// No `with_statement` arm -- the Python twin does not have one.
pub fn module_bound_names(ctx: &Ctx, root: Node) -> R<HashSet<String>> {
    let mut bound = HashSet::new();
    scan_bindings(ctx, root, &mut bound, false)?;
    Ok(bound)
}

/// `_python_decorator_name`: the head symbol of a `decorator` node.
/// `@traced` -> the identifier; `@retry(times=3)` -> the call's `function`;
/// `@app.route("/")` -> the attribute (the symbol, not the module alias).
pub fn decorator_name<'a>(ctx: &Ctx<'a, '_>, deco_node: Node) -> R<Option<String>> {
    for child in children(deco_node) {
        if !child.is_named() {
            continue;
        }
        let mut target = child;
        if target.kind() == "call" {
            target = target.child_by_field_name("function").unwrap_or(target);
        }
        if target.kind() == "attribute" {
            return Ok(match target.child_by_field_name("attribute") {
                Some(attr) => Some(ctx.text(attr)?.to_string()),
                None => None,
            });
        }
        if target.kind() == "identifier" {
            return Ok(Some(ctx.text(target)?.to_string()));
        }
        // Python returns after the FIRST named child regardless of its kind.
        return Ok(None);
    }
    Ok(None)
}
