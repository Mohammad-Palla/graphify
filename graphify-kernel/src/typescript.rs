//! TypeScript walker.
//!
//! Handles the pure-AST part of `_extract_generic` for TypeScript and defers
//! everything else. The deferral list is deliberately long and explicit: a
//! construct this walker has not been taught is a *deferral*, never a guess,
//! because the failure mode of guessing is a silently missing node or edge.
//!
//! # Why imports defer
//!
//! `_import_js` resolves a module specifier through `_resolve_js_import_target`,
//! which probes the filesystem: extension candidates, index files, `is_file()`
//! checks, tsconfig `paths` aliases and workspace package manifests. That work is
//! I/O-bound -- filesystem calls were ~5% of phase 2, against ~88% for the walk --
//! so moving it to Rust would buy almost nothing while reproducing a large,
//! heavily special-cased surface. Files containing imports therefore defer for
//! now. That is most TS files (86.3% of a 900-file Bun sample contain an
//! `import_statement`), which is exactly why the honest next step is to measure
//! the native rate on the remainder rather than assume a hybrid boundary is worth
//! building.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::{Node, Parser};

use crate::ids::{file_stem, make_id_ascii};

/// Top-level constructs that carry no extractable symbol and can be skipped
/// outright. Anything not listed here and not handled below defers the file.
/// Constructs that mint no symbol THEMSELVES. They are still recursed into:
/// Python's `walk` ends in a default `for child in node.children: walk(child)`,
/// so a declaration nested inside an `if`, a `try` or a bare block is found and
/// emitted exactly as a top-level one would be.
///
/// An earlier version SKIPPED these outright, on the strength of a probe showing
/// `if (a) { helper(); }` emits nothing. The probe was the wrong shape: it tested
/// a statement containing a CALL, not one containing a DECLARATION. The parity
/// harness caught it as five divergences -- one file lost 45 raw_calls and every
/// node but the file node, because its functions live inside a conditional.
const INERT_BUT_RECURSED: &[&str] = &[
    "comment",
    "hash_bang_line",
    ";",
    "empty_statement",
    "expression_statement",
    "if_statement",
    "for_statement",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "try_statement",
    "switch_statement",
    "throw_statement",
    "statement_block",
    "debugger_statement",
    "labeled_statement",
    "return_statement",
    "lexical_declaration_body",
];

/// Constructs that force a deferral because resolving them needs Python.
const DEFER_TOP_LEVEL: &[&str] = &[
    "import_statement",      // needs _resolve_js_import_target (filesystem)
    "export_statement",      // may be a re-export, same resolution path
    "ambient_declaration",
    "internal_module",
    "module",
];

struct Emit {
    nodes: Vec<NodeRow>,
    edges: Vec<EdgeRow>,
    raw_calls: Vec<CallRow>,
}

struct NodeRow {
    id: String,
    label: String,
    line: usize,
    callable: bool,
    callable_class: bool,
}

struct EdgeRow {
    source: String,
    target: String,
    relation: &'static str,
    context: Option<&'static str>,
    line: usize,
}

struct CallRow {
    caller_nid: String,
    callee: String,
    is_member_call: bool,
    receiver: Option<String>,
    line: usize,
}

fn text<'a>(node: Node, src: &'a [u8]) -> &'a str {
    std::str::from_utf8(&src[node.byte_range()]).unwrap_or("")
}

fn line_of(node: Node) -> usize {
    node.start_position().row + 1
}

/// Name of a declaration via its `name` field.
fn named<'a>(node: Node, src: &'a [u8]) -> Option<&'a str> {
    node.child_by_field_name("name").map(|n| text(n, src))
}

impl Emit {
    fn new() -> Self {
        Emit { nodes: Vec::new(), edges: Vec::new(), raw_calls: Vec::new() }
    }

    /// Collect calls inside a callable body, attributed to `caller_nid`.
    /// Recursion stops at nested function boundaries, which get their own caller.
    fn walk_calls(&mut self, node: Node, src: &[u8], caller_nid: &str) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                // Nested callables are their own scope; the Python walker stops
                // here too, so a call inside them must not be attributed upward.
                "function_declaration" | "function_expression" | "arrow_function"
                | "method_definition" | "class_declaration" | "generator_function"
                | "generator_function_declaration" => continue,
                "call_expression" => {
                    if !self.record_call(child, src, caller_nid) {
                        return false;
                    }
                    if let Some(args) = child.child_by_field_name("arguments") {
                        if !self.walk_calls(args, src, caller_nid) {
                            return false;
                        }
                    }
                }
                _ => {
                    if !self.walk_calls(child, src, caller_nid) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn record_call(&mut self, call: Node, src: &[u8], caller_nid: &str) -> bool {
        let Some(func) = call.child_by_field_name("function") else {
            return true;
        };
        match func.kind() {
            "identifier" => {
                self.raw_calls.push(CallRow {
                    caller_nid: caller_nid.to_string(),
                    callee: text(func, src).to_string(),
                    is_member_call: false,
                    receiver: None,
                    line: line_of(call),
                });
                true
            }
            "member_expression" => {
                let prop = func.child_by_field_name("property");
                let obj = func.child_by_field_name("object");
                match (prop, obj) {
                    (Some(p), Some(o)) if o.kind() == "identifier" || o.kind() == "this" => {
                        self.raw_calls.push(CallRow {
                            caller_nid: caller_nid.to_string(),
                            callee: text(p, src).to_string(),
                            is_member_call: true,
                            receiver: Some(text(o, src).to_string()),
                            line: line_of(call),
                        });
                        true
                    }
                    // Chained / computed receivers have their own Python rules.
                    _ => false,
                }
            }
            // `f()()`, `(await x)()`, dynamic `import()`: all have dedicated
            // Python handling. Defer rather than approximate.
            _ => false,
        }
    }
}

/// Returns `Ok(None)` to defer the file.
pub fn walk<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(stem) = file_stem(path) else {
        return Ok(None); // path needs pathlib normalization
    };
    let Some(file_nid) = make_id_ascii(&[path]) else {
        return Ok(None); // non-ASCII path
    };

    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .is_err()
    {
        return Ok(None);
    }
    let Some(tree) = parser.parse(source, None) else {
        return Ok(None);
    };
    let root = tree.root_node();
    if root.has_error() {
        return Ok(None); // Python's error recovery is authoritative
    }

    let mut emit = Emit::new();
    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    emit.nodes.push(NodeRow {
        id: file_nid.clone(),
        label: file_label,
        line: 1,
        callable: false,
        callable_class: false,
    });

    if !visit(&mut emit, root, source, &stem, &file_nid) {
        return Ok(None);
    }

    Ok(Some(to_py(py, path, emit)?))
}


/// Mirrors Python's `walk`: handle the declaration kinds, recurse into everything
/// else, defer on anything whose Python behaviour this walker does not reproduce.
fn visit(
    emit: &mut Emit,
    node: Node,
    src: &[u8],
    stem: &str,
    file_nid: &str,
) -> bool {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if DEFER_TOP_LEVEL.contains(&kind) {
            return false;
        }
        match kind {
            "function_declaration" | "generator_function_declaration" => {
                if !handle_function(emit, child, src, stem, file_nid) {
                    return false;
                }
            }
            "class_declaration" => {
                if !handle_class(emit, child, src, stem, file_nid) {
                    return false;
                }
            }
            "lexical_declaration" | "variable_declaration" => {
                if !handle_declaration(emit, child, src, stem, file_nid) {
                    return false;
                }
            }
            _ if INERT_BUT_RECURSED.contains(&kind) => {
                if !visit(emit, child, src, stem, file_nid) {
                    return false;
                }
            }
            // Anything else -- type aliases, interfaces, enums, namespaces,
            // decorators -- emits symbols this walker has not been taught.
            _ => return false,
        }
    }
    true
}

fn handle_function(
    emit: &mut Emit,
    node: Node,
    src: &[u8],
    stem: &str,
    file_nid: &str,
) -> bool {
    let Some(name) = named(node, src) else { return false };
    let Some(nid) = make_id_ascii(&[stem, name]) else { return false };
    let line = line_of(node);
    emit.nodes.push(NodeRow {
        id: nid.clone(),
        label: format!("{name}()"),
        line,
        callable: true,
        callable_class: false,
    });
    emit.edges.push(EdgeRow {
        source: file_nid.to_string(),
        target: nid.clone(),
        relation: "contains",
        context: None,
        line,
    });
    match node.child_by_field_name("body") {
        Some(body) => emit.walk_calls(body, src, &nid),
        None => true,
    }
}

fn handle_class(
    emit: &mut Emit,
    node: Node,
    src: &[u8],
    stem: &str,
    file_nid: &str,
) -> bool {
    let Some(name) = named(node, src) else { return false };
    let Some(nid) = make_id_ascii(&[stem, name]) else { return false };
    let line = line_of(node);
    emit.nodes.push(NodeRow {
        id: nid.clone(),
        label: name.to_string(),
        line,
        callable: true,
        callable_class: true,
    });
    emit.edges.push(EdgeRow {
        source: file_nid.to_string(),
        target: nid.clone(),
        relation: "contains",
        context: None,
        line,
    });

    let Some(body) = node.child_by_field_name("body") else { return true };
    let mut bc = body.walk();
    for member in body.children(&mut bc) {
        match member.kind() {
            "method_definition" => {
                let Some(mname) = named(member, src) else { return false };
                let Some(mnid) = make_id_ascii(&[stem, name, mname]) else { return false };
                let mline = line_of(member);
                emit.nodes.push(NodeRow {
                    id: mnid.clone(),
                    label: format!(".{mname}()"),
                    line: mline,
                    callable: true,
                    callable_class: false,
                });
                emit.edges.push(EdgeRow {
                    source: nid.clone(),
                    target: mnid.clone(),
                    relation: "method",
                    context: None,
                    line: mline,
                });
                if let Some(mbody) = member.child_by_field_name("body") {
                    if !emit.walk_calls(mbody, src, &mnid) {
                        return false;
                    }
                }
            }
            "comment" | "{" | "}" | ";" => {}
            // Fields, index signatures, decorators, static blocks, accessors:
            // all emit references/type edges in Python.
            _ => return false,
        }
    }
    true
}

fn handle_declaration(
    emit: &mut Emit,
    node: Node,
    src: &[u8],
    stem: &str,
    file_nid: &str,
) -> bool {
    let mut c = node.walk();
    for decl in node.children(&mut c) {
        if decl.kind() != "variable_declarator" {
            continue;
        }
        let Some(nm) = decl.child_by_field_name("name") else { return false };
        if nm.kind() != "identifier" {
            return false; // destructuring has its own binding rules
        }
        let Some(value) = decl.child_by_field_name("value") else {
            // `let x;` -- no initializer, nothing emitted.
            continue;
        };
        match value.kind() {
            // Scalar initializers emit nothing (`const x = 1` produces no node).
            // Object and array literals DO mint a node, and `new X()` feeds the
            // ts_type_table, so both stay deferred below.
            "number" | "string" | "template_string" | "true" | "false" | "null"
            | "regex" => continue,
            "arrow_function" | "function_expression" => {
                let name = text(nm, src);
                let Some(nid) = make_id_ascii(&[stem, name]) else { return false };
                let line = line_of(node);
                emit.nodes.push(NodeRow {
                    id: nid.clone(),
                    label: format!("{name}()"),
                    line,
                    callable: true,
                    callable_class: false,
                });
                emit.edges.push(EdgeRow {
                    source: file_nid.to_string(),
                    target: nid.clone(),
                    relation: "contains",
                    context: None,
                    line,
                });
                if let Some(body) = value.child_by_field_name("body") {
                    if !emit.walk_calls(body, src, &nid) {
                        return false;
                    }
                }
            }
            _ => return false,
        }
    }
    true
}

fn to_py<'py>(py: Python<'py>, path: &str, emit: Emit) -> PyResult<Bound<'py, PyDict>> {
    let nodes = PyList::empty(py);
    for n in &emit.nodes {
        let d = PyDict::new(py);
        d.set_item("id", &n.id)?;
        d.set_item("label", &n.label)?;
        d.set_item("file_type", "code")?;
        d.set_item("source_file", path)?;
        d.set_item("source_location", format!("L{}", n.line))?;
        if n.callable {
            d.set_item("_callable", true)?;
        }
        if n.callable_class {
            d.set_item("_callable_class", true)?;
        }
        nodes.append(d)?;
    }
    let edges = PyList::empty(py);
    for e in &emit.edges {
        let d = PyDict::new(py);
        d.set_item("source", &e.source)?;
        d.set_item("target", &e.target)?;
        d.set_item("relation", e.relation)?;
        if let Some(ctx) = e.context {
            d.set_item("context", ctx)?;
        }
        d.set_item("confidence", "EXTRACTED")?;
        d.set_item("source_file", path)?;
        d.set_item("source_location", format!("L{}", e.line))?;
        d.set_item("weight", 1.0)?;
        edges.append(d)?;
    }
    let raw_calls = PyList::empty(py);
    for c in &emit.raw_calls {
        let d = PyDict::new(py);
        d.set_item("caller_nid", &c.caller_nid)?;
        d.set_item("callee", &c.callee)?;
        d.set_item("is_member_call", c.is_member_call)?;
        d.set_item("source_file", path)?;
        d.set_item("source_location", format!("L{}", c.line))?;
        match &c.receiver {
            Some(r) => d.set_item("receiver", r)?,
            None => d.set_item("receiver", py.None())?,
        }
        raw_calls.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("nodes", nodes)?;
    out.set_item("edges", edges)?;
    out.set_item("raw_calls", raw_calls)?;
    Ok(out)
}
