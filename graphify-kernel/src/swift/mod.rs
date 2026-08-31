//! Swift, on the shared engine — the largest of the four config-driven ports.
//!
//! Nine guard sites in `engine.py`, all `_is_swift` (no `ts_module` guard, so an
//! `_is_swift` inventory is complete here -- unlike C and Groovy):
//!
//! ```text
//! 3039  prescan          _swift_pre_scan -> (protocols, class-likes)
//! 3213  on_class         `extension Foo` collected for the corpus-level merge
//! 3229  on_class         inheritance_specifier -> inherits / implements
//! 4030  before_function  property_declaration: types, initializers, computed bodies
//! 4443  on_function      parameter types + return type (+ the plain-return mark)
//! 4764  extra_walk       enum_entry -> case_of, and associated-value type refs
//! 5146  call_info        simple_identifier / navigation_expression + receiver
//! 5841  before_calls     _swift_local_var_types over every function body
//! 6000  result_extra     swift_type_table (+ factory) and swift_extensions
//! ```
//!
//! # Why this one needed an engine change
//!
//! `prescan` returned a single `HashSet` because C#, its only caller, asks one
//! question: is this base name an interface? Swift asks TWO -- protocol or
//! class -- and the difference is not a default it could infer, because a Swift
//! base name that is in NEITHER set still has to be classified (see
//! [`helpers::classify_base`]). So the hook widened to a pair rather than
//! growing a second slot; C# fills `.0` and leaves `.1` empty.
//!
//! # The two accumulators
//!
//! `swift_extensions` and `swift_factory_bindings` live on `Ctx`, not on this
//! struct: `HOOKS` is a `static` and therefore immutable, so per-file state has
//! nowhere else to go. Same reason PHP's `pending_listen_edges` is a `Ctx` field.
//!
//! # Parse ceiling
//!
//! 79.8% clean over alamofire / swift-nio / vapor (903 files). Well above the
//! rate floor that kept Groovy at 16% out of `supported()`, and below the ~99%
//! the JVM languages reach -- so roughly one Swift file in five is parsed twice
//! (once natively, then again by Python after the `has_error` defer).

pub mod helpers;

use std::collections::HashSet;

use pyo3::prelude::*;
use tree_sitter::Node;

use crate::engine::{CallInfo, Ctx, EngineConfig, Handled, LangHooks, R};
use crate::js::ast::children;
use crate::js::emit::{NodeRow, Val};
use crate::Outcome;

struct Swift;

impl LangHooks for Swift {
    fn prescan<'tree>(
        &self,
        ctx: &Ctx<'_, 'tree>,
        root: Node<'tree>,
    ) -> R<(HashSet<String>, HashSet<String>)> {
        helpers::pre_scan(ctx, root)
    }

    /// `_import_swift`, plus the module ANCHOR node.
    ///
    /// `import CoreKit` names a module, not a file path, so -- unlike every
    /// file-resolving import handler -- there is no existing node for the edge
    /// to point at, and `build_from_json` would prune the edge as a dangling
    /// external reference (#1327). The Python materializes the anchor in the
    /// engine's import branch, from the `(id, label)` pairs the handler returns;
    /// here the handler does it directly, because the hook returns `()`.
    ///
    /// The anchor's key order is its own: `type` sits between `file_type` and
    /// `source_file`, where `add_node` puts it AFTER `source_location`. That is
    /// why this is a hand-built `NodeRow` rather than an `add_node` call.
    fn import_handler<'tree>(&self, ctx: &mut Ctx<'_, 'tree>, node: Node<'tree>) -> R<()> {
        for child in children(node) {
            if child.kind() != "identifier" {
                continue;
            }
            let raw = ctx.text(child)?.to_string();
            // `_make_id(raw)` -- the bare name, with no file stem, so the same
            // module imported from N files collapses onto ONE shared node.
            let tgt = ctx.mkid(&[&raw])?;
            let line = node.start_position().row + 1;
            ctx.add_import_edge(&tgt, line);
            if ctx.seen_ids.insert(tgt.clone()) {
                ctx.nodes.push(NodeRow {
                    id: tgt,
                    fields: vec![
                        ("label", Val::S(raw)),
                        ("file_type", Val::Static("code")),
                        // `file_type=code` keeps build.py validation happy;
                        // `type=module` exempts the node from id
                        // disambiguation, which is what lets the N files share
                        // one node instead of minting N of them.
                        ("type", Val::Static("module")),
                        ("source_file", Val::S(ctx.str_path.to_string())),
                        ("source_location", Val::S(format!("L{line}"))),
                    ],
                });
            }
            // Only the first `identifier` child is the module name.
            break;
        }
        Ok(())
    }

    fn on_class<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        class_nid: &str,
        class_name: &str,
        line: usize,
    ) -> R<()> {
        let t = node.kind();

        // `extension Foo` parses as `class_declaration`, indistinguishable from
        // `class Foo` by kind alone. A same-file pair collapses through
        // `seen_ids`; a cross-file one cannot, because the id carries the file
        // stem -- so it is recorded for the corpus-level merge.
        if children(node).into_iter().any(|c| c.kind() == "extension") {
            ctx.swift_extensions
                .push((class_nid.to_string(), class_name.to_string()));
        }

        let swift_kind = if t == "class_declaration" {
            helpers::declaration_keyword(node)
        } else {
            Some("protocol")
        };
        let mut seen_base = false;
        for child in children(node) {
            if child.kind() != "inheritance_specifier" {
                continue;
            }
            // The FIRST `user_type` or `type_identifier` wins and the scan
            // stops; a `user_type` also carries the generic arguments handled
            // below, a bare `type_identifier` does not.
            let mut base_name: Option<String> = None;
            let mut user_type_node: Option<Node> = None;
            for sub in children(child) {
                if sub.kind() == "user_type" {
                    user_type_node = Some(sub);
                    base_name = helpers::user_type_name(ctx, sub)?;
                    break;
                }
                if sub.kind() == "type_identifier" {
                    let text = ctx.text(sub)?;
                    // `_read_text(...) or None`: empty text is not a base.
                    base_name = if text.is_empty() {
                        None
                    } else {
                        Some(text.to_string())
                    };
                    break;
                }
            }
            let base_name = match base_name {
                Some(b) if !b.is_empty() => b,
                _ => continue,
            };

            let base_nid = ctx.ensure_parent_node(&base_name)?;
            // A protocol's own inheritance list is always other protocols, and
            // the Python calls that `inherits` -- NOT `implements`. Only a
            // class-like declaration goes through the classifier.
            let relation = if t == "protocol_declaration" {
                "inherits"
            } else {
                helpers::classify_base(
                    &base_name,
                    swift_kind,
                    !seen_base,
                    &ctx.prescan,
                    &ctx.prescan_classes,
                )
            };
            seen_base = true;
            // No `!= class_nid` guard here, unlike Scala's: the Python emits
            // this edge unconditionally.
            ctx.add_edge(class_nid, &base_nid, relation, line);

            if let Some(utn) = user_type_node {
                for arg_child in children(utn) {
                    if arg_child.kind() != "type_arguments" {
                        continue;
                    }
                    for arg in children(arg_child) {
                        if !arg.is_named() {
                            continue;
                        }
                        let mut refs = Vec::new();
                        helpers::collect_type_refs(ctx, Some(arg), true, &mut refs)?;
                        for (ref_name, _role) in refs {
                            let target = ctx.ensure_named_node(&ref_name, line)?;
                            ctx.add_edge_ctx(class_nid, &target, "references", line, "generic_arg");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// `property_declaration` inside a type: type references, the initializer,
    /// the inferred type, and -- for a computed or observed property -- a
    /// method-like node with a deferred body.
    ///
    /// Consumes the node (`Handled::Yes`), so a property is never reached by the
    /// function branch or the default recurse.
    fn before_function<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        if node.kind() != "property_declaration" {
            return Ok(Handled::No);
        }
        let parent = match parent_class_nid {
            Some(p) if !p.is_empty() => p.to_string(),
            // A file-level property falls through to the default recurse, as in
            // the Python -- the branch is guarded on `parent_class_nid`.
            _ => return Ok(Handled::No),
        };
        let line = node.start_position().row + 1;

        // 1. The declared type, if annotated. The FIRST `role == "type"` ref is
        //    the property's own type; a `generic_arg` never is.
        let mut prop_type: Option<String> = None;
        if let Some(type_anno) = helpers::property_type_node(node) {
            let mut refs = Vec::new();
            helpers::collect_type_refs(ctx, Some(type_anno), false, &mut refs)?;
            for (ref_name, role) in refs {
                let context = if role == "generic_arg" {
                    "generic_arg"
                } else {
                    "field"
                };
                let target = ctx.ensure_named_node(&ref_name, line)?;
                if target != parent {
                    ctx.add_edge_ctx(&parent, &target, "references", line, context);
                }
                if prop_type.is_none() && role == "type" {
                    prop_type = Some(ref_name);
                }
            }
        }

        // 2. The initializer. #1356 Stage 1: a call in a property initializer
        //    lives in no function body, so without `initializer_nodes` the call
        //    pass never reaches it. Stage 2a/2b: infer the type from a
        //    constructor call or a `Type.shared` static access when there is no
        //    annotation.
        //
        //    The loop does NOT break -- every child is considered, so a later
        //    call still registers its initializer even after the type is known.
        let mut pending_factory: Option<(String, String)> = None;
        for child in children(node) {
            if crate::engine::has(ctx.cfg.call_types, child.kind()) {
                ctx.initializer_nodes.push((parent.clone(), child));
                if prop_type.is_none() {
                    match helpers::constructor_type(ctx, child)? {
                        Some(ctor) => prop_type = Some(ctor),
                        // #2561: `let x = Factory.make()` has no in-file type.
                        None => pending_factory = helpers::factory_call(ctx, child)?,
                    }
                }
            } else if child.kind() == "navigation_expression" && prop_type.is_none() {
                // #1604 Stage 2b: `let x = Type.shared` -- the singleton idiom
                // cached into a property, then called on a later line. This is a
                // navigation_expression, not a call, so Stage 2a never saw it.
                if let Some(head) = child.child(0) {
                    if head.kind() == "simple_identifier" {
                        let htext = ctx.text(head)?;
                        if !htext.is_empty()
                            && htext.chars().next().is_some_and(|c| c.is_uppercase())
                        {
                            prop_type = Some(htext.to_string());
                        }
                    }
                }
            }
        }

        // 3. `@Environment(Store.self) var store` names the type only inside the
        //    attribute argument, which the direct-children scan never reaches.
        //    Last resort: annotation and constructor inference keep priority.
        if prop_type.is_none() {
            prop_type = helpers::attribute_type_name(ctx, node)?;
        }

        let prop_name = helpers::property_name(ctx, node)?.filter(|s| !s.is_empty());
        if let Some(pname) = &prop_name {
            if let Some(pt) = &prop_type {
                // Overwrites: unlike `local_var_types`, the walk's table has no
                // first-wins guard.
                ctx.type_table.insert(pname.clone(), pt.clone());
            } else if let Some(pf) = pending_factory {
                if !ctx.swift_factory_bindings.contains_key(pname) {
                    ctx.swift_factory_bindings.insert(pname.clone(), pf);
                }
            }
        }

        // 4. #2181: a computed property (`var body: some View { ... }`) or an
        //    observed one (`willSet`/`didSet`) carries a body no branch above
        //    emitted -- so the property AND every call inside it were dropped.
        //    For SwiftUI that erases the whole view layer, `body` being a
        //    computed property. A stored property has no such child, so its
        //    behaviour is unchanged.
        let comp_bodies: Vec<Node> = children(node)
            .into_iter()
            .filter(|c| matches!(c.kind(), "computed_property" | "willset_didset_block"))
            .collect();
        if !comp_bodies.is_empty() {
            if let Some(pname) = &prop_name {
                let prop_nid = ctx.mkid(&[&parent, pname])?;
                ctx.add_node(&prop_nid, &format!(".{pname}"), line);
                ctx.add_edge(&parent, &prop_nid, "method", line);
                for body_block in comp_bodies {
                    ctx.function_bodies.push((prop_nid.clone(), body_block));
                }
            }
        }
        Ok(Handled::Yes)
    }

    fn on_function<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        func_nid: &str,
        _func_name: &str,
        line: usize,
        _parent_class_nid: Option<&str>,
    ) -> R<()> {
        for p in children(node) {
            if p.kind() != "parameter" {
                continue;
            }
            let type_node = p.child_by_field_name("type");
            let mut refs = Vec::new();
            // `collect_type_refs` takes an Option and no-ops on None, matching
            // the Python's unguarded call on a possibly-missing type field.
            helpers::collect_type_refs(ctx, type_node, false, &mut refs)?;
            let mut param_type: Option<String> = None;
            for (ref_name, role) in refs {
                let context = if role == "generic_arg" {
                    "generic_arg"
                } else {
                    "parameter_type"
                };
                let target = ctx.ensure_named_node(&ref_name, line)?;
                if target != func_nid {
                    ctx.add_edge_ctx(func_nid, &target, "references", line, context);
                }
                if param_type.is_none() && role == "type" {
                    param_type = Some(ref_name);
                }
            }
            // #1356 Stage 2a: param name -> type. A flat per-file table, so a
            // later param of the same name wins -- fine for depth-1 resolution.
            if let Some(pt) = param_type {
                if let Some(name_node) = p.child_by_field_name("name") {
                    let pname = ctx.text(name_node)?;
                    if !pname.is_empty() {
                        ctx.type_table.insert(pname.to_string(), pt);
                    }
                }
            }
        }

        if let Some(return_node) = node.child_by_field_name("return_type") {
            let mut refs = Vec::new();
            helpers::collect_type_refs(ctx, Some(return_node), false, &mut refs)?;
            // #2561: a PLAIN concrete return (`-> Type`, node kind `user_type`
            // -- not `some P` / `[T]` / `T?`, which parse as opaque_type /
            // array_type / optional_type) with exactly one `type` ref is marked,
            // so the factory-receiver pass can read the method's return label
            // corpus-side. More than one `type` ref means the return is
            // composite and names no single class.
            let plain_return = return_node.kind() == "user_type"
                && refs.iter().filter(|(_, r)| *r == "type").count() == 1;
            for (ref_name, role) in refs {
                let context = if role == "generic_arg" {
                    "generic_arg"
                } else {
                    "return_type"
                };
                let target = ctx.ensure_named_node(&ref_name, line)?;
                if target != func_nid {
                    let md = if plain_return && role == "type" {
                        vec![("swift_plain_return", Val::B(true))]
                    } else {
                        Vec::new()
                    };
                    ctx.add_edge_meta(
                        func_nid,
                        &target,
                        "references",
                        line,
                        Some(context),
                        md,
                    );
                }
            }
        }
        Ok(())
    }

    /// `_swift_extra_walk`: `enum_entry` -> a `case_of` node per case name, plus
    /// `references` edges for associated-value types.
    fn extra_walk<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        if node.kind() != "enum_entry" {
            return Ok(Handled::No);
        }
        let parent = match parent_class_nid {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => return Ok(Handled::No),
        };
        let line = node.start_position().row + 1;

        // One `enum_entry` can declare several cases (`case a, b, c`), so this
        // is a loop over every `simple_identifier`, not a first-match lookup.
        for child in children(node) {
            if child.kind() != "simple_identifier" {
                continue;
            }
            let case_name = ctx.text(child)?.to_string();
            let case_nid = ctx.mkid(&[&parent, &case_name])?;
            ctx.add_node(&case_nid, &case_name, line);
            ctx.add_edge(&parent, &case_nid, "case_of", line);
        }

        // Associated values nest as `enum_type_parameters -> user_type ->
        // type_identifier`, siblings of the case-name identifier that the loop
        // above never descends into -- so `case started(Session)` used to drop
        // the Event -> Session reference entirely. The edge is from the ENUM,
        // matching the property/parameter emit style, not from the case node.
        for child in children(node) {
            if child.kind() != "enum_type_parameters" {
                continue;
            }
            for grand in children(child) {
                if !grand.is_named() {
                    continue;
                }
                let mut refs = Vec::new();
                helpers::collect_type_refs(ctx, Some(grand), false, &mut refs)?;
                for (ref_name, role) in refs {
                    let context = if role == "generic_arg" {
                        "generic_arg"
                    } else {
                        "type"
                    };
                    let target = ctx.ensure_named_node(&ref_name, line)?;
                    if target != parent {
                        ctx.add_edge_ctx(&parent, &target, "references", line, context);
                    }
                }
            }
        }
        Ok(Handled::Yes)
    }

    fn call_info<'tree>(
        &self,
        ctx: &mut Ctx<'_, 'tree>,
        node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        let mut info = CallInfo::default();
        if let Some(first) = node.child(0) {
            if first.kind() == "simple_identifier" {
                info.callee_name = Some(ctx.text(first)?.to_string());
            } else if first.kind() == "navigation_expression" {
                info.is_member_call = true;
                // No break: the Python keeps assigning, so with a chained
                // `a.b.c()` the LAST suffix identifier is the callee.
                for child in children(first) {
                    if child.kind() != "navigation_suffix" {
                        continue;
                    }
                    for sc in children(child) {
                        if sc.kind() == "simple_identifier" {
                            info.callee_name = Some(ctx.text(sc)?.to_string());
                        }
                    }
                }
                // #1356: the receiver, for the cross-file pass to type through
                // this file's `swift_type_table`. Swift fills `swift_receiver`
                // and never `member_receiver`, which is why a Swift member call
                // does NOT hit the engine's generic capitalized-receiver defer:
                // it binds in-file by bare name when it can, and the raw_call
                // carries the receiver when it cannot.
                info.swift_receiver = helpers::receiver_name(ctx, first.child(0))?;
            }
        }
        Ok(Some(info))
    }

    /// Local `let x = Type()` / `let x = Type.shared` bindings inside method
    /// bodies. Class properties are typed during the walk; body locals were not
    /// (#1604). File-scoped, not per-body, and first-binding-wins -- so a later
    /// body cannot clobber an earlier binding.
    fn before_calls<'tree>(&self, ctx: &mut Ctx<'_, 'tree>) -> R<()> {
        let bodies: Vec<Node> = ctx.function_bodies.iter().map(|(_, b)| *b).collect();
        let mut table = std::mem::take(&mut ctx.type_table);
        let mut factory = std::mem::take(&mut ctx.swift_factory_bindings);
        for body in bodies {
            helpers::local_var_types(ctx, body, &mut table, &mut factory)?;
        }
        ctx.type_table = table;
        ctx.swift_factory_bindings = factory;
        Ok(())
    }

    /// `swift_extensions` and `swift_type_table`, in the Python's key order.
    ///
    /// Swift does NOT use `EngineConfig::type_table_key`: its table is a
    /// different SHAPE (an optional `factory` sub-dict) and is emitted whenever
    /// EITHER the table or the factory bindings are non-empty, where the generic
    /// path keys on the table alone.
    fn result_extra<'tree>(
        &self,
        ctx: &Ctx<'_, 'tree>,
        _root: Node<'tree>,
    ) -> R<Vec<(&'static str, Val)>> {
        let mut out: Vec<(&'static str, Val)> = Vec::new();
        if !ctx.swift_extensions.is_empty() {
            out.push((
                "swift_extensions",
                Val::List(
                    ctx.swift_extensions
                        .iter()
                        .map(|(nid, label)| {
                            Val::Meta(vec![
                                ("nid".to_string(), Val::S(nid.clone())),
                                ("label".to_string(), Val::S(label.clone())),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        if !ctx.type_table.is_empty() || !ctx.swift_factory_bindings.is_empty() {
            // Sorted, as the generic `type_table_key` path is: Python inserts in
            // walk order and a HashMap has none. Both are pure lookup tables --
            // every consumer indexes them by key -- so the order cannot reach a
            // node or an edge.
            let mut pairs: Vec<(&String, &String)> = ctx.type_table.iter().collect();
            pairs.sort();
            let table = Val::Meta(
                pairs
                    .into_iter()
                    .map(|(k, v)| (k.clone(), Val::S(v.clone())))
                    .collect(),
            );
            let mut entries: Vec<(String, Val)> = vec![
                ("path".to_string(), Val::S(ctx.str_path.to_string())),
                ("table".to_string(), table),
            ];
            if !ctx.swift_factory_bindings.is_empty() {
                let mut fpairs: Vec<(&String, &(String, String))> =
                    ctx.swift_factory_bindings.iter().collect();
                fpairs.sort();
                entries.push((
                    "factory".to_string(),
                    Val::Meta(
                        fpairs
                            .into_iter()
                            .map(|(k, (t, m))| {
                                // LISTS, not tuples: the value has to round-trip
                                // the JSON AST cache, and a tuple would come
                                // back as a list and compare unequal.
                                (
                                    k.clone(),
                                    Val::List(vec![Val::S(t.clone()), Val::S(m.clone())]),
                                )
                            })
                            .collect(),
                    ),
                ));
            }
            out.push(("swift_type_table", Val::Meta(entries)));
        }
        Ok(out)
    }
}

static HOOKS: Swift = Swift;

pub static CONFIG: EngineConfig = EngineConfig {
    language: "swift",
    grammar: || tree_sitter_swift::LANGUAGE.into(),
    class_types: &["class_declaration", "protocol_declaration"],
    function_types: &[
        "function_declaration",
        "init_declaration",
        "deinit_declaration",
        "subscript_declaration",
    ],
    import_types: &["import_declaration"],
    call_types: &["call_expression"],
    function_boundary_types: &[
        "function_declaration",
        "init_declaration",
        "deinit_declaration",
        "subscript_declaration",
    ],
    static_prop_types: &[],
    helper_fn_names: &[],
    container_bind_methods: &[],
    event_listener_properties: &[],
    name_field: "name",
    name_fallback_child_types: &["simple_identifier", "type_identifier", "user_type"],
    body_field: "body",
    body_fallback_child_types: &[
        "class_body",
        "protocol_body",
        "function_body",
        "enum_class_body",
    ],
    // Both unset in `_SWIFT_CONFIG`: the callee comes from `call_info`.
    call_function_field: "",
    call_accessor_node_types: &["navigation_expression"],
    call_accessor_field: "",
    call_accessor_object_field: "",
    function_label_parens: true,
    resolve_function_name: None,
    sanitize_symbol_name: None,
    // Deliberately None -- see `result_extra`.
    type_table_key: None,
    hooks: &HOOKS,
};

pub fn walk_swift<'py>(
    py: Python<'py>,
    path: &str,
    source: &[u8],
    _res: &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>> {
    crate::engine::run(
        py,
        &CONFIG,
        path,
        source,
        crate::engine::no_receiver_types,
        None,
    )
}
