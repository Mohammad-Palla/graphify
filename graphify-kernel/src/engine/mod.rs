//! A config-driven port of `engine.py::_extract_generic`.
//!
//! # Why this exists
//!
//! `_extract_generic` is ONE 3,144-line function driving fifteen
//! `LanguageConfig`s. The `js/` and `py/` walkers each re-derive the slice of it
//! their language reaches, which was right for two languages and is not right
//! for ten: re-deriving the same skeleton ten times is ten chances for it to
//! drift from the Python in a way no single language's parity run would reveal.
//!
//! So this module mirrors the Python's own structure instead. The skeleton is
//! shared and its control flow matches `walk` / `walk_calls` branch for branch;
//! everything a language does differently is either DATA on [`EngineConfig`] --
//! the same fields `LanguageConfig` carries -- or one of the nine [`LangHooks`]
//! methods, which sit at exactly the nine points where the Python has an
//! `_is_<lang>` guard.
//!
//! `js/` and `py/` deliberately stay as they are. They are gated at DIVERGENT 0
//! over 15,000+ files, and rewriting a walker that already passes buys nothing
//! and risks a regression the corpora might not catch.
//!
//! # The hook points, and why exactly these
//!
//! Taken from an inventory of every `_is_<lang>` guard in `walk` and
//! `walk_calls`, not invented:
//!
//! ```text
//! walk
//!   import branch          -> config.import_handler
//!   class, pre-add_node    -> class_metadata      (C# alone: is_nested_type/is_partial)
//!   class, post-edges      -> on_class            (10 languages: inheritance, annotations)
//!   between class & fn     -> before_function     (7: field/property declarations)
//!   function, post-edges   -> on_function         (all: params, return type, annotations)
//!   trailing               -> extra_walk          (7: enum_constant, companion_object, ...)
//! walk_calls
//!   call branch            -> call_info           (8: callee/receiver extraction)
//!   the tgt_nid decision   -> defers              (3: python, csharp, java)
//!   raw_call construction  -> raw_call_extra      (4: lang tag, receiver_type)
//! ```
//!
//! Every method has a no-op default, so a language implements only what its
//! Python guards actually do -- and a language with no guards at all (Lua,
//! Groovy) needs no impl beyond the config.

use std::collections::{HashMap, HashSet};

use tree_sitter::{Language, Node};

use crate::ids::{make_id_ascii, normalize_id_ascii};
use crate::js::ast::text_checked;
use crate::js::emit::{EdgeRow, NodeRow, RawCall, Val};

pub mod calls;
pub mod meta;
pub mod walk;

pub type R<T> = Result<T, &'static str>;

/// See `js::MAX_DEPTH`. A Rust stack overflow is a SIGSEGV that takes the whole
/// pool worker down, where Python raises a catchable `RecursionError`.
pub const MAX_DEPTH: u32 = 1000;

/// Membership test for a config set.
///
/// A LINEAR scan over a `&'static [&'static str]`, never a binary search. These
/// sets have at most a handful of entries so the scan is free, and the
/// alternative has already cost this project real time: `BUILTIN_GLOBALS` is
/// written grouped by language rather than sorted, and `binary_search` over it
/// silently reported most of the set as absent -- 64 of 264 gson files
/// DIVERGENT, on names as ordinary as `set` and `next`.
#[inline]
pub fn has(set: &'static [&'static str], kind: &str) -> bool {
    set.contains(&kind)
}

/// The data half of `LanguageConfig`, field for field.
///
/// Every field here is the Python attribute of the same name. Fields the Python
/// carries but no ported language uses yet are present anyway, so that adding
/// such a language is a config change rather than a skeleton change.
pub struct EngineConfig {
    /// The kernel's language key, e.g. `"java"`. Matches `languages::supported`.
    pub language: &'static str,
    /// The grammar, as the linked crate provides it.
    pub grammar: fn() -> Language,

    pub class_types: &'static [&'static str],
    pub function_types: &'static [&'static str],
    pub import_types: &'static [&'static str],
    pub call_types: &'static [&'static str],
    pub function_boundary_types: &'static [&'static str],

    /// The four FRAMEWORK sets. Empty for every language but PHP, where they
    /// encode Laravel conventions rather than language syntax. They live on the
    /// config, exactly as they do on `LanguageConfig`, so
    /// `test_engine_configs_match_their_language_config` can compare them --
    /// hard-coding them inside `php/` would put them beyond the reach of the one
    /// test that checks the two sides agree.
    pub static_prop_types: &'static [&'static str],
    pub helper_fn_names: &'static [&'static str],
    pub container_bind_methods: &'static [&'static str],
    pub event_listener_properties: &'static [&'static str],

    pub name_field: &'static str,
    pub name_fallback_child_types: &'static [&'static str],
    pub body_field: &'static str,
    pub body_fallback_child_types: &'static [&'static str],

    pub call_function_field: &'static str,
    pub call_accessor_node_types: &'static [&'static str],
    pub call_accessor_field: &'static str,
    /// Empty string means "unset", matching the Python default `""` rather than
    /// `None`; the Python tests it for truthiness.
    pub call_accessor_object_field: &'static str,

    /// `function_label_parens`: when false a function's label is bare.
    pub function_label_parens: bool,

    /// `resolve_function_name_fn`. C and C++ name a function by unwrapping its
    /// `declarator` rather than reading a `name` field, and the Python branches
    /// on `is not None` -- so this is `Option` for the same reason, not a hook
    /// with a no-op default: `None` selects a DIFFERENT branch, it does not mean
    /// "do nothing".
    pub resolve_function_name: Option<fn(&Ctx, Node) -> R<Option<String>>>,

    /// `sanitize_symbol_name_fn`. Ruby encodes a trailing `!`, `?` or `=` into
    /// a safe id component (#3077). The ID uses the sanitized name; the LABEL
    /// keeps the raw one.
    pub sanitize_symbol_name: Option<fn(&str) -> String>,

    /// The result key `Ctx::type_table` is emitted under, when non-empty --
    /// `"cpp_type_table"` for C++. `None` means the language builds no table, and
    /// the key is absent rather than empty, matching the Python's `elif
    /// type_table:` guard.
    pub type_table_key: Option<&'static str>,

    pub hooks: &'static (dyn LangHooks + Sync),
}

/// The one filesystem question a walker is allowed to ask, and it asks Python.
///
/// `#include "foo.h"` resolves through `Path.resolve()` -- symlinks, `..`
/// normalization, a non-existent tail -- and reproducing that in Rust is exactly
/// the kind of surface whose failures are silent. So the walker does the string
/// work and Graphify's own `_resolve_c_include_path` answers, the same seam
/// `js::imports::Resolver` and `py::imports::Resolver` already use.
pub trait PathResolver {
    /// `(raw) -> resolved absolute path`, or None when it is not a real file.
    fn resolve(&self, raw: &str) -> R<Option<String>>;
}

/// Whether a hook consumed the node. `Handled::Yes` means the Python returned.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Handled {
    Yes,
    No,
}

/// What one call site resolved to, before the defer decision.
#[derive(Default)]
pub struct CallInfo {
    pub callee_name: Option<String>,
    pub is_member_call: bool,
    pub is_this_field_call: bool,
    pub member_receiver: Option<String>,
    /// Swift's separate receiver slot: the Python writes
    /// `swift_receiver or member_receiver` into the raw_call.
    pub swift_receiver: Option<String>,
    /// Kotlin / C# fully-qualified call prefixes.
    pub qualified_prefix: Option<String>,
}

/// The per-language blocks. Every method defaults to doing nothing, so a
/// language implements only the guards its Python actually has.
pub trait LangHooks {
    /// `config.import_handler`. Runs for a node in `import_types`.
    fn import_handler<'tree>(&self, _ctx: &mut Ctx<'_, 'tree>, _node: Node<'tree>) -> R<()> {
        Ok(())
    }

    /// C#'s `metadata` on a class node, computed BEFORE `add_node`.
    fn class_metadata<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _parent_class_nid: Option<&str>,
    ) -> R<Vec<(&'static str, Val)>> {
        Ok(Vec::new())
    }

    /// Rewrite a class's LABEL against the enclosing scope, and return the
    /// segments to push while its body is walked.
    ///
    /// Ruby alone: `module Billing; class Invoice` labels `Billing::Invoice`,
    /// and a compact `class Billing::Invoice` splits into the same two segments
    /// so both declaration styles converge on one label (#2302).
    fn qualify_class_name(&self, _ctx: &Ctx, name: &str) -> R<(String, Vec<String>)> {
        Ok((name.to_string(), Vec::new()))
    }

    /// The per-language class block: inheritance, interfaces, annotations,
    /// record components. Runs after the class node and its containment edge,
    /// before the body is recursed into.
    fn on_class<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _class_nid: &str,
        _class_name: &str,
        _line: usize,
    ) -> R<()> {
        Ok(())
    }

    /// The declarations that sit BETWEEN the class and function branches --
    /// fields, properties, annotation elements. Returning `Handled::Yes` means
    /// the Python returned and the node is consumed.
    fn before_function<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        Ok(Handled::No)
    }

    /// The per-language function block: parameter types, return type,
    /// annotations. Runs after the function node and its edge.
    fn on_function<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _func_nid: &str,
        _func_name: &str,
        _line: usize,
        _parent_class_nid: Option<&str>,
    ) -> R<()> {
        Ok(())
    }

    /// Immediately after a function's body is pushed to `function_bodies`.
    ///
    /// Kotlin alone: `object : Foo { ... }` anonymous objects live inside a
    /// function body, which the function branch never recurses into, so their
    /// members and every call inside them got no nodes at all (#2347).
    fn on_function_body<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _func_nid: &str,
        _body: Node<'tree>,
    ) -> R<()> {
        Ok(())
    }

    /// Extra top-level keys on the result dict, after nodes/edges/raw_calls.
    ///
    /// Kotlin's declared `package` qualifies every node in the file, and the
    /// import-target and qualified-call resolvers key their per-package symbol
    /// indexes off it (#2526/#2550).
    fn result_extra<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        _root: Node<'tree>,
    ) -> R<Vec<(&'static str, Val)>> {
        Ok(Vec::new())
    }

    /// The trailing `_<lang>_extra_walk` slot, before the default recurse.
    fn extra_walk<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _parent_class_nid: Option<&str>,
    ) -> R<Handled> {
        Ok(Handled::No)
    }

    /// Extract the callee and receiver from a call node. Returning `None` means
    /// "use the generic accessor path" (`call_function_field` +
    /// `call_accessor_node_types`), which is what a language with no call guard
    /// does.
    ///
    /// Takes `&mut Ctx` and the caller because the branch it stands for is not
    /// purely an extraction in every language: C#'s `invocation_expression` arm
    /// also emits a `references[generic_arg]` edge per call-site type argument
    /// (#2911), from the caller, before the defer decision below.
    fn call_info<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<Option<CallInfo>> {
        Ok(None)
    }

    /// Whether this call must defer to receiver-typed cross-file resolution
    /// instead of binding to a bare name here.
    fn defers(&self, _info: &CallInfo) -> bool {
        false
    }

    /// A last look at the resolved target, AFTER the `label_to_nid` lookup.
    ///
    /// One language reaches this: a C# `new A.B.Foo()` whose bare name matches
    /// only a sourceless stub in this file would bind to the stub and never
    /// reach `_resolve_csharp_qualified_calls`, the one pass that can honour the
    /// namespace. It is a separate slot from `defers` because it needs the
    /// looked-up id, which `defers` by definition does not have.
    fn refine_target(&self, _ctx: &Ctx, _info: &CallInfo, tgt: Option<String>) -> Option<String> {
        tgt
    }

    /// Extra keys appended to a `raw_call`, in Python's order, after `receiver`.
    ///
    /// `node` is the CALL node: C# resolves a receiver's type by the call's byte
    /// offset, so the position is part of the lookup, not just the name.
    fn raw_call_extra<'tree>(
        &self,
        _ctx: &Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _caller_nid: &str,
        _info: &CallInfo,
        _receiver_types: &RecvTable,
    ) -> Vec<(&'static str, Val)> {
        Vec::new()
    }

    /// Run after the declaration walk and BEFORE the call pass.
    ///
    /// The `_is_<lang>` guards between the two walks, which an inventory of the
    /// walk bodies alone misses: C++ builds its `var -> ClassName` table from
    /// every function body here (#1547), and Ruby and Swift do the same thing at
    /// the same point. File-scoped, not per-body -- a later body's `Foo f;` must
    /// not clobber an earlier binding.
    fn before_calls<'tree>(&self, _ctx: &mut Ctx<'_, 'tree>) -> R<()> {
        Ok(())
    }

    /// Inside the call branch of `walk_calls`, after the edge or raw_call.
    ///
    /// PHP's framework blocks live here: a `config('a.b')` helper call and a
    /// `$this->app->bind(A::class, B::class)` container binding both need the
    /// resolved `callee_name`, so they cannot run before the call is classified.
    fn after_call<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _caller_nid: &str,
        _info: &CallInfo,
    ) -> R<()> {
        Ok(())
    }

    /// At the BODY level of `walk_calls`, before it recurses -- so it sees every
    /// node, not only calls. PHP's `Foo::$bar` static-property and `Foo::BAR`
    /// class-constant edges are emitted here.
    fn walk_calls_extra<'tree>(
        &self,
        _ctx: &mut Ctx<'_, 'tree>,
        _node: Node<'tree>,
        _caller_nid: &str,
    ) -> R<()> {
        Ok(())
    }

    /// After the whole call pass. PHP resolves `pending_listen_edges` here,
    /// because a `$listen` array names classes that may be declared later.
    fn after_calls<'tree>(&self, _ctx: &mut Ctx<'_, 'tree>) -> R<()> {
        Ok(())
    }

    /// A per-file name pre-scan, run over the whole tree BEFORE the walk.
    ///
    /// C# is the one caller: `_csharp_pre_scan_interfaces` collects the file's
    /// `interface_declaration` names so `on_class` can classify a base type as
    /// `implements` rather than `inherits` without a second pass. Languages with
    /// no pre-scan leave the set empty. (Swift's `_swift_pre_scan` returns TWO
    /// sets; when Swift lands, this widens rather than a second slot appearing.)
    fn prescan<'tree>(&self, _ctx: &Ctx<'_, 'tree>, _root: Node<'tree>) -> R<HashSet<String>> {
        Ok(HashSet::new())
    }
}

/// The per-method receiver table: `name -> declared type`, for the two languages
/// that build one.
///
/// The two shapes are not interchangeable and the difference is load-bearing.
/// Java's is a flat map built once per method. C#'s is POSITIONAL (#2472): a
/// name can be bound several times in one method -- a parameter, a `var` local
/// in an inner block, a pattern binding in one arm of an `if` -- and the binding
/// that applies is the innermost one whose lexical range contains the call.
///
/// They are one enum with one accessor rather than two types so that neither
/// hook can reach for the wrong lookup: `type_of` takes the call offset in both
/// cases and `Flat` simply ignores it.
pub enum RecvTable {
    Flat(HashMap<String, String>),
    Scoped {
        /// name -> [(scope_start_byte, scope_end_byte, type_or_untypable)]
        bindings: HashMap<String, Vec<(usize, usize, Option<String>)>>,
        /// The class field/property base scope, consulted when no lexical
        /// binding covers the call.
        base: HashMap<String, String>,
    },
}

impl Default for RecvTable {
    fn default() -> Self {
        RecvTable::Flat(HashMap::new())
    }
}

impl RecvTable {
    /// The type of `name` as seen from `call_byte`, or None for "no edge".
    ///
    /// `Scoped` is `_csharp_scoped_receiver_type`: candidates are the bindings
    /// whose range contains the offset, the smallest range wins, and a TIE at
    /// the innermost range (an illegal same-declaration-space clash, or two
    /// sibling pattern bindings) yields None -- never a guess. No candidate at
    /// all falls back to the class fields.
    pub fn type_of(&self, name: &str, call_byte: usize) -> Option<&str> {
        match self {
            RecvTable::Flat(m) => m.get(name).map(String::as_str),
            RecvTable::Scoped { bindings, base } => {
                let candidates: Vec<&(usize, usize, Option<String>)> = bindings
                    .get(name)
                    .map(|v| v.iter().filter(|b| b.0 <= call_byte && call_byte < b.1).collect())
                    .unwrap_or_default();
                if candidates.is_empty() {
                    return base.get(name).map(String::as_str);
                }
                let innermost = candidates.iter().map(|b| b.1 - b.0).min()?;
                let mut inner = candidates.iter().filter(|b| b.1 - b.0 == innermost);
                let first = inner.next()?;
                if inner.next().is_some() {
                    return None;
                }
                first.2.as_deref()
            }
        }
    }
}

/// Everything `_extract_generic` keeps in its local scope for one file.
pub struct Ctx<'a, 'tree> {
    pub cfg: &'static EngineConfig,
    pub src: &'a [u8],
    pub str_path: &'a str,
    pub stem: String,
    pub file_nid: String,

    pub nodes: Vec<NodeRow>,
    pub seen_ids: HashSet<String>,
    pub edges: Vec<EdgeRow>,
    pub raw_calls: Vec<RawCall>,

    pub callable_def_nids: HashSet<String>,
    pub callable_class_nids: HashSet<String>,
    pub function_bodies: Vec<(String, Node<'tree>)>,

    /// `<lang>_field_types`: {class_nid: {field_name: declared_type}}.
    pub field_types: HashMap<String, HashMap<String, String>>,
    /// `<lang>_method_scopes`, keyed by the body's byte range rather than
    /// Python's `id(body)`: unique within one tree and stable across the clone
    /// the call pass makes, which a pointer would not be.
    pub method_scopes: HashMap<(usize, usize), (Node<'tree>, String)>,

    pub label_to_nid: HashMap<String, String>,
    pub nid_to_sf: HashMap<String, String>,
    pub seen_call_pairs: HashSet<(String, String)>,

    /// C#'s `namespace_stack`. Every id minted from the file stem carries the
    /// enclosing namespace (`_make_id(stem, ".".join(namespace_stack), name)`),
    /// and `add_node` stamps it as `metadata.namespace`. Empty for every other
    /// language on this engine, where the joined middle part is `""` and
    /// `make_id` drops it -- so the ids are unchanged by its presence.
    pub namespace_stack: Vec<String>,
    /// C#'s `scope_stack`: one `s<start_byte>` entry per enclosing namespace,
    /// stamped on every non-namespace node as `metadata.scope_chain` and read by
    /// the `using` import handler for `scope_kind` / `scope_id`.
    pub scope_stack: Vec<String>,
    /// Whatever `LangHooks::prescan` collected for this file. C#: the names
    /// declared as `interface` here.
    pub prescan: HashSet<String>,

    /// `label_to_nid_ci`: the same map as `label_to_nid`, keyed on the
    /// LOWERCASED label. PHP's framework blocks resolve a class named in a
    /// string or a `::class` constant, where the source casing is not reliable.
    pub label_to_nid_ci: HashMap<String, String>,

    /// `seen_helper_ref_pairs` / `seen_static_ref_pairs` / `seen_bind_pairs`,
    /// merged. The Python keeps three sets; the RELATION is part of every key
    /// and the three relations are disjoint (`uses_config`, `uses_static_prop`,
    /// `bound_to`), so one set is exactly equivalent and cannot mix them up.
    pub seen_rel_triples: HashSet<(String, String, String)>,

    /// `pending_listen_edges`: `(event_class, listener_class, line)` harvested
    /// during the walk and resolved AFTER the call pass, because a Laravel
    /// `$listen` array names classes that may be declared later in the file.
    pub pending_listen_edges: Vec<(String, String, usize)>,

    /// The enclosing class/module segments, for a language that QUALIFIES a
    /// nested declaration's label. Ruby alone: `module Billing; class Invoice`
    /// labels `Billing::Invoice`, and both declaration styles converge on one
    /// label (#2302). Distinct from `namespace_stack`, which C# uses for ids and
    /// node metadata and Ruby never touches.
    pub scope_segments: Vec<String>,

    /// Property / field INITIALIZERS, walked by the call pass after every
    /// function body. `val repo = createRepo()` is a call that lives in no
    /// function, so without this it produced no edge (#1356/#2565). Keyed by the
    /// node that OWNS the initializer -- the enclosing class, or the file for a
    /// top-level property.
    pub initializer_nodes: Vec<(String, Node<'tree>)>,

    /// `<lang>_var_types`: per-CALLER `var -> ClassName`, where the other
    /// languages' tables are per-method-body or per-file. Ruby alone.
    pub caller_var_types: HashMap<String, HashMap<String, Option<String>>>,

    /// raw_calls collected during the DECLARATION walk, appended after the call
    /// pass so they land at the END of the list. Ruby's `include`/`extend`
    /// mixins alone -- they are found in a class body, before `raw_calls` exists.
    pub deferred_raw_calls: Vec<RawCall>,

    /// `type_table`: the per-file `var -> ClassName` map the cross-file
    /// member-call pass uses to type a receiver. Populated by `before_calls` and
    /// emitted under `EngineConfig::type_table_key`.
    pub type_table: HashMap<String, String>,

    /// The include resolver, for the languages that have one. `None` makes any
    /// file that needs it defer, which is the same rule the JS and Python
    /// resolvers follow: resolution has no safe default.
    pub path_resolver: Option<&'a dyn PathResolver>,
}

impl<'a, 'tree> Ctx<'a, 'tree> {
    pub fn mkid(&self, parts: &[&str]) -> R<String> {
        make_id_ascii(parts).ok_or("non_ascii_id")
    }

    pub fn text(&self, node: Node) -> R<&'a str> {
        text_checked(node, self.src).ok_or("invalid_utf8_text")
    }

    /// The joined namespace: `".".join(namespace_stack)`.
    pub fn ns(&self) -> String {
        self.namespace_stack.join(".")
    }

    /// `add_node`, whole.
    ///
    /// The namespace / scope_chain merge is NOT optional or C#-specific in the
    /// Python: `add_node` applies it to every node it mints, including functions,
    /// properties and enum members. It is a no-op for a language that never
    /// pushes either stack, which is every language on this engine but C#.
    ///
    /// Key order is Python's: `id, label, file_type, source_file,
    /// source_location, [type], [metadata]`, with `metadata`'s own keys in the
    /// order `dict(metadata)` then the two `setdefault`s produce.
    pub fn add_node_full(
        &mut self,
        nid: &str,
        label: &str,
        line: usize,
        node_type: Option<&'static str>,
        metadata: Vec<(&'static str, Val)>,
    ) {
        if !self.seen_ids.insert(nid.to_string()) {
            return;
        }
        let mut merged: Vec<(String, Val)> = metadata
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        // `setdefault`: an explicit entry of the same name wins.
        if !self.namespace_stack.is_empty() && !merged.iter().any(|(k, _)| k == "namespace") {
            merged.push(("namespace".to_string(), Val::S(self.ns())));
        }
        if !self.scope_stack.is_empty()
            && node_type != Some("namespace")
            && !merged.iter().any(|(k, _)| k == "scope_chain")
        {
            merged.push((
                "scope_chain".to_string(),
                Val::List(self.scope_stack.iter().map(|s| Val::S(s.clone())).collect()),
            ));
        }
        let mut fields = vec![
            ("label", Val::S(label.to_string())),
            ("file_type", Val::Static("code")),
            ("source_file", Val::S(self.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
        ];
        if let Some(t) = node_type {
            fields.push(("type", Val::Static(t)));
        }
        // `if merged:` -- an empty dict adds no key at all.
        if !merged.is_empty() {
            fields.push(("metadata", Val::Meta(meta::sanitize(merged))));
        }
        self.nodes.push(NodeRow {
            id: nid.to_string(),
            fields,
        });
    }

    pub fn add_node_meta(&mut self, nid: &str, label: &str, line: usize, metadata: Vec<(&'static str, Val)>) {
        self.add_node_full(nid, label, line, None, metadata);
    }

    pub fn add_node(&mut self, nid: &str, label: &str, line: usize) {
        self.add_node_full(nid, label, line, None, Vec::new());
    }

    pub fn add_edge(&mut self, src: &str, tgt: &str, relation: &'static str, line: usize) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields: vec![
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    /// `add_edge(..., context=...)`: same shape, `context` LAST.
    pub fn add_edge_ctx(
        &mut self,
        src: &str,
        tgt: &str,
        relation: &'static str,
        line: usize,
        context: &'static str,
    ) {
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields: vec![
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
                ("context", Val::Static(context)),
            ],
        });
    }

    /// `add_edge(..., context=..., metadata=...)`: `context` then `metadata`,
    /// both last and both conditional on being non-empty -- the Python appends
    /// each only under `if context:` / `if metadata:`.
    pub fn add_edge_meta(
        &mut self,
        src: &str,
        tgt: &str,
        relation: &'static str,
        line: usize,
        context: Option<&'static str>,
        metadata: Vec<(&'static str, Val)>,
    ) {
        let mut fields = vec![
            ("confidence", Val::Static("EXTRACTED")),
            ("source_file", Val::S(self.str_path.to_string())),
            ("source_location", Val::S(format!("L{line}"))),
            ("weight", Val::F(1.0)),
        ];
        if let Some(c) = context {
            fields.push(("context", Val::Static(c)));
        }
        if !metadata.is_empty() {
            let owned: Vec<(String, Val)> =
                metadata.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
            fields.push(("metadata", Val::Meta(meta::sanitize(owned))));
        }
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation,
            fields,
        });
    }

    /// A framework edge with `confidence_score`, deduped on
    /// `(source, target, relation)`.
    ///
    /// Its own emitter because the key order differs from `add_edge`'s:
    /// `confidence_score` sits between `confidence` and `source_file`, and there
    /// is no `context`. The order reaches the exported JSON.
    pub fn add_edge_scored(
        &mut self,
        src: &str,
        tgt: &str,
        relation: String,
        line: usize,
    ) -> bool {
        let key = (src.to_string(), tgt.to_string(), relation.clone());
        if !self.seen_rel_triples.insert(key) {
            return false;
        }
        self.edges.push(EdgeRow {
            source: src.to_string(),
            target: tgt.to_string(),
            relation: "",
            fields: vec![
                ("__relation", Val::S(relation)),
                ("confidence", Val::Static("EXTRACTED")),
                ("confidence_score", Val::F(1.0)),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
        true
    }

    /// The import edge appended directly by a handler: `context` THIRD, right
    /// after `relation`. A different key order from `add_edge_ctx`, and the
    /// order reaches the pickled result, so it is its own emitter.
    pub fn add_import_edge(&mut self, tgt: &str, line: usize) {
        let src = self.file_nid.clone();
        self.edges.push(EdgeRow {
            source: src,
            target: tgt.to_string(),
            relation: "imports",
            fields: vec![
                ("context", Val::Static("import")),
                ("confidence", Val::Static("EXTRACTED")),
                ("source_file", Val::S(self.str_path.to_string())),
                ("source_location", Val::S(format!("L{line}"))),
                ("weight", Val::F(1.0)),
            ],
        });
    }

    /// `ensure_named_node`: a SOURCELESS stub when the name is not defined in
    /// this file, so the corpus-level rewire can collapse it onto the real
    /// definition (#1402). The scoped probe is `_make_id(stem,
    /// ".".join(namespace_stack), name)`; outside C# the middle part is `""` and
    /// `make_id` drops it.
    pub fn ensure_named_node(&mut self, name: &str, _line: usize) -> R<String> {
        let scoped = self.mkid(&[&self.stem.clone(), &self.ns(), name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if !self.seen_ids.contains(&bare) {
            self.seen_ids.insert(bare.clone());
            self.nodes.push(NodeRow {
                id: bare.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                    ("origin_file", Val::S(self.str_path.to_string())),
                ],
            });
        }
        Ok(bare)
    }

    /// The bare `_make_id(stem, base) else _make_id(base)` stub the Java/Groovy
    /// parent emitter uses. NOT `ensure_named_node`: no `origin_file` key, and
    /// the scoped probe omits the empty namespace part.
    pub fn ensure_parent_node(&mut self, name: &str) -> R<String> {
        let scoped = self.mkid(&[&self.stem.clone(), name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if !self.seen_ids.contains(&bare) {
            self.seen_ids.insert(bare.clone());
            self.nodes.push(NodeRow {
                id: bare.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                ],
            });
        }
        Ok(bare)
    }

    /// C#'s base-type stub, from the `base_list` block.
    ///
    /// A third variant, not a redundant one: the scoped probe is the THREE-part
    /// `_make_id(stem, namespace, base)` (where `ensure_parent_node` uses two),
    /// and the stub carries no `origin_file` (where `ensure_named_node` does).
    /// All three mint the same id when the namespace is empty and differ in node
    /// SHAPE, which is exactly the kind of difference a per-file parity run sees
    /// and a casual reading does not.
    pub fn ensure_scoped_stub(&mut self, name: &str) -> R<String> {
        let scoped = self.mkid(&[&self.stem.clone(), &self.ns(), name])?;
        if self.seen_ids.contains(&scoped) {
            return Ok(scoped);
        }
        let bare = self.mkid(&[name])?;
        if !self.seen_ids.contains(&bare) {
            self.seen_ids.insert(bare.clone());
            self.nodes.push(NodeRow {
                id: bare.clone(),
                fields: vec![
                    ("label", Val::S(name.to_string())),
                    ("file_type", Val::Static("code")),
                    ("source_file", Val::Static("")),
                    ("source_location", Val::Static("")),
                ],
            });
        }
        Ok(bare)
    }

    /// The #1899 guard: a name that normalizes to nothing would collapse
    /// `_make_id(prefix, name)` onto its path-derived prefix.
    pub fn normalizes_to_something(&self, name: &str) -> R<bool> {
        Ok(!normalize_id_ascii(name).ok_or("non_ascii_id")?.is_empty())
    }
}

// ── The per-file driver ──────────────────────────────────────────────────────

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tree_sitter::Parser;

use crate::ids::file_stem;
use crate::js::ast::children;
use crate::js::emit;
use crate::Outcome;

fn tree_depth(root: Node) -> u32 {
    let mut stack = vec![(root, 1u32)];
    let mut max = 1u32;
    while let Some((n, d)) = stack.pop() {
        if d > max {
            max = d;
            if max > MAX_DEPTH {
                return max;
            }
        }
        for c in children(n) {
            stack.push((c, d + 1));
        }
    }
    max
}

/// The whole per-file pipeline, shared by every language on this engine.
///
/// `receiver_types_for` is the one thing that varies structurally: Java and C#
/// build a per-method receiver table from the class's field types, and other
/// languages build nothing. It is a plain function rather than a hook because it
/// runs BETWEEN the two walks, not inside either.
pub fn run<'py>(
    py: Python<'py>,
    cfg: &'static EngineConfig,
    path: &str,
    source: &[u8],
    receiver_types_for: fn(&Ctx, Node, &HashMap<String, String>) -> R<RecvTable>,
    path_resolver: Option<&dyn PathResolver>,
) -> PyResult<Outcome<'py>> {
    match extract(py, cfg, path, source, receiver_types_for, path_resolver) {
        Ok(dict) => Ok(Outcome::Native(dict)),
        Err(reason) => Ok(Outcome::Defer(reason)),
    }
}

fn extract<'py>(
    py: Python<'py>,
    cfg: &'static EngineConfig,
    path: &str,
    source: &[u8],
    receiver_types_for: fn(&Ctx, Node, &HashMap<String, String>) -> R<RecvTable>,
    path_resolver: Option<&dyn PathResolver>,
) -> Result<Bound<'py, PyDict>, &'static str> {
    // One validation of the whole buffer makes every later `text()` sound by
    // construction -- see `js::extract` for the U+FFFD divergence this prevents.
    if std::str::from_utf8(source).is_err() {
        return Err("source_not_utf8");
    }
    let stem = file_stem(path).ok_or("path_needs_pathlib")?;
    let file_nid = make_id_ascii(&[path]).ok_or("non_ascii_path")?;

    let mut parser = Parser::new();
    parser
        .set_language(&(cfg.grammar)())
        .map_err(|_| "grammar_load_failed")?;
    let tree = parser.parse(source, None).ok_or("parse_failed")?;
    let root = tree.root_node();
    // Python attaches a `parse_errors` block and keeps going; its recovery is
    // authoritative and reproducing it is a separate surface, so defer.
    if root.has_error() {
        return Err("parse_error");
    }
    if tree_depth(root) > MAX_DEPTH {
        return Err("tree_too_deep");
    }

    let mut ctx = Ctx {
        cfg,
        src: source,
        str_path: path,
        stem,
        file_nid: file_nid.clone(),
        nodes: Vec::new(),
        seen_ids: HashSet::new(),
        edges: Vec::new(),
        raw_calls: Vec::new(),
        callable_def_nids: HashSet::new(),
        callable_class_nids: HashSet::new(),
        function_bodies: Vec::new(),
        field_types: HashMap::new(),
        method_scopes: HashMap::new(),
        label_to_nid: HashMap::new(),
        nid_to_sf: HashMap::new(),
        seen_call_pairs: HashSet::new(),
        namespace_stack: Vec::new(),
        scope_stack: Vec::new(),
        prescan: HashSet::new(),
        label_to_nid_ci: HashMap::new(),
        seen_rel_triples: HashSet::new(),
        pending_listen_edges: Vec::new(),
        scope_segments: Vec::new(),
        initializer_nodes: Vec::new(),
        caller_var_types: HashMap::new(),
        deferred_raw_calls: Vec::new(),
        type_table: HashMap::new(),
        path_resolver,
    };

    // Before the file node, as in Python: `_csharp_pre_scan_interfaces(root,
    // source)` runs above `add_node(file_nid, ...)`.
    ctx.prescan = cfg.hooks.prescan(&ctx, root)?;

    let file_label = path.rsplit('/').next().unwrap_or(path).to_string();
    ctx.add_node(&file_nid, &file_label, 1);

    walk::walk(&mut ctx, root, None)?;

    // ── Call-graph pass ─────────────────────────────────────────────────────
    // `label_to_nid` is built from the COMPLETED node list, exactly as in
    // Python: a call may name something declared later in the file.
    for n in &ctx.nodes {
        let mut sf = String::new();
        let mut label = String::new();
        for (k, v) in &n.fields {
            match (*k, v) {
                ("source_file", Val::S(s)) => sf = s.clone(),
                ("source_file", Val::Static(s)) => sf = s.to_string(),
                ("label", Val::S(s)) => label = s.clone(),
                _ => {}
            }
        }
        ctx.nid_to_sf.insert(n.id.clone(), sf);
        let normalised = label.trim_matches(|c| c == '(' || c == ')').trim_start_matches('.');
        ctx.label_to_nid.insert(normalised.to_string(), n.id.clone());
        ctx.label_to_nid_ci
            .insert(normalised.to_lowercase(), n.id.clone());
    }

    cfg.hooks.before_calls(&mut ctx)?;

    // Every per-method receiver table is built BEFORE any body is walked, as in
    // Python, where the whole dict comprehension is evaluated up front.
    let scopes: Vec<((usize, usize), (Node, String))> = ctx
        .method_scopes
        .iter()
        .map(|(k, (n, c))| (*k, (*n, c.clone())))
        .collect();
    let empty: HashMap<String, String> = HashMap::new();
    let mut tables: HashMap<(usize, usize), RecvTable> = HashMap::new();
    for (body_key, (method_node, class_nid)) in scopes {
        let fields = ctx.field_types.get(&class_nid).unwrap_or(&empty).clone();
        tables.insert(body_key, receiver_types_for(&ctx, method_node, &fields)?);
    }

    let bodies: Vec<(String, Node)> = ctx.function_bodies.clone();
    let no_table = RecvTable::default();
    for (caller_nid, body) in bodies {
        let key = (body.start_byte(), body.end_byte());
        let table = tables.get(&key).unwrap_or(&no_table);
        calls::walk_calls(&mut ctx, body, &caller_nid, table)?;
    }
    // #1356/#2565: property initializers, AFTER every function body.
    // `walk_calls` self-guards against re-entering a function body and dedups
    // through `seen_call_pairs`, so a closure inside an initializer is not
    // double-walked.
    let inits: Vec<(String, Node)> = ctx.initializer_nodes.clone();
    for (owner_nid, init_node) in inits {
        calls::walk_calls(&mut ctx, init_node, &owner_nid, &no_table)?;
    }

    cfg.hooks.after_calls(&mut ctx)?;
    // Appended AFTER every raw_call the call pass produced, matching the
    // Python's `raw_calls.extend(_ruby_mixin_calls)` at the very end.
    let deferred = std::mem::take(&mut ctx.deferred_raw_calls);
    ctx.raw_calls.extend(deferred);

    // ── Clean edges ─────────────────────────────────────────────────────────
    let mut clean: Vec<&EdgeRow> = Vec::with_capacity(ctx.edges.len());
    for e in &ctx.edges {
        let target_ok = ctx.seen_ids.contains(&e.target)
            || matches!(e.relation, "imports" | "imports_from" | "re_exports");
        if ctx.seen_ids.contains(&e.source) && target_ok {
            clean.push(e);
        }
    }

    let out = PyDict::new(py);
    let nodes = PyList::empty(py);
    for n in &ctx.nodes {
        let is_callable = ctx.callable_def_nids.contains(&n.id);
        let is_class = ctx.callable_class_nids.contains(&n.id);
        nodes
            .append(emit::node_to_py(py, n, is_callable, is_class).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let edges = PyList::empty(py);
    for e in clean {
        edges
            .append(emit::edge_to_py(py, e).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    let raw_calls = PyList::empty(py);
    for c in &ctx.raw_calls {
        raw_calls
            .append(emit::raw_call_to_py(py, c).map_err(|_| "py_error")?)
            .map_err(|_| "py_error")?;
    }
    out.set_item("nodes", nodes).map_err(|_| "py_error")?;
    out.set_item("edges", edges).map_err(|_| "py_error")?;
    out.set_item("raw_calls", raw_calls).map_err(|_| "py_error")?;
    for (k, v) in cfg.hooks.result_extra(&ctx, root)? {
        let d = PyDict::new(py);
        v.set_on(&d, k).map_err(|_| "py_error")?;
        let item = d.get_item(k).map_err(|_| "py_error")?.ok_or("py_error")?;
        out.set_item(k, item).map_err(|_| "py_error")?;
    }
    // `elif type_table:` -- absent, not empty, when nothing was recorded.
    if let Some(key) = cfg.type_table_key {
        if !ctx.type_table.is_empty() {
            let d = PyDict::new(py);
            d.set_item("path", path).map_err(|_| "py_error")?;
            let t = PyDict::new(py);
            // Sorted: Python inserts in walk order, and a HashMap has none.
            let mut pairs: Vec<(&String, &String)> = ctx.type_table.iter().collect();
            pairs.sort();
            for (k, v) in pairs {
                t.set_item(k, v).map_err(|_| "py_error")?;
            }
            d.set_item("table", t).map_err(|_| "py_error")?;
            out.set_item(key, d).map_err(|_| "py_error")?;
        }
    }
    Ok(out)
}

/// Every language driven by this engine, for the seam test to compare against
/// the `LanguageConfig` it is meant to mirror.
///
/// A hand-written walker (`js/`, `py/`) HARD-CODES its dispatch sets, so the
/// seam test can only assert that the Python config's optional fields are empty
/// -- the walker would ignore them. An engine-driven language is the opposite:
/// every one of these fields is READ from the config at run time, so the test
/// can check the real thing, field for field. That check gets stronger with each
/// language added rather than needing a new exemption.
pub fn engine_configs() -> Vec<&'static EngineConfig> {
    vec![
        &crate::java::CONFIG,
        &crate::csharp::CONFIG,
        &crate::c::CONFIG,
        &crate::cpp::CONFIG,
        &crate::php::CONFIG,
        &crate::ruby::CONFIG,
        &crate::kotlin::CONFIG,
        &crate::lua::CONFIG,
        &crate::groovy::CONFIG,
        &crate::scala::CONFIG,
    ]
}

/// The default for a language that builds no receiver table.
pub fn no_receiver_types(
    _ctx: &Ctx,
    _method_node: Node,
    _fields: &HashMap<String, String>,
) -> R<RecvTable> {
    Ok(RecvTable::default())
}
