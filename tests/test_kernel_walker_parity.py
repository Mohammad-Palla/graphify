"""Byte-for-byte parity between the native walker and `_extract_generic`, on the
constructs real corpora barely contain.

`harness/kernel_walker_parity.py` compares every JS/TS file in four corpora, which
is the primary gate -- but corpus frequency and *risk* are not the same thing. The
whole import surface is reimplemented in Rust (see `src/js/imports.rs` on why the
resolver stayed in Python but the edge building did not), and some of its shapes
are almost absent from the corpora: `export { x } from './m'` produced exactly ONE
`re_exports` edge across a 900-file Bun sample, and `import x = require('./m')`
produced none. A path that rare is effectively untested by the corpus gate while
being exactly as load-bearing as the common ones.

So each case below is a small fixture written to disk -- imports resolve against
the real filesystem, so they have to be real files -- run through both
implementations and compared with `sort_keys` canonicalization. A deferral is a
PASS only where the test says so explicitly; anywhere else it would silently hide
a gap, so the default assertion is "native AND identical".
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("graphify_kernel")

import graphify_kernel as gk  # noqa: E402

from graphify.extract import _JS_CONFIG, _TSX_CONFIG, _TS_CONFIG  # noqa: E402
from graphify.extractors import kernel as kseam  # noqa: E402
from graphify.extractors.engine import _extract_generic  # noqa: E402

CONFIG = {"typescript": _TS_CONFIG, "tsx": _TSX_CONFIG, "javascript": _JS_CONFIG}
SUFFIX = {"typescript": ".ts", "tsx": ".tsx", "javascript": ".js"}


@pytest.fixture(autouse=True)
def _python_arm_never_routes_natively(monkeypatch):
    """`_extract_generic` calls `try_extract` at its top, so without this the
    "expected" side would BE the kernel and every case would pass vacuously."""
    monkeypatch.setattr(kseam, "try_extract", lambda *a, **kw: None)


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def compare(tmp_path, source: str, *, language="typescript", extra: dict | None = None):
    """Run both arms over `source`. Returns (native_result_or_None, defer_reason)."""
    for name, text in (extra or {}).items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    path = tmp_path / f"subject{SUFFIX[language]}"
    path.write_text(source)
    got, reason = gk.extract_file(
        str(path), path.read_bytes(), language, kseam._import_resolver(str(path))
    )
    if got is not None:
        want = _extract_generic(path, CONFIG[language])
        assert _canon(got) == _canon(want), (
            f"DIVERGENT\n native: {_canon(got)}\n python: {_canon(want)}"
        )
    return got, reason


def identical(tmp_path, source: str, **kw):
    """Assert the walker handled it natively AND matched. A deferral fails here:
    it is correct but buys nothing, and silently accepting one turns this suite
    into a no-op the day a construct stops being handled."""
    got, reason = compare(tmp_path, source, **kw)
    assert got is not None, f"unexpectedly DEFERRED ({reason})"
    return got


# ── imports: the reimplemented surface ───────────────────────────────────────

MODULE = "export const value = 1;\nexport function helper() { return 2; }\n"


def test_named_import_from_local_file(tmp_path):
    identical(tmp_path, "import { helper } from './mod';\nhelper();\n",
              extra={"mod.ts": MODULE})


def test_named_import_with_alias_and_type_only(tmp_path):
    identical(tmp_path,
              "import { helper as h, type value } from './mod';\n"
              "export function use() { return h(); }\n",
              extra={"mod.ts": MODULE})


def test_default_and_namespace_imports(tmp_path):
    identical(tmp_path,
              "import Thing from './mod';\nimport * as NS from './mod';\n"
              "export function use() { return NS.helper(Thing); }\n",
              extra={"mod.ts": MODULE})


def test_side_effect_only_import(tmp_path):
    identical(tmp_path, "import './mod';\n", extra={"mod.ts": MODULE})


def test_import_of_a_package_outside_the_corpus(tmp_path):
    """An unresolvable specifier takes the `ref`-namespaced branch, and the names
    it binds must shadow in every scope -- otherwise they resolve against the
    corpus label index and fabricate `indirect_call` edges."""
    identical(tmp_path,
              "import { Palette, Search } from 'some-ui-kit';\n"
              "export function draw() { return render(Palette, Search); }\n"
              "export function render(a, b) { return [a, b]; }\n")


def test_reexport_named(tmp_path):
    """One `re_exports` edge appeared in a 900-file Bun sample. This is that path."""
    identical(tmp_path, "export { helper, value } from './mod';\n",
              extra={"mod.ts": MODULE})


def test_reexport_default_is_skipped(tmp_path):
    identical(tmp_path, "export { default as thing, helper } from './mod';\n",
              extra={"mod.ts": MODULE})


def test_export_star(tmp_path):
    identical(tmp_path, "export * from './mod';\n", extra={"mod.ts": MODULE})


def test_export_without_a_from_clause_falls_through_to_children(tmp_path):
    """`export class C {}` must still emit C: only a re-export stops the walk."""
    got = identical(tmp_path,
                    "export const n = 1;\nexport class Widget { render() { return 1; } }\n"
                    "export function make() { return new Widget(); }\n")
    labels = {n["label"] for n in got["nodes"]}
    assert {"Widget", ".render()", "make()"} <= labels, labels


def test_import_equals_require_clause(tmp_path):
    """The module string lives inside `import_require_clause`, not as a direct
    child, so the direct-child scan never sees it."""
    identical(tmp_path, "import mod = require('./mod');\n", extra={"mod.ts": MODULE})


def test_require_whole_module(tmp_path):
    identical(tmp_path, "const mod = require('./mod');\nmod.helper();\n",
              language="javascript", extra={"mod.js": MODULE})


def test_require_destructured(tmp_path):
    identical(tmp_path, "const { helper, value: v } = require('./mod');\nhelper();\n",
              language="javascript", extra={"mod.js": MODULE})


def test_require_member_access(tmp_path):
    identical(tmp_path, "const h = require('./mod').helper;\nh();\n",
              language="javascript", extra={"mod.js": MODULE})


def test_require_inside_a_function_body(tmp_path):
    """`walk_calls` runs `_require_imports_js` too, attributing to the caller."""
    identical(tmp_path,
              "export function load() { const m = require('./mod'); return m.helper(); }\n",
              language="javascript", extra={"mod.js": MODULE})


def test_dynamic_import_awaited(tmp_path):
    identical(tmp_path,
              "export async function load() { const m = await import('./mod'); return m; }\n",
              extra={"mod.ts": MODULE})


def test_dynamic_import_template_string_without_substitution(tmp_path):
    identical(tmp_path,
              "export async function load() { return import(`./mod`); }\n",
              extra={"mod.ts": MODULE})


def test_dynamic_import_with_substitution_is_unresolvable(tmp_path):
    """A computed specifier must emit NO edge, not a guess."""
    got = identical(tmp_path,
                    "export async function load(n) { return import(`./${n}`); }\n")
    assert not [e for e in got["edges"] if e["relation"] == "imports_from"]


def test_dynamic_import_deduplicates_per_caller(tmp_path):
    identical(tmp_path,
              "export async function load() {\n"
              "  await import('./mod');\n  await import('./mod');\n}\n",
              extra={"mod.ts": MODULE})


# ── declarations ─────────────────────────────────────────────────────────────

def test_interface_and_type_alias_and_method_signature(tmp_path):
    identical(tmp_path,
              "export interface Shape { area(): number; name: string }\n"
              "export type Id = string;\n")


def test_enum_members_and_quoted_member_names(tmp_path):
    identical(tmp_path,
              "export enum Kind { Red, Green = 5, 'Odd Name' = 7 }\n")


def test_enum_member_initializer_is_still_walked(tmp_path):
    """Claiming the member must not swallow its initializer."""
    identical(tmp_path,
              "enum E { A = class Inner { m() { return 1; } }.name }\n")


def test_enum_case_insensitive_id_collision_keeps_the_first(tmp_path):
    """`enum E { Value, value }` is legal TS but the id recipe casefolds."""
    identical(tmp_path, "enum E { Value, value }\n")


def test_abstract_class_and_nested_class(tmp_path):
    identical(tmp_path,
              "export abstract class Base { abstract run(): void; }\n"
              "export class Outer { inner() { return 1; } }\n")


def test_namespace_container(tmp_path):
    identical(tmp_path,
              "export namespace Outer { export function inside() { return 1; } }\n")


def test_declare_module_container(tmp_path):
    identical(tmp_path, 'declare module "pkg" { export function f(): void; }\n')


def test_class_field_arrow_becomes_a_method(tmp_path):
    identical(tmp_path,
              "export class C { handler = () => this.other(); other() { return 1; } }\n")


def test_constructor_parameter_properties_type_the_receiver(tmp_path):
    identical(tmp_path,
              "class Svc { run() { return 1; } }\n"
              "export class Uses {\n"
              "  constructor(private svc: Svc) {}\n"
              "  go() { return this.svc.run(); }\n"
              "}\n")


def test_receiver_table_from_new_binding_and_typed_parameter(tmp_path):
    identical(tmp_path,
              "class Svc { run() { return 1; } }\n"
              "export function a() { const s = new Svc(); return s.run(); }\n"
              "export function b(svc: Svc) { return svc.run(); }\n")


def test_receiver_table_rejects_non_bare_types(tmp_path):
    """Arrays, unions and generics are skipped: precision over recall."""
    identical(tmp_path,
              "class Svc { run() { return 1; } }\n"
              "export function a(xs: Svc[]) { return xs; }\n"
              "export function b(u: Svc | null) { return u; }\n"
              "export function c(p: Promise<Svc>) { return p; }\n")


def test_exports_and_prototype_assignment(tmp_path):
    identical(tmp_path,
              "function Foo() {}\n"
              "exports.run = function () { return 1; };\n"
              "module.exports.go = () => 2;\n"
              "Foo.prototype.bar = function () { return 3; };\n",
              language="javascript")


def test_this_and_object_method_assignment_inside_a_function(tmp_path):
    identical(tmp_path,
              "export function make() {\n"
              "  const api = {};\n"
              "  api.run = function () { return 1; };\n"
              "  this.go = () => 2;\n"
              "  return api;\n"
              "}\n",
              language="javascript")


def test_nested_function_declarations_are_scoped_to_the_parent(tmp_path):
    identical(tmp_path,
              "export const Panel = () => {\n"
              "  function handleClick() { return helper(); }\n"
              "  return handleClick;\n"
              "};\n"
              "export function helper() { return 1; }\n")


def test_declaration_nested_inside_a_conditional_is_still_found(tmp_path):
    """The default recurse is what finds these. An earlier walker SKIPPED
    statement kinds outright and one file lost 45 raw_calls plus every node but
    the file node."""
    identical(tmp_path,
              "declare const flag: boolean;\n"
              "if (flag) {\n"
              "  function inner() { return helper(); }\n"
              "}\n"
              "export function helper() { return 1; }\n")


def test_const_initializer_closures_are_tracked(tmp_path):
    identical(tmp_path,
              "export function wrapper(fn) { return fn; }\n"
              "export const handler = wrapper(async (req) => target(req));\n"
              "export function target(r) { return r; }\n")


def test_exported_scalar_binding_is_a_node_but_a_local_one_is_not(tmp_path):
    identical(tmp_path, "export const exported = 1;\nconst internal = 2;\n")


def test_minified_name_that_normalizes_to_nothing_is_skipped(tmp_path):
    """A name normalizing to "" would collapse the id onto the file stem and leak
    the scan path (#1899)."""
    identical(tmp_path, "const $ = () => 1;\nexport const _ = () => 2;\n")


# ── calls and indirect references ────────────────────────────────────────────

def test_member_call_with_capitalized_receiver_defers_to_the_resolver(tmp_path):
    identical(tmp_path,
              "class Helper { static run() { return 1; } }\n"
              "export function go() { return Helper.run(); }\n")


def test_this_field_call_defers_to_the_resolver(tmp_path):
    identical(tmp_path,
              "export class C { go() { return this.dep.run(); } run() { return 1; } }\n")


def test_builtin_globals_are_not_calls(tmp_path):
    identical(tmp_path,
              "export function go() { return new Map(); }\n"
              "export function n() { return parseInt('1'); }\n")


def test_callback_passed_by_name_is_an_indirect_call(tmp_path):
    identical(tmp_path,
              "export function cb() { return 1; }\n"
              "export function go(xs) { return xs.map(cb); }\n")


def test_a_parameter_shadows_a_module_function_of_the_same_name(tmp_path):
    identical(tmp_path,
              "export function cb() { return 1; }\n"
              "export function go(cb) { return [cb].map(cb); }\n")


def test_single_unparenthesised_arrow_parameter_shadows(tmp_path):
    """`x => f(x)` exposes its parameter as `parameter`, not `parameters`."""
    identical(tmp_path,
              "export function cb() { return 1; }\n"
              "export const go = cb => [cb];\n")


def test_catch_binding_shadows_for_the_rest_of_the_clause(tmp_path):
    identical(tmp_path,
              "export function err() { return 1; }\n"
              "export function go() { try { return 1; } catch (err) { return [err]; } }\n")


def test_for_of_loop_binding_shadows(tmp_path):
    identical(tmp_path,
              "export function entry() { return 1; }\n"
              "export function go(xs) { for (const entry of xs) { return [entry]; } }\n")


def test_destructured_and_defaulted_parameters_shadow(tmp_path):
    identical(tmp_path,
              "export function cb() { return 1; }\n"
              "export function other() { return 2; }\n"
              "export function go({ cb }, other = 1) { return [cb, other]; }\n")


def test_module_level_dispatch_table(tmp_path):
    identical(tmp_path,
              "export function handler() { return 1; }\n"
              "export const routes = { get: handler };\n"
              "export const list = [handler];\n")


def test_class_referenced_as_a_value_is_not_an_invocation(tmp_path):
    """A class passed as a descriptor must not mint an `indirect_call` (#2137)."""
    identical(tmp_path,
              "export class Model {}\n"
              "export function select(m) { return m; }\n"
              "export const chosen = select(Model);\n")


def test_anonymous_generator_function_expression_body_is_walked(tmp_path):
    """`generator_function` is a `walk_calls` boundary; omitting it from the
    descend set silently dropped every call in its body."""
    identical(tmp_path,
              "export function sleep(n) { return n; }\n"
              "export function go() {\n"
              "  return wrap(async function* gen() { await sleep(30); yield 1; });\n"
              "}\n"
              "export function wrap(f) { return f; }\n")


def test_tsx_jsx_expression_calls_are_found(tmp_path):
    identical(tmp_path,
              "export function fmt(d) { return d; }\n"
              "export const View = (props) => <div>{fmt(props.d)}</div>;\n",
              language="tsx")


# ── the deferrals, asserted as deferrals ─────────────────────────────────────

@pytest.mark.parametrize("source,expected", [
    ("@Component({}) export class C {}\n", "decorator"),
    ("export class C { @Input() name: string; }\n", "decorator"),
])
def test_decorators_defer_rather_than_guess(tmp_path, source, expected):
    got, reason = compare(tmp_path, source)
    assert got is None and reason == expected, (got, reason)


def test_a_parse_error_defers(tmp_path):
    got, reason = compare(tmp_path, "export function broken( {\n")
    assert got is None and reason == "parse_error", (got, reason)


def test_non_ascii_identifier_defers(tmp_path):
    """`normalize_id` iterates casefold+NFKC to a fixpoint; `ids.rs` reproduces
    that only for ASCII and defers otherwise, provably rather than hopefully."""
    got, reason = compare(tmp_path, "export function İslemYap() { return 1; }\n")
    assert got is None and reason == "non_ascii_id", (got, reason)


def test_invalid_utf8_source_defers(tmp_path):
    """Python decodes with errors="replace"; Rust cannot without allocating per
    call, so the whole file defers instead of yielding a different name."""
    path = tmp_path / "bad.ts"
    path.write_bytes(b"export function f\xff\xfe() { return 1; }\n")
    got, reason = gk.extract_file(
        str(path), path.read_bytes(), "typescript", kseam._import_resolver(str(path))
    )
    assert got is None and reason in ("source_not_utf8", "parse_error"), reason


def test_a_pathologically_deep_tree_defers_instead_of_overflowing(tmp_path):
    """A Rust stack overflow is a SIGSEGV, which the fail-open seam cannot catch --
    it kills the pool worker and fails every file that worker held."""
    depth = 4000
    source = "let c = 0;\n" + "for (let i = 0; i < 1; i++) " * depth + "c++;\n"
    path = tmp_path / "deep.js"
    path.write_text(source)
    got, reason = gk.extract_file(
        str(path), path.read_bytes(), "javascript", kseam._import_resolver(str(path))
    )
    assert got is None and reason == "tree_too_deep", reason


def test_a_missing_resolver_defers_rather_than_assuming_external(tmp_path):
    """Resolution has no safe default: guessing would drop or invent an edge."""
    path = tmp_path / "a.ts"
    path.write_text("import { x } from './mod';\n")
    (tmp_path / "mod.ts").write_text(MODULE)
    got, reason = gk.extract_file(str(path), path.read_bytes(), "typescript", None)
    assert got is None and reason == "no_resolver", reason


def test_a_raising_resolver_defers(tmp_path):
    def _boom(_raw):
        raise RuntimeError("resolver exploded")

    path = tmp_path / "a.ts"
    path.write_text("import { x } from './mod';\n")
    got, reason = gk.extract_file(str(path), path.read_bytes(), "typescript", _boom)
    assert got is None and reason == "resolver_raised", reason
