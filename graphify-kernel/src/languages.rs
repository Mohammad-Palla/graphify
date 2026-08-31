//! Language registry: the single place that decides whether a file goes native.
//!
//! A language appears in [`supported`] only once its walker has passed the parity
//! gate on a real corpus (see `harness/kernel_parity.py`). Until then it is
//! absent, and every file of that language defers to Python. This is deliberately
//! a *whitelist* keyed on the language Graphify already resolved -- never a
//! suffix sniff of our own, which could disagree with `_get_extractor`'s
//! filename special cases (`.blade.php`, MCP configs, ObjC vs MATLAB `.m`,
//! C vs C++ `.h`) and silently route a file to the wrong grammar.

use pyo3::prelude::*;
use tree_sitter::Parser;

use crate::Outcome;

/// A native walker: bytes in, either a Graphify result dict or a reasoned deferral.
///
/// Every walker receives the whole [`crate::Resolvers`] bundle rather than the
/// one resolver its language uses. The alternative -- a per-language resolver
/// type in the signature -- would need either a separate registry per language or
/// an enum at the dispatch point, and the registry being ONE table is the
/// property that makes "which languages can go native" answerable by reading one
/// function.
pub type Walker = for<'py> fn(
    Python<'py>,
    &str,
    &[u8],
    &crate::Resolvers<'py>,
) -> PyResult<Outcome<'py>>;

/// Languages with a parity-gated native walker.
///
/// A language appears here only after `harness/kernel_walker_parity.py` reports
/// **DIVERGENT 0** over every file of that language in the corpora, AND the
/// native rate is high enough for routing to be worth anything. Measured before
/// this list was populated, over 11,834 files in bun / vue / django:
///
/// ```text
/// corpus   language     files  native   deferred  DIVERGENT
/// bun      typescript    3165   99.1%      0.9%          0
/// bun      tsx            176   99.4%      0.6%          0
/// bun      javascript    7867   98.6%      1.4%          0
/// vue      typescript     474   99.6%      0.4%          0
/// vue      javascript      36  100.0%      0.0%          0
/// django   javascript     111   98.2%      1.8%          0
/// django   python        2929   99.9%      0.1%          0
/// bun      python           8  100.0%      0.0%          0
/// graphify python         346  100.0%      0.0%          0
/// guava    java          3275   99.6%      0.4%          0
/// gson     java           264  100.0%      0.0%          0
/// java     java            50  100.0%      0.0%          0
/// serilog  csharp         216   91.7%      8.3%          0
/// newtonsoft csharp       945   93.1%      6.9%          0
/// eShopOnWeb csharp       254  100.0%      0.0%          0
/// efcore   csharp        5762   91.2%      8.8%          0
/// libuv    c              367   80.7%     19.3%          0
/// folly    cpp           2287   65.9%     34.1%          0
/// leveldb  cpp            128   75.0%     25.0%          0
/// spdlog   cpp            143   13.3%     86.7%          0
/// symfony  php          11306   99.8%      0.2%          0
/// laravel  php           3052  100.0%      0.0%          0
/// guzzle   php            137  100.0%      0.0%          0
/// bats     bash           281   96.8%      3.2%          0
/// (6 more) bash           134   72.4%     27.6%          0
/// k8s      go           17865   99.3%      0.7%          0
/// prometheus go           730  100.0%      0.0%          0
/// gin      go              99  100.0%      0.0%          0
/// cargo    rust          1373   62.0%     38.0%          0
/// tokio    rust           793  100.0%      0.0%          0
/// bun      rust          1527   87.9%     12.1%          0
/// serde    rust           208  100.0%      0.0%          0
/// rails    ruby          3458  100.0%      0.0%          0
/// sinatra  ruby           147  100.0%      0.0%          0
/// ktor     kotlin        2527   97.1%      2.9%          0
/// coroutines kotlin      1082   98.2%      1.8%          0
/// okhttp   kotlin         617   98.9%      1.1%          0
/// redis    c              756   56.2%     43.8%          0
/// curl     c             1014   73.7%     26.3%          0
/// kong     lua           1309   99.9%      0.1%          0
/// neovim   lua            844   99.6%      0.4%          0
/// luals    lua            479  100.0%      0.0%          0
/// cats     scala          835   95.6%      4.4%          0
/// akka     scala         1200  100.0%      0.0%          0
/// alamofire swift          98   90.8%      9.2%          0
/// swift-nio swift         554   73.5%     26.5%          0
/// vapor    swift          251   89.6%     10.4%          0
/// zls      zig            103   95.1%      4.9%          0
/// tigerbeetle zig         247   98.4%      1.6%          0
/// ziglang  zig           1200   93.7%      6.3%          0
/// elixir   elixir          568  100.0%      0.0%          0
/// phoenix  elixir          177  100.0%      0.0%          0
/// ecto     elixir          126  100.0%      0.0%          0
/// dune     ocaml           400  100.0%      0.0%          0
/// ocaml    ocaml           900   97.7%      2.3%          0
/// ocaml    ocaml_iface     400   98.8%      1.2%          0
/// flux     julia            96   93.8%      6.2%          0
/// dataframes julia          73   95.9%      4.1%          0
/// julia    julia           937   77.4%     22.6%          0
/// stdlib   fortran         416   99.8%      0.2%          0
/// json-fortran fortran      61   54.1%     45.9%          0
/// lapack   fortran         500    0.2%     99.8%          0
/// afnetworking objc         79   87.3%     12.7%          0
/// sdwebimage objc          262   82.1%     17.9%          0
/// texture  objc            844   67.5%     32.5%          0
/// powershell powershell    507   64.1%     35.9%          0
/// poshgit  powershell       32   50.0%     50.0%          0
/// psreadline powershell      9   33.3%     66.7%          0
/// sqlfluff sql            400   99.8%      0.2%          0
/// sqlfluff sql (rest)    2187  100.0%      0.0%          0
/// postgres sql            855   99.1%      0.9%          0
/// ```
///
/// **SQL's rate is the one to read against its ceiling.** Its parse ceiling is
/// 18.5% -- tree-sitter-sql leaves an ERROR node in four files out of five -- and
/// it is nonetheless 99%+ native, because `sql/` is the one walker here that does
/// NOT defer on `has_error`. `extract_sql` is BUILT around errored trees (an
/// `ERROR` branch, a `fb_proc_or_trigger` branch, two whole-file regex
/// fallbacks), so deferring would have routed almost nothing. See `sql/mod.rs`
/// for the 3,442-file tree-identity measurement that justifies walking them.
///
/// **Read Fortran's three rows together or not at all.** The aggregate ceiling
/// over those corpora is 12.5% and it is meaningless: split by EXTENSION, `.f90`
/// free-form is 99.6% clean over 509 files and `.f` fixed-form FORTRAN 77 is
/// 0.0% over 3,581 -- the grammar cannot parse fixed form at all, and LAPACK is
/// 3,581 fixed-form files. Rejecting Fortran on the aggregate would have been
/// rejecting it on a corpus-selection artefact. json-fortran sits in between
/// because it is capital-`.F90`, so it is C-preprocessed first and the expansion
/// introduces constructs the grammar rejects.
///
/// Scala is the only language ported here with NO parse-error floor: 400/400
/// sampled files parse clean on cats and akka. Its 4.4% deferral on cats is
/// entirely `non_ascii_id` -- that codebase leans on symbolic operators, and the
/// id recipe's casefold+NFKC fixpoint is not reproduced in Rust.
///
/// Swift is the largest port on this engine -- nine guard sites, eleven helper
/// functions, and the only one that needed the ENGINE widened (`prescan` now
/// returns a pair; see `swift/mod.rs`). Two numbers are worth keeping. Its
/// native rate is 79.8% over 903 files, the lowest of any ROUTED language, and
/// every single deferral is `parse_error` -- not one construct gap, so the
/// walker is at its grammar's ceiling and no further Swift work can raise the
/// rate. And that ceiling is why the rate clause is a judgement and not a
/// threshold: 79.8% pays (one file in five parsed twice), where Groovy's 16%
/// does not.
///
/// Lua is the cheapest language ported here and the number worth remembering is
/// not its 99.8%: `engine.py` has ZERO `_is_lua` guards and zero
/// `tree_sitter_lua` guards, so it needed no new engine hook position at all --
/// only the config data and one `import_handler` for `require()`. It reached
/// DIVERGENT 0 on the first parity run, the only language here that has. The
/// four deferrals over 2,632 files are two parse errors and two non-UTF-8
/// sources.
///
/// C's native rate is the lowest of any language here and, like C#'s, it is a
/// parser limit rather than a walker gap -- but a much harder one. 31.8% of
/// those 2,137 files make `tree-sitter-c` produce an ERROR node, because
/// tree-sitter has no preprocessor and a function-like macro in DECLARATION
/// position derails the parse: `UNUSED static int f(void)`,
/// `HEAP_EXPORT(void heap_init(struct heap*))`, `TEST_IMPL(foo)`, `WINAPI`. That
/// is inherent to parsing C without expanding macros, and it caps what any
/// native C walker can reach at roughly two thirds of a real codebase.
///
/// C++ inherits that and adds template metaprogramming. The spread across
/// corpora is the widest of any language here -- leveldb 75%, folly 66%, spdlog
/// 13% -- and spdlog is the shape that explains it: a header-only library whose
/// headers are almost entirely templates and macros. A C++ corpus's native rate
/// is a property of its house style, so the number to quote is a RANGE.
///
/// PHP is the opposite extreme and the best result of any language ported here
/// after JS/TS and Python: 99.8% over 14,495 files. There is no preprocessor and
/// no contextual-keyword trap, so essentially every file parses.
///
/// Bash is the first BESPOKE walker ported (see `bash/`), and its deferrals are
/// a deliberate scope choice rather than a parser limit: a `source` command or a
/// `.sh` script invocation resolves through the filesystem -- `Path.resolve()`,
/// `is_file()`, a `var_bases` table and a traversal guard against an
/// attacker-controllable corpus -- so a file containing either defers whole.
/// Measured across every corpus, 10% of shell files contain a `source` -- but
/// they are not spread evenly, and the per-corpus native rate runs from 96.8%
/// (bats) down to 34.5% (efcore, whose shell scripts are almost all
/// source-and-dispatch wrappers). 368 of 415 files over seven corpora.
///
/// Go is the best result here by volume: 99.3% over 18,694 files. It is the
/// second bespoke walker and, unlike Bash, touches no filesystem at all, so
/// there is no scope deferral -- every file the grammar parses is handled.
///
/// Rust is the third bespoke walker. Its corpus spread is wide for one reason,
/// and it is worth naming so the 62% is not read as a walker gap: 516 of cargo's
/// 522 erroring files are under `tests/testsuite`, which embeds deliberately
/// malformed Rust in string literals for the compiler to reject. tokio and serde
/// are both 100%.
///
/// C#'s deferral rate is an order of magnitude above every other language's and
/// it is NOT a gap in the walker: 8.2% of those 7,177 files make
/// `tree-sitter-c-sharp` 0.23.5 produce an ERROR node, and Python's recovery is
/// authoritative, so the file defers. The dominant cause, measured rather than
/// assumed, is `async` used as an ordinary identifier in EXPRESSION position --
/// `await base.M(async)`, `M(async, x)`, `async ? a : b`, `var x = async` --
/// where the grammar commits to an async-lambda parse. In DECLARATION position
/// (`Task Foo(bool async)`) it parses fine, so it is the call and not the
/// signature that breaks. That is 43% of the erroring files on its own,
/// concentrated in EF Core's parameterized test suites, which take `bool async`
/// and forward it. Conditional
/// compilation (`#if` around an enum member or a collection initializer) is only
/// 13%; it is worth naming because it is the cause CodeGraph documents for its
/// own C# grammar trouble, and it is not the cause here.
///
/// The residual deferrals are, in order: a parse error (Python's recovery is
/// authoritative), a decorator (mints stub nodes through `ensure_named_node`), a
/// non-ASCII identifier (the id recipe's Unicode fixpoint is not reproduced), and
/// one 9 MB fixture too deep to recurse into. Python's two are one parse error and
/// one non-ASCII identifier.
///
/// Graphify's own source is in that table deliberately. django is one codebase
/// with one house style, and a walker that only ever saw django would be gated on
/// a sample that cannot exercise what it does not contain -- `graphify-src` brings
/// walrus operators, `match`, heavy `typing` generics and `getattr` dispatch that
/// django's 2,929 files barely use.
pub fn supported() -> Vec<&'static str> {
    vec![
        "typescript", "tsx", "javascript", "python", "java", "csharp", "c", "cpp",
        "php", "bash", "go", "rust", "ruby", "kotlin", "lua", "scala",
        "swift", "zig", "elixir", "ocaml", "ocaml_interface", "julia",
        "fortran", "objc", "powershell", "sql",
    ]
}

pub fn walker_for(language: &str) -> Option<Walker> {
    match language {
        "typescript" => Some(crate::js::walk_typescript),
        "tsx" => Some(crate::js::walk_tsx),
        "javascript" => Some(crate::js::walk_javascript),
        "python" => Some(crate::py::walk_python),
        "java" => Some(crate::java::walk_java),
        "csharp" => Some(crate::csharp::walk_csharp),
        "c" => Some(crate::c::walk_c),
        "cpp" => Some(crate::cpp::walk_cpp),
        "php" => Some(crate::php::walk_php),
        "bash" => Some(crate::bash::walk_bash),
        "go" => Some(crate::go::walk_go),
        "rust" => Some(crate::rust::walk_rust),
        "ruby" => Some(crate::ruby::walk_ruby),
        "kotlin" => Some(crate::kotlin::walk_kotlin),
        "lua" => Some(crate::lua::walk_lua),
        // STAGED DELIBERATELY, and not because parity failed -- Groovy is
        // DIVERGENT 0 over 2,133 files. It is absent from `supported` because
        // its NATIVE RATE is too low for routing to pay: `tree-sitter-groovy`
        // 0.1.2 produces an ERROR node on 84% of Apache Groovy's own files and
        // 82% of `.gradle` build scripts, measured on the PYTHON side too, so
        // it is the grammar and not this walker. At ~13% native, 87% of files
        // pay a wasted Rust parse and are then parsed again by Python.
        // Measured cold on groovy-apache, ABBA: kernel off 38.7s / 42.8s, on
        // 41.7s / 43.1s -- no win, and the sign never favours ON. Flip the one
        // line in `supported` if the grammar improves; the walker is gated and
        // ready.
        "groovy" => Some(crate::groovy::walk_groovy),
        "scala" => Some(crate::scala::walk_scala),
        "swift" => Some(crate::swift::walk_swift),
        "zig" => Some(crate::zig::walk_zig),
        "elixir" => Some(crate::elixir::walk_elixir),
        "fortran" => Some(crate::fortran::walk_fortran),
        "julia" => Some(crate::julia::walk_julia),
        "objc" => Some(crate::objc::walk_objc),
        "powershell" => Some(crate::powershell::walk_powershell),
        "sql" => Some(crate::sql::walk_sql),
        "ocaml" => Some(crate::ocaml::walk_ocaml),
        "ocaml_interface" => Some(crate::ocaml::walk_ocaml_interface),
        _ => None,
    }
}

/// A structural fingerprint of each linked grammar, for the Python side to
/// compare against the grammar IT loads.
///
/// This exists because a version skew between the two sides is invisible and
/// produces exactly the failure mode this whole design is built to avoid. The
/// walker was measured DIVERGENT=0 on 3,165 TypeScript files and DIVERGENT on 8
/// of 7,867 JavaScript files, for one reason: `tree-sitter-typescript` was 0.23.2
/// on both sides, while `tree-sitter-javascript` was 0.25.0 in Python and 0.23.1
/// here. The two grammars parse `await f()` differently and disagree about
/// whether some files contain an error node at all -- so the kernel was faithfully
/// walking a *different tree* than Python would have.
///
/// Nothing in the build would have caught that: the crate pins a semver RANGE, and
/// `pip install --upgrade tree-sitter-javascript` moves the other side with no
/// signal. So the version agreement is checked at load time and a mismatch
/// disables that language.
///
/// # What is compared, and why it changed
///
/// This used to be `(abi_version, node_kind_count, field_count)`. It is now
/// `(abi_version, kind_names_digest, field_names_digest)` -- a SHA-256 over
/// every node-kind name by id and every field name by id.
///
/// That is strictly stronger on the question actually being asked. The counts
/// were a proxy: a grammar revision that ADDS one kind and REMOVES another
/// passes a count comparison silently while parsing differently. The digests
/// cannot: they pin the whole symbol table, name for name, in id order.
///
/// The ABI version is still reported but is NO LONGER part of the match, and
/// that is a deliberate, measured decision. ABI is a property of the
/// tree-sitter CLI that GENERATED the parser, not of the grammar -- so it
/// blocked a legitimate case:
///
/// > PyPI `tree-sitter-sql` 0.3.11 is ABI 15 and the crate `tree-sitter-sequel`
/// > 0.3.11 is ABI 14. All 729 kind names and all 54 field names are identical
/// > by id. Parsing all 3,442 files of postgres + sqlfluff with BOTH and
/// > comparing a preorder digest of EVERY node (kind, byte range, MISSING and
/// > ERROR flags) gives **3,442 identical trees out of 3,442 -- including all
/// > 2,797 files that contain ERROR nodes.** Error recovery is identical too.
///
/// Keeping ABI in the gate would reject that while adding no protection the
/// digests do not already give: two different grammar REVISIONS with identical
/// symbol tables (a precedence-only change) are invisible to ABI as well, since
/// ABI moves with the CLI and not the grammar. The real safety net is unchanged
/// and is per-language: every routed language is gated by a DIVERGENT-0 parity
/// run over real corpora.
pub fn grammar_fingerprints() -> Vec<(&'static str, u32, String, String)> {
    let langs: [(&'static str, tree_sitter::Language); 27] = [
        ("typescript", tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        ("tsx", tree_sitter_typescript::LANGUAGE_TSX.into()),
        ("javascript", tree_sitter_javascript::LANGUAGE.into()),
        ("python", tree_sitter_python::LANGUAGE.into()),
        ("java", tree_sitter_java::LANGUAGE.into()),
        ("csharp", tree_sitter_c_sharp::LANGUAGE.into()),
        ("c", tree_sitter_c::LANGUAGE.into()),
        ("cpp", tree_sitter_cpp::LANGUAGE.into()),
        ("php", tree_sitter_php::LANGUAGE_PHP.into()),
        ("bash", tree_sitter_bash::LANGUAGE.into()),
        ("go", tree_sitter_go::LANGUAGE.into()),
        ("rust", tree_sitter_rust::LANGUAGE.into()),
        ("ruby", tree_sitter_ruby::LANGUAGE.into()),
        ("kotlin", tree_sitter_kotlin_ng::LANGUAGE.into()),
        ("lua", tree_sitter_lua::LANGUAGE.into()),
        ("groovy", tree_sitter_groovy::LANGUAGE.into()),
        ("scala", tree_sitter_scala::LANGUAGE.into()),
        ("swift", tree_sitter_swift::LANGUAGE.into()),
        ("zig", tree_sitter_zig::LANGUAGE.into()),
        ("elixir", tree_sitter_elixir::LANGUAGE.into()),
        ("julia", tree_sitter_julia::LANGUAGE.into()),
        ("fortran", tree_sitter_fortran::LANGUAGE.into()),
        ("objc", tree_sitter_objc::LANGUAGE.into()),
        ("powershell", tree_sitter_powershell::LANGUAGE.into()),
        ("sql", tree_sitter_sequel::LANGUAGE.into()),
        ("ocaml", tree_sitter_ocaml::LANGUAGE_OCAML.into()),
        ("ocaml_interface", tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into()),
    ];
    langs
        .iter()
        .map(|(name, l)| {
            (
                *name,
                l.abi_version() as u32,
                names_digest((0..l.node_kind_count()).map(|i| l.node_kind_for_id(i as u16))),
                names_digest((0..=l.field_count()).map(|i| l.field_name_for_id(i as u16))),
            )
        })
        .collect()
}

/// SHA-256 over the grammar's names, `\0`-joined in id order, with a missing
/// name written as the empty string.
///
/// The Python side computes this byte for byte the same way; the two must be
/// edited together, so the recipe is spelled out rather than left implicit:
/// join with `\0`, hash the UTF-8 bytes, lowercase hex.
fn names_digest<'x>(names: impl Iterator<Item = Option<&'x str>>) -> String {
    use sha2::Digest as _;
    let joined = names
        .map(|n| n.unwrap_or(""))
        .collect::<Vec<&str>>()
        .join("\0");
    let mut h = sha2::Sha256::new();
    h.update(joined.as_bytes());
    format!("{:x}", h.finalize())
}

/// Parse a trivial TypeScript source to prove the grammar is linked and
/// ABI-compatible. Returns false rather than panicking: a failure here must make
/// `kernel.py` disable the kernel, not abort the build.
pub fn grammar_smoke_test() -> bool {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .is_err()
    {
        return false;
    }
    match parser.parse("const x: number = 1;", None) {
        Some(tree) => {
            let root = tree.root_node();
            root.kind() == "program" && !root.has_error()
        }
        None => false,
    }
}
