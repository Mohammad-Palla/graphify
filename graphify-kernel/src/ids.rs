//! Node-id construction, byte-exact with `graphify.ids.make_id` -- for ASCII.
//!
//! Python's `normalize_id` iterates `casefold()` then NFKC to a fixpoint and then
//! filters on `re`'s Unicode `\w`. Reproducing that in Rust is possible but it is
//! exactly the kind of thing that goes subtly wrong for a handful of files and
//! never raises: a Turkish dotted capital, a combining mark, a full-width digit.
//! The consequence would be a differently-keyed node -- silent, and invisible to
//! anything but a byte-comparison of the whole graph.
//!
//! So this module does not attempt it. Every entry point returns `None` the moment
//! it sees a byte >= 0x80, and the caller defers that file to Python. On ASCII the
//! equivalence is provable rather than tested-and-hoped:
//!
//!   * `casefold()` on ASCII is `to_ascii_lowercase()`;
//!   * NFKC on ASCII is the identity, so the fixpoint loop runs once;
//!   * `\w` on ASCII is exactly `[A-Za-z0-9_]`.
//!
//! Real corpora are overwhelmingly ASCII, so the deferral cost is small -- and it
//! is *measured* by `harness/kernel_parity.py` rather than assumed.

/// `graphify.ids.normalize_id`, ASCII only. `None` means "contains non-ASCII,
/// defer to Python".
pub fn normalize_id_ascii(s: &str) -> Option<String> {
    if !s.is_ascii() {
        return None;
    }
    // [^\w]+ -> "_", then _+ -> "_", then strip "_". Both collapses are the same
    // operation once the first has run, so one pass with a "last char was an
    // underscore" latch produces the identical result.
    let mut out = String::with_capacity(s.len());
    let mut pending_underscore = false;
    for &b in s.as_bytes() {
        // NOT `|| b == b'_'`. Python applies TWO substitutions: `[^\w]+ -> "_"`
        // (underscore is a word char, so it survives) and then `_+ -> "_"`, which
        // collapses the runs the first pass just created together with any
        // underscores already present. Composed, the rule is simply: every maximal
        // run of non-alphanumerics becomes one underscore. Treating `_` as a word
        // char here left `a__b` uncollapsed.
        let is_word = b.is_ascii_alphanumeric();
        if is_word {
            // Only emit a separator once we know a word char follows it, which
            // gives the leading-strip for free.
            if pending_underscore && !out.is_empty() {
                out.push('_');
            }
            pending_underscore = false;
            out.push(b.to_ascii_lowercase() as char);
        } else {
            pending_underscore = true;
        }
    }
    // A trailing run of non-word chars leaves `pending_underscore` set and is
    // dropped, which is the trailing strip.
    Some(out)
}

/// `graphify.extractors.base._make_id` / `graphify.ids.make_id`, ASCII only.
///
/// Mirrors Python's `"_".join(p.strip("_.") for p in parts if p)` exactly,
/// including the detail that the emptiness filter runs on the ORIGINAL part and
/// the strip runs after: `make_id("a", "__", "b")` joins to `"a__b"`, not
/// `"a_b"`, before normalization collapses it.
pub fn make_id_ascii(parts: &[&str]) -> Option<String> {
    let mut joined = String::new();
    let mut first = true;
    for p in parts {
        if p.is_empty() {
            continue; // `if p` in the generator: filters the raw part
        }
        if !first {
            joined.push('_');
        }
        first = false;
        joined.push_str(p.trim_matches(|c| c == '_' || c == '.'));
    }
    normalize_id_ascii(&joined)
}

/// `graphify.extractors.base._file_stem`: `path.with_suffix("").as_posix()`.
///
/// `None` means "pathlib would normalize this path, defer to Python". pathlib
/// does more than string surgery -- it drops `.` components (`a/.` -> `a`),
/// collapses `//`, and strips trailing slashes -- and reimplementing that is
/// surface we do not need: real source paths arrive from a directory walk and are
/// already clean. Differential testing over 25,841 Bun paths found exactly two
/// disagreements, both pathological (`.` and `....`), and both are now deferrals
/// rather than guesses.
///
/// The suffix rule itself is pathlib's, which is NOT "everything after the last
/// dot": the last dot counts only when `0 < i < len(name) - 1`, so `.gitignore`,
/// `a.` and `....` all have no suffix. An earlier version searched backwards for
/// the first dot satisfying that test, which turned `....` into `..`; pathlib
/// looks at the last dot only and gives up if it fails.
pub fn file_stem(path: &str) -> Option<String> {
    // A backslash is NOT a separator here. `_file_stem` uses `pathlib.Path`, which
    // is `PosixPath` on Linux (backslash is an ordinary filename character) and
    // `WindowsPath` on Windows (backslash IS a separator) -- so the correct answer
    // depends on the host, and guessing either way is wrong on the other.
    // Normalizing `\` to `/` here produced 2364 disagreements under fuzzing while
    // the entire 25,841-path Bun corpus showed none: real POSIX paths never
    // contain one. Defer instead of guessing.
    if path.contains('\\') {
        return None;
    }
    let posix = path.to_string();
    if posix.is_empty() || posix.ends_with('/') || posix.contains("//") {
        return None;
    }
    if posix.split('/').any(|c| c == "." || c == "..") {
        return None;
    }
    let (dir, name) = match posix.rfind('/') {
        Some(i) => (posix[..=i].to_string(), posix[i + 1..].to_string()),
        None => (String::new(), posix.clone()),
    };
    if name.is_empty() {
        return None;
    }
    match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => Some(format!("{dir}{}", &name[..i])),
        _ => Some(posix),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_ascii_defers() {
        assert!(normalize_id_ascii("café").is_none());
        assert!(make_id_ascii(&["caf\u{e9}", "x"]).is_none());
    }

    #[test]
    fn collapses_and_strips() {
        assert_eq!(normalize_id_ascii("__a--b__").unwrap(), "a_b");
        assert_eq!(normalize_id_ascii("///").unwrap(), "");
        assert_eq!(normalize_id_ascii("A.B/C").unwrap(), "a_b_c");
    }

    #[test]
    fn join_filters_raw_part_then_strips() {
        assert_eq!(make_id_ascii(&["a", "__", "b"]).unwrap(), "a_b");
        assert_eq!(make_id_ascii(&["a", "", "b"]).unwrap(), "a_b");
        assert_eq!(make_id_ascii(&["_x_", "y"]).unwrap(), "x_y");
    }

    #[test]
    fn stem_follows_pathlib() {
        assert_eq!(file_stem("/tmp/a/sample.ts").unwrap(), "/tmp/a/sample");
        assert_eq!(file_stem("a.b.c").unwrap(), "a.b");
        assert_eq!(file_stem("/x/.gitignore").unwrap(), "/x/.gitignore");
        assert_eq!(file_stem("/x/noext").unwrap(), "/x/noext");
        // pathlib looks at the LAST dot only; `....` has no suffix.
        assert_eq!(file_stem("....").unwrap(), "....");
        assert_eq!(file_stem("a.").unwrap(), "a.");
        assert_eq!(file_stem("..a").unwrap(), ".");
    }

    #[test]
    fn paths_needing_pathlib_normalization_defer() {
        for p in [".", "..", "a/.", "a/..", "", "/", "a//b", "a/", "a\\b"] {
            assert!(file_stem(p).is_none(), "{p} should defer");
        }
    }
}

/// `Path(path).parent.name` -- the immediate parent DIRECTORY's name, or `""`
/// for a top-level file.
///
/// Go uses it as the package scope, so methods on one type across several files
/// of a package share a canonical type node. It carries the same backslash
/// caveat as `file_stem` and defers for the same reason.
pub fn parent_name(path: &str) -> Option<String> {
    if path.contains('\\') {
        return None;
    }
    if path.is_empty() || path.ends_with('/') || path.contains("//") {
        return None;
    }
    if path.split('/').any(|c| c == "." || c == "..") {
        return None;
    }
    let dir = match path.rfind('/') {
        // `Path("a/b.go").parent` is `a`; `Path("/b.go").parent` is `/`, whose
        // `.name` is "" -- the leading-slash case falls out of the empty slice.
        Some(i) => &path[..i],
        None => "",
    };
    Some(match dir.rfind('/') {
        Some(i) => dir[i + 1..].to_string(),
        None => dir.to_string(),
    })
}
