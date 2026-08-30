//! `graphify.security.sanitize_metadata`, reproduced.
//!
//! Every `metadata` block on a node or edge passes through this in Python, so a
//! native walker that skipped it would emit a *differently escaped* dict for any
//! value containing `& < > " '` -- which C# reaches routinely, because a `using`
//! alias target (`using L = List<int>;`) and a generic ref qualifier both carry
//! angle brackets.
//!
//! The three rules, in Python's order: strip control characters, HTML-escape
//! with `quote=True`, then cap at 512 CHARACTERS (not bytes -- the Python slices
//! a `str`). Keys get the same treatment and are dropped when they sanitize to
//! empty. Lists are capped at 50 items. Bools, numbers and None pass through;
//! `bool` is checked before `int` in the Python because it is a subclass, which
//! is moot here since `Val` keeps them apart by construction.

use crate::js::emit::Val;

const MAX_VALUE_LEN: usize = 512;
const MAX_LIST_ITEMS: usize = 50;

/// `_sanitize_metadata_string`.
pub fn sanitize_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        // `_CONTROL_CHAR_RE = [\x00-\x1f\x7f]`, applied BEFORE the escape.
        if (c as u32) <= 0x1f || c as u32 == 0x7f {
            continue;
        }
        // `html.escape(quote=True)`, in the order Python applies it: `&` first,
        // so the ampersands it introduces are not re-escaped.
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(c),
        }
    }
    if out.chars().count() > MAX_VALUE_LEN {
        return out.chars().take(MAX_VALUE_LEN).collect();
    }
    out
}

fn sanitize_value(v: Val) -> Val {
    match v {
        Val::S(s) => Val::S(sanitize_string(&s)),
        Val::Static(s) => Val::S(sanitize_string(s)),
        Val::Meta(pairs) => Val::Meta(sanitize(pairs)),
        Val::List(items) => Val::List(
            items
                .into_iter()
                .take(MAX_LIST_ITEMS)
                .map(sanitize_value)
                .collect(),
        ),
        // bool / float / None pass through untouched.
        other => other,
    }
}

/// `sanitize_metadata`: sanitize each key, drop the entry when the key
/// sanitizes to empty, and sanitize the value.
pub fn sanitize(pairs: Vec<(String, Val)>) -> Vec<(String, Val)> {
    let mut out = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        let clean_key = sanitize_string(&k);
        if clean_key.is_empty() {
            continue;
        }
        out.push((clean_key, sanitize_value(v)));
    }
    out
}
