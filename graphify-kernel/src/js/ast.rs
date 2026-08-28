//! Thin tree-sitter helpers that match Graphify's Python accessors exactly.

use tree_sitter::Node;

/// `_read_text(node, source)`.
///
/// Python decodes with `errors="replace"`, so invalid UTF-8 becomes U+FFFD
/// rather than raising. Rust's `from_utf8` cannot express that without
/// allocating, and a byte range that is not valid UTF-8 would produce a
/// different string on the two sides -- so callers treat the lossy case as a
/// deferral via [`text_checked`] and this function is only used where the range
/// has already been proven valid.
pub fn text<'a>(node: Node, src: &'a [u8]) -> &'a str {
    std::str::from_utf8(&src[node.byte_range()]).unwrap_or("")
}

/// `_read_text`, but `None` when the bytes are not valid UTF-8 -- Python would
/// substitute U+FFFD and we would silently produce a different name.
pub fn text_checked<'a>(node: Node, src: &'a [u8]) -> Option<&'a str> {
    std::str::from_utf8(&src[node.byte_range()]).ok()
}

/// `node.start_point[0] + 1`.
pub fn line_of(node: Node) -> usize {
    node.start_position().row + 1
}

/// `node.children` as a plain Vec.
///
/// tree-sitter's Rust API needs a `TreeCursor` borrowed for the whole iteration,
/// which makes recursive walks that also mutate shared state awkward. Children
/// counts are small (a few dozen at most), so materializing them keeps the port
/// readable and matches Python's list semantics, including that Python's `walk`
/// iterates a snapshot.
pub fn children<'tree>(node: Node<'tree>) -> Vec<Node<'tree>> {
    let mut cur = node.walk();
    node.children(&mut cur).collect()
}

/// `[c for c in node.children if c.is_named]`.
pub fn named_children<'tree>(node: Node<'tree>) -> Vec<Node<'tree>> {
    let mut cur = node.walk();
    node.children(&mut cur).filter(|c| c.is_named()).collect()
}

/// Python's `str.strip(chars)`: trim any leading/trailing char in the set.
pub fn strip_chars(s: &str, chars: &str) -> String {
    s.trim_matches(|c| chars.contains(c)).to_string()
}
