//! `_KOTLIN_BUILTIN_TYPES`, generated from `engine.py` so the two cannot drift.
//!
//! Kotlin compiles to the JVM and freely references `java.*` types, so this is
//! combined with `java::consts::BUILTIN_TYPES` at the call site rather than
//! duplicating those 180 names here.

/// 64 names never emitted as type references.
pub const BUILTIN_TYPES: &[&str] = &[
    "Any", "Unit", "Nothing", "Boolean", "Byte", "Short",
    "Int", "Long", "Float", "Double", "Char", "String",
    "CharSequence", "Number", "Comparable", "Enum", "Annotation", "Pair",
    "Triple", "Lazy", "Function", "Throwable", "Exception", "RuntimeException",
    "Error", "IllegalArgumentException", "IllegalStateException", "NullPointerException", "IndexOutOfBoundsException", "ClassCastException",
    "NumberFormatException", "ArithmeticException", "UnsupportedOperationException", "NoSuchElementException", "ConcurrentModificationException", "StackOverflowError",
    "OutOfMemoryError", "AssertionError", "InterruptedException", "Array", "List", "MutableList",
    "ArrayList", "Set", "MutableSet", "HashSet", "LinkedHashSet", "Map",
    "MutableMap", "HashMap", "LinkedHashMap", "Collection", "MutableCollection", "Iterable",
    "MutableIterable", "Iterator", "MutableIterator", "ListIterator", "MutableListIterator", "Sequence",
    "Comparator", "Regex", "MatchResult", "StringBuilder",
];
