//! Entry parsing and representation.
//!
//! A `.weaver` file consists of YAML frontmatter delimited by `---` lines
//! followed by a weaver-lang template body:
//!
//! ```text
//! ---
//! id: dark_forest
//! name: Dark Forest Description
//! keywords: ["dark forest", "shadowed path"]
//! slot: context
//! fallback: [foundation]
//! priority: 100
//! ---
//! The dark forest looms ahead...
//! ```

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use weaver_lang::{CompiledExpr, CompiledTemplate};

use crate::assembler::Slot;
use crate::ContextWeaverError;

// ── Entry ───────────────────────────────────────────────────────────────

/// A single lorebook entry: metadata + compiled template.
#[derive(Clone)]
pub struct Entry {
    pub meta: EntryMeta,
    pub compiled: Arc<CompiledTemplate>,
    /// An optional weaver-based condition statement to further fine-tune activation.
    pub condition: Option<Arc<CompiledExpr>>,
    /// Compiled regex patterns, cached at parse time to avoid recompilation
    /// on every activation scan.
    pub compiled_regex: Vec<Regex>,
    /// Raw body source, preserved for diagnostics and re-serialization.
    pub source_body: String,
}

/// Structured metadata parsed from the YAML frontmatter.
///
/// Fields map directly to the frontmatter keys. Unknown keys are
/// preserved in `extensions` so plugins and community tools can
/// stash their own data without it being silently dropped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMeta {
    /// Unique identifier for this entry. Used in triggers and document refs.
    pub id: String,

    /// Human-readable display name.
    #[serde(default)]
    pub name: String,

    // ── Activation (Tier 1) ─────────────────────────────────────────
    /// Keywords that activate this entry when found in chat messages.
    /// Matching is case-insensitive by default.
    #[serde(default)]
    pub keywords: Vec<String>,

    /// Regex patterns for activation. Matched against recent messages.
    /// These are compiled and cached on the [`Entry`] struct at parse time.
    #[serde(default)]
    pub regex: Vec<String>,

    /// A weaver-based expression for further fine-tuning activation conditions.
    pub condition: Option<String>,

    /// How many recent messages to scan for keywords/regex.
    /// `None` means use the lorebook default.
    #[serde(default)]
    pub scan_depth: Option<usize>,

    /// If true, this entry is always active regardless of keywords.
    #[serde(default)]
    pub constant: bool,

    // ── Ordering & placement ────────────────────────────────────────
    /// Higher priority entries are evaluated first and take precedence
    /// in token budget conflicts.
    #[serde(default = "default_priority")]
    pub priority: i32,

    /// Target slot for this entry's output in the prompt.
    ///
    /// Standard slots form a gradient from deep background to immediate
    /// foreground: `preamble`, `foundation`, `context`, `reference`,
    /// `framing`, `guidance`, `emphasis`, `immediate`, `aftermath`.
    #[serde(default)]
    pub slot: Slot,

    /// Fallback slots to try if the primary slot is not available
    /// in the user's ContextDefinition template. Tried in order.
    /// If no fallback matches, the entry is silently dropped.
    #[serde(default)]
    pub fallback: Vec<Slot>,

    /// Tie-breaker for entries at the same slot and priority.
    #[serde(default = "default_insertion_order")]
    pub insertion_order: i32,

    // ── Behavior ────────────────────────────────────────────────────
    /// Whether this entry is enabled. Disabled entries are skipped entirely.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Once activated, stay active for this many turns even if keywords
    /// no longer match. 0 means re-evaluate every turn.
    #[serde(default)]
    pub sticky_turns: usize,

    /// Minimum turns between activations. Prevents rapid re-triggering.
    #[serde(default)]
    pub cooldown: usize,

    /// If set, this entry counts toward the named group's token budget
    /// rather than the global budget. Groups allow authors to say
    /// "these combat entries share a 500-token pool."
    #[serde(default)]
    pub group: Option<String>,

    /// Tags for organizational purposes and for other entries to query.
    #[serde(default)]
    pub tags: Vec<String>,

    // ── Extensions ──────────────────────────────────────────────────
    /// Catch-all for unknown frontmatter keys. Preserved on round-trip.
    #[serde(flatten)]
    pub extensions: std::collections::HashMap<String, serde_yaml::Value>,
}

fn default_priority() -> i32 {
    100
}
fn default_insertion_order() -> i32 {
    50
}
fn default_true() -> bool {
    true
}

// ── Parsing ─────────────────────────────────────────────────────────────

impl Entry {
    /// Parse a `.weaver` file from its raw contents.
    pub fn parse(source: &str, file_path: Option<&str>) -> Result<Self, ContextWeaverError> {
        let (frontmatter, body) =
            split_frontmatter(source).ok_or_else(|| ContextWeaverError::MetaParse {
                entry_path: file_path.unwrap_or("<unknown>").to_string(),
                message: "missing frontmatter delimiters (---)".to_string(),
            })?;

        let meta: EntryMeta =
            serde_yaml::from_str(frontmatter).map_err(|e| ContextWeaverError::MetaParse {
                entry_path: file_path.unwrap_or("<unknown>").to_string(),
                message: e.to_string(),
            })?;

        let compiled = CompiledTemplate::compile(body).map_err(|errors| {
            ContextWeaverError::TemplateParse {
                entry_id: meta.id.clone(),
                errors,
            }
        })?;

        let condition = meta
            .condition
            .as_ref()
            .map(|src| CompiledExpr::compile(src))
            .transpose()
            .map_err(|errors| ContextWeaverError::TemplateParse {
                entry_id: meta.id.clone(),
                errors,
            })?
            .map(Arc::new);

        // Compile regex patterns once at parse time
        let compiled_regex = meta
            .regex
            .iter()
            .filter_map(|pattern| match Regex::new(pattern) {
                Ok(re) => Some(re),
                Err(e) => {
                    // Log but don't fail — bad regexes are skipped
                    tracing::error!(
                        "warning: entry '{}': invalid regex '{}': {}",
                        meta.id, pattern, e
                    );
                    None
                }
            })
            .collect();

        Ok(Entry {
            meta,
            compiled: Arc::new(compiled),
            source_body: body.to_string(),
            condition,
            compiled_regex,
        })
    }

    /// Load and parse a `.weaver` file from disk.
    pub fn load(path: &Path) -> Result<Self, ContextWeaverError> {
        let source = std::fs::read_to_string(path)?;
        Self::parse(&source, path.to_str())
    }
}

/// Split source into (frontmatter, body) at the `---` delimiters.
///
/// Expects the file to start with `---\n`, have frontmatter content,
/// then `---\n`, then the template body.
fn split_frontmatter(source: &str) -> Option<(&str, &str)> {
    let s = source.strip_prefix("---")?;
    let s = s.strip_prefix('\n').or_else(|| s.strip_prefix("\r\n"))?;

    let end = s
        .find("\n---\n")
        .or_else(|| s.find("\r\n---\r\n"))
        .or_else(|| s.find("\n---\r\n"))?;

    let frontmatter = &s[..end];
    let rest = &s[end..];

    // Skip past the closing --- and its newline
    let body_start = rest.find("---").unwrap() + 3;
    let body = &rest[body_start..];
    let body = body
        .strip_prefix('\n')
        .or_else(|| body.strip_prefix("\r\n"))
        .unwrap_or(body);

    Some((frontmatter, body))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_entry() {
        let source = r#"---
id: test_entry
name: Test Entry
keywords: ["hello", "world"]
slot: foundation
priority: 50
---
Hello, {{char:name}}!
"#;
        let entry = Entry::parse(source, None).unwrap();
        assert_eq!(entry.meta.id, "test_entry");
        assert_eq!(entry.meta.keywords, vec!["hello", "world"]);
        assert_eq!(entry.meta.priority, 50);
        assert_eq!(entry.meta.slot, Slot::Foundation);
    }

    #[test]
    fn test_default_values() {
        let source = r#"---
id: minimal
---
content
"#;
        let entry = Entry::parse(source, None).unwrap();
        assert_eq!(entry.meta.priority, 100);
        assert!(entry.meta.enabled);
        assert!(entry.meta.keywords.is_empty());
        assert!(!entry.meta.constant);
        assert_eq!(entry.meta.slot, Slot::Context); // new default
        assert!(entry.meta.fallback.is_empty());
    }

    #[test]
    fn test_fallback_parsed() {
        let source = r#"---
id: with_fallback
slot: reference
fallback: [context, foundation]
---
content
"#;
        let entry = Entry::parse(source, None).unwrap();
        assert_eq!(entry.meta.slot, Slot::Reference);
        assert_eq!(entry.meta.fallback, vec![Slot::Context, Slot::Foundation]);
    }

    #[test]
    fn test_regex_compiled_at_parse_time() {
        let source = r#"---
id: regex_entry
regex: ['\b(attack|fight)\b', '\d{3,}']
---
content
"#;
        let entry = Entry::parse(source, None).unwrap();
        assert_eq!(entry.compiled_regex.len(), 2);
        assert!(entry.compiled_regex[0].is_match("attack now"));
        assert!(entry.compiled_regex[1].is_match("found 1000 gold"));
    }

    #[test]
    fn test_invalid_regex_skipped() {
        let source = r#"---
id: bad_regex
regex: ['[invalid', '\d+']
---
content
"#;
        let entry = Entry::parse(source, None).unwrap();
        // The invalid regex is skipped, only the valid one is kept
        assert_eq!(entry.compiled_regex.len(), 1);
        assert!(entry.compiled_regex[0].is_match("42"));
    }

    #[test]
    fn test_extensions_preserved() {
        let source = r#"---
id: extended
my_custom_field: "hello"
plugin_data:
  foo: bar
---
content
"#;
        let entry = Entry::parse(source, None).unwrap();
        assert!(entry.meta.extensions.contains_key("my_custom_field"));
        assert!(entry.meta.extensions.contains_key("plugin_data"));
    }

    #[test]
    fn test_missing_frontmatter_errors() {
        let result = Entry::parse("no frontmatter here", None);
        assert!(matches!(result, Err(ContextWeaverError::MetaParse { .. })));
    }
}
