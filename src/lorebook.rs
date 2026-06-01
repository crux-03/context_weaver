//! Lorebook: a collection of entries with shared configuration.
//!
//! A lorebook lives on disk as a directory:
//!
//! ```text
//! my_character/
//!   lorebook.yaml       # book-level config
//!   entries/
//!     dark_forest.weaver
//!     combat_system.weaver
//!     npc_merchant.weaver
//! ```
//!
//! The `lorebook.yaml` declares namespaces, defaults, and metadata for the
//! entire book. Individual entries inherit defaults from the book config
//! and can override them in their own frontmatter.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::ContextWeaverError;
use crate::assembler::Slot;
use crate::entry::Entry;
use crate::host::NamespaceConfig;

// ── Lorebook ────────────────────────────────────────────────────────────

/// Stable handle identifying a lorebook within a set. Assigned by
/// registration order (the first book added is `BookId(0)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BookId(pub usize);

/// An ordered collection of lorebooks evaluated together.
///
/// A set always contains at least one book — the *primary* book at
/// `BookId(0)`. Additional books are appended via [`add`](LorebookSet::add)
/// and assigned sequential ids. Iteration order equals id order, which is
/// also the order [`BookTemplates`](crate::resolver::BookTemplates) uses for
/// the global-fallback scan — keeping the two id spaces aligned.
pub struct LorebookSet {
    books: Vec<Lorebook>,
}

impl LorebookSet {
    /// Create a set containing a single primary book.
    pub fn single(book: Lorebook) -> Self {
        Self { books: vec![book] }
    }

    pub fn get(&self, id: BookId) -> Option<&Lorebook> {
        self.books.get(id.0)
    }

    /// Append a book, returning its assigned [`BookId`].
    pub fn add(&mut self, book: Lorebook) -> BookId {
        let id = BookId(self.books.len());
        self.books.push(book);
        id
    }

    /// The primary book (`BookId(0)`). Always present.
    pub fn primary(&self) -> &Lorebook {
        &self.books[0]
    }

    /// Iterate books with their ids, in registration order.
    pub fn iter(&self) -> impl Iterator<Item = (BookId, &Lorebook)> {
        self.books.iter().enumerate().map(|(i, b)| (BookId(i), b))
    }
}

pub struct Lorebook {
    pub config: LorebookConfig,
    entries: HashMap<String, Entry>,
    /// Entries sorted by priority (descending) for evaluation ordering.
    eval_order: Vec<String>,
}

impl Lorebook {
    /// Create an empty lorebook with default configuration.
    pub fn new() -> Self {
        Self {
            config: LorebookConfig::default(),
            entries: HashMap::new(),
            eval_order: Vec::new(),
        }
    }

    /// Load a lorebook from a directory on disk.
    ///
    /// Expects `lorebook.yaml` at the root and `.weaver` files in an
    /// `entries/` subdirectory (or directly in the root).
    pub fn load_from_directory(path: impl AsRef<Path>) -> Result<Self, ContextWeaverError> {
        let root = path.as_ref();

        // Load book config
        let config_path = root.join("lorebook.yaml");
        let config = if config_path.exists() {
            let raw = std::fs::read_to_string(&config_path)?;
            serde_yaml::from_str(&raw).map_err(|e| ContextWeaverError::MetaParse {
                entry_path: config_path.display().to_string(),
                message: e.to_string(),
            })?
        } else {
            LorebookConfig::default()
        };

        let mut lorebook = Self {
            config,
            entries: HashMap::new(),
            eval_order: Vec::new(),
        };

        // Scan for .weaver files
        let entries_dir = root.join("entries");
        let scan_dir = if entries_dir.is_dir() {
            &entries_dir
        } else {
            root
        };

        for dir_entry in std::fs::read_dir(scan_dir)? {
            let dir_entry = dir_entry?;
            let file_path = dir_entry.path();
            if file_path.extension().is_some_and(|ext| ext == "weaver") {
                let entry = Entry::load(&file_path)?;
                lorebook.add_entry(entry);
            }
        }

        lorebook.rebuild_eval_order();
        Ok(lorebook)
    }

    /// Add an entry to the lorebook. Replaces any existing entry with
    /// the same ID.
    pub fn add_entry(&mut self, entry: Entry) {
        let id = entry.meta.id.clone();
        self.entries.insert(id, entry);
        self.rebuild_eval_order();
    }

    /// Remove an entry by ID.
    pub fn remove_entry(&mut self, id: &str) -> Option<Entry> {
        let entry = self.entries.remove(id);
        if entry.is_some() {
            self.rebuild_eval_order();
        }
        entry
    }

    /// Look up an entry by ID.
    pub fn get_entry(&self, id: &str) -> Option<&Entry> {
        self.entries.get(id)
    }

    /// Iterate all entries in evaluation order (highest priority first).
    pub fn entries_in_order(&self) -> impl Iterator<Item = &Entry> {
        self.eval_order.iter().filter_map(|id| self.entries.get(id))
    }

    /// Iterate all enabled entries (skips disabled).
    pub fn active_entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries_in_order().filter(|e| e.meta.enabled)
    }

    /// Get all entry IDs.
    pub fn entry_ids(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    fn rebuild_eval_order(&mut self) {
        let mut ids: Vec<_> = self.entries.keys().cloned().collect();
        ids.sort_by(|a, b| {
            let ea = &self.entries[a].meta;
            let eb = &self.entries[b].meta;
            eb.priority
                .cmp(&ea.priority)
                .then(ea.insertion_order.cmp(&eb.insertion_order))
                .then(ea.id.cmp(&eb.id))
        });
        self.eval_order = ids;
    }
}

impl Default for Lorebook {
    fn default() -> Self {
        Self::new()
    }
}

// ── Book-level config ───────────────────────────────────────────────────

/// Configuration for the entire lorebook, parsed from `lorebook.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LorebookConfig {
    /// Display name for this lorebook.
    #[serde(default)]
    pub name: String,

    /// Description / author notes.
    #[serde(default)]
    pub description: String,

    /// Namespace declarations and access control.
    /// The host uses these to configure the WeaverHost's permissions.
    #[serde(default)]
    pub namespaces: HashMap<String, NamespaceConfig>,

    // ── Defaults (inherited by entries that don't override) ──────────
    /// Default number of recent messages to scan for keywords.
    #[serde(default = "default_scan_depth")]
    pub default_scan_depth: usize,

    /// Default entry priority.
    #[serde(default = "default_priority")]
    pub default_priority: i32,

    /// Default slot for entries that don't specify one.
    #[serde(default)]
    pub default_slot: Slot,

    /// Total token budget for all activated entries combined.
    /// `None` means unlimited.
    #[serde(default)]
    pub token_budget: Option<usize>,

    /// Named group budgets. Entries with a matching `group` field
    /// draw from their group's pool instead of the global budget.
    #[serde(default)]
    pub group_budgets: HashMap<String, usize>,

    /// Whether keyword matching is case-sensitive. Default: false.
    #[serde(default)]
    pub case_sensitive_keywords: bool,

    // ── Extensions ──────────────────────────────────────────────────
    #[serde(flatten)]
    pub extensions: HashMap<String, serde_yaml::Value>,
}

fn default_scan_depth() -> usize {
    10
}
fn default_priority() -> i32 {
    100
}

impl Default for LorebookConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            namespaces: HashMap::new(),
            default_scan_depth: default_scan_depth(),
            default_priority: default_priority(),
            default_slot: Slot::default(),
            token_budget: None,
            group_budgets: HashMap::new(),
            case_sensitive_keywords: false,
            extensions: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_assigns_sequential_ids() {
        let mut set = LorebookSet::single(Lorebook::new());
        let b1 = set.add(Lorebook::new());
        let b2 = set.add(Lorebook::new());
        assert_eq!(b1, BookId(1));
        assert_eq!(b2, BookId(2));

        let ids: Vec<BookId> = set.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![BookId(0), BookId(1), BookId(2)]);
    }
}
