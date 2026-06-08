//! Host context implementation for ContextWeaver.
//!
//! [`WeaverHost`] implements weaver-lang's [`EvalContext`] trait, providing:
//!
//! - **Namespace management** with configurable read/write access
//! - **Read-only enforcement** for host-provided variables
//! - **Trigger handling** that records activations without producing output
//! - **Document resolution** with recursive entry evaluation and cycle detection

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use weaver_lang::Value;
use weaver_lang::{CompiledTemplate, EvalContext, EvalError, EvalErrorKind, Registry};

use crate::lorebook::BookId;
use crate::lorebook::LorebookConfig;
use crate::resolver::{BookTemplates, DefaultIdResolver, IdResolver};

// ── Namespace access control ────────────────────────────────────────────

/// Access level for a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NamespaceAccess {
    /// Templates can read but not write. The host populates these.
    ReadOnly,
    /// Templates can both read and write.
    ReadWrite,
}

/// Configuration for a single namespace, from `lorebook.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceConfig {
    pub access: NamespaceAccess,
    #[serde(default)]
    pub description: String,
}

// ── WeaverHost ──────────────────────────────────────────────────────────

/// The evaluation context for ContextWeaver.
///
/// Manages variable storage across namespaces, enforces access control,
/// and handles trigger/document resolution with cycle detection.
pub struct WeaverHost {
    /// Host-reserved namespaces: uniform across all books, not declarable
    /// or overridable by any book. Populated via `reserve_namespace`.
    reserved_namespaces: HashMap<String, NamespaceAccess>,
    /// Per-book namespace access from book declarations. A book's
    /// declaration governs only that book's own access. A scope in neither
    /// map is invalid.
    namespace_access: HashMap<String, HashMap<BookId, NamespaceAccess>>,

    /// Variable storage: namespace → (name → value).
    variables: HashMap<String, HashMap<String, Value>>,

    /// Persistent state that survives across evaluation passes.
    /// This is the `state:` namespace — it gets serialized between sessions.
    persistent_state: HashMap<String, Value>,

    /// Set of entry IDs currently being evaluated — used for cycle detection
    /// in recursive document resolution.
    eval_stack: Vec<(BookId, String)>,

    /// Maximum recursion depth for document chains.
    max_recursion_depth: usize,

    /// Set of entry IDs that are currently active (populated before evaluation).
    active_entries: HashSet<(BookId, String)>,

    /// The entry currently being evaluated (for diagnostics).
    current_entry: Option<String>,

    /// Compiled entry templates, partitioned by book. Replaced each
    /// evaluation pass so `resolve_document` can evaluate entries inline.
    book_templates: BookTemplates,

    /// Entry IDs activated via `<trigger>` during the current evaluation
    /// pass. The engine drains this after evaluation to feed the next
    /// activation pass.
    triggered_entries: Vec<(BookId, String)>,

    /// Strategy for mapping a document id to its template. Defaults to a
    /// direct lookup; a host may override it via `set_id_resolver`.
    resolver: Box<dyn IdResolver>,
}

impl WeaverHost {
    /// Create a WeaverHost from lorebook configuration.
    pub fn from_lorebook_config(book: BookId, config: &LorebookConfig) -> Self {
        let mut namespace_access: HashMap<String, HashMap<BookId, NamespaceAccess>> =
            HashMap::new();
        for (name, cfg) in &config.namespaces {
            namespace_access
                .entry(name.clone())
                .or_default()
                .insert(book, cfg.access);
        }

        Self {
            reserved_namespaces: HashMap::new(),
            namespace_access,
            variables: HashMap::new(),
            persistent_state: HashMap::new(),
            eval_stack: Vec::new(),
            max_recursion_depth: 10,
            active_entries: HashSet::new(),
            current_entry: None,
            book_templates: BookTemplates::new(),
            triggered_entries: Vec::new(),
            resolver: Box::new(DefaultIdResolver),
        }
    }

    /// Reserve a namespace with fixed access, uniform across all books.
    /// Reserved namespaces win over any book declaration and cannot be
    /// changed by a book.
    pub fn reserve_namespace(&mut self, name: impl Into<String>, access: NamespaceAccess) {
        self.reserved_namespaces.insert(name.into(), access);
    }

    /// Record an additional book's namespace declarations. Each governs only
    /// that book; declarations of reserved names are recorded but never win.
    pub fn add_book_namespaces(
        &mut self,
        book: BookId,
        namespaces: &HashMap<String, NamespaceConfig>,
    ) {
        for (name, cfg) in namespaces {
            self.namespace_access
                .entry(name.clone())
                .or_default()
                .insert(book, cfg.access);
        }
    }

    // ── Host-side variable access (bypasses access control) ─────────

    /// Set a variable from the host side (bypasses access control).
    ///
    /// This is how the host application feeds character data, user info,
    /// and chat metadata into the evaluation context.
    pub fn set_host_variable(&mut self, scope: &str, name: &str, value: Value) {
        // Mirror to persistent state so host-set state variables survive
        // serialization round-trips and are visible via persistent_state()
        if scope == "state" {
            self.persistent_state
                .insert(name.to_string(), value.clone());
        }
        self.variables
            .entry(scope.to_string())
            .or_default()
            .insert(name.to_string(), value);
    }

    /// Bulk-set all variables in a namespace from the host.
    pub fn set_namespace(&mut self, scope: &str, vars: HashMap<String, Value>) {
        self.variables.insert(scope.to_string(), vars);
    }

    // ── Active entry tracking ───────────────────────────────────────

    /// Mark a set of entry IDs as active (called before evaluation).
    ///
    /// The `is_active` / `is_active_global` commands read this set by
    /// downcasting the evaluation context back to `WeaverHost`.
    pub fn set_active_entries(&mut self, ids: HashSet<(BookId, String)>) {
        self.active_entries = ids;
    }

    /// Check if an entry is active in *any* book (the global view).
    pub fn is_entry_active(&self, id: &str) -> bool {
        self.active_entries.iter().any(|(_, eid)| eid == id)
    }

    /// Check if an entry is active in the *local* book — the book of the
    /// entry currently being evaluated (top of the eval stack). With no
    /// frame on the stack there is no local book, so this is always false.
    pub fn is_entry_active_local(&self, id: &str) -> bool {
        self.eval_stack
            .last()
            .map(|(book, _)| self.active_entries.contains(&(*book, id.to_string())))
            .unwrap_or(false)
    }

    // ── Entry template management ───────────────────────────────────

    /// Provide templates for a single book (wraps them as book 0).
    /// Convenience for single-book callers and tests.
    pub fn set_entry_templates(&mut self, templates: HashMap<String, Arc<CompiledTemplate>>) {
        let mut books = BookTemplates::new();
        books.push(templates);
        self.book_templates = books;
    }

    /// Provide book-partitioned templates for document resolution.
    /// Called by the engine before each evaluation pass.
    pub fn set_book_templates(&mut self, templates: BookTemplates) {
        self.book_templates = templates;
    }

    // ── Trigger collection ──────────────────────────────────────────

    /// Drain the list of entry IDs that were triggered during evaluation.
    ///
    /// Called by the engine after an evaluation pass. The returned IDs
    /// become candidates for the next activation pass, subject to
    /// cooldown, budget, and other constraints.
    pub fn drain_triggered_entries(&mut self) -> Vec<(BookId, String)> {
        std::mem::take(&mut self.triggered_entries)
    }

    // ── Eval stack management ───────────────────────────────────────

    /// Called before evaluating an entry — pushes onto the eval stack
    /// for cycle detection.
    pub fn begin_entry(&mut self, book: BookId, entry_id: &str) {
        self.current_entry = Some(entry_id.to_string());
        self.eval_stack.push((book, entry_id.to_string()));
    }

    /// Called after evaluating an entry — pops from the eval stack.
    pub fn end_entry(&mut self) {
        self.eval_stack.pop();
        self.current_entry = self.eval_stack.last().map(|(_, id)| id.clone());
    }

    // ── Persistent state ────────────────────────────────────────────

    /// Get the persistent state map (for serialization between sessions).
    #[deprecated(
        since = "0.2.1",
        note = "Superseded by `WeaverHost::export_persistent`"
    )]
    pub fn persistent_state(&self) -> &HashMap<String, Value> {
        &self.persistent_state
    }

    /// Restore persistent state (e.g. loaded from a save file).
    ///
    /// Note: this only restores the `state` namespace. For a full snapshot
    /// covering every writable namespace, prefer
    /// [`restore_persistent`](Self::restore_persistent).
    #[deprecated(
        since = "0.2.1",
        note = "Superseded by `WeaverHost::restore_persistent`"
    )]
    pub fn restore_persistent_state(&mut self, state: HashMap<String, Value>) {
        self.persistent_state = state;
        // Mirror into the variables map so templates can access it
        self.variables
            .insert("state".to_string(), self.persistent_state.clone());
    }

    /// Snapshot every persistable namespace for serialization between
    /// sessions, as `namespace → (name → value)`.
    ///
    /// Includes the `state` namespace plus any other namespace a template
    /// could have written to — i.e. every scope that is `ReadWrite` for at
    /// least one book, or reserved `ReadWrite`. Excludes:
    ///
    /// - the transient `temp` scope (cleared every turn by
    ///   [`clear_transient`](Self::clear_transient)), and
    /// - host-provided `ReadOnly` scopes (`char`, `user`, …), which the host
    ///   re-supplies each session and which must not be clobbered by a stale
    ///   saved copy.
    pub fn export_persistent(&self) -> HashMap<String, HashMap<String, Value>> {
        let mut out: HashMap<String, HashMap<String, Value>> = HashMap::new();
        for (scope, vars) in &self.variables {
            if vars.is_empty() || !self.is_scope_persistable(scope) {
                continue;
            }
            out.insert(scope.clone(), vars.clone());
        }
        // `state` is also mirrored in `persistent_state`; make sure anything
        // there is represented even if it never landed in `variables`.
        if !self.persistent_state.is_empty() {
            out.entry("state".to_string()).or_default().extend(
                self.persistent_state
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone())),
            );
        }
        out
    }

    /// Restore a full multi-namespace snapshot produced by
    /// [`export_persistent`](Self::export_persistent).
    ///
    /// Each scope's stored variables are replaced wholesale. The `state`
    /// namespace is additionally mirrored back into `persistent_state` so the
    /// two stores stay in sync.
    pub fn restore_persistent(&mut self, snapshot: HashMap<String, HashMap<String, Value>>) {
        for (scope, vars) in snapshot {
            if scope == "state" {
                self.persistent_state = vars.clone();
            }
            self.variables.insert(scope, vars);
        }
    }

    /// Whether a scope's contents should be persisted: it is writable by some
    /// book or reserved `ReadWrite` (so a template could have mutated it), and
    /// it is not the transient `temp` namespace.
    /// 
    /// Additionally namespaces that starts with an underscore are considered
    /// non-persistent scopes by convention.
    fn is_scope_persistable(&self, scope: &str) -> bool {
        if scope == "temp" || scope.starts_with('_') {
            return false;
        }
        if let Some(access) = self.reserved_namespaces.get(scope) {
            return *access == NamespaceAccess::ReadWrite;
        }
        self.namespace_access
            .get(scope)
            .is_some_and(|per_book| per_book.values().any(|a| *a == NamespaceAccess::ReadWrite))
    }

    /// Clear temporary (non-persistent) variables. Called between turns.
    pub fn clear_transient(&mut self) {
        // Keep: state, host-provided namespaces
        // Clear: temp, triggered entries
        self.variables.remove("temp");
        self.triggered_entries.clear();
    }

    /// Set the maximum recursion depth for document resolution chains.
    pub fn set_max_recursion_depth(&mut self, depth: usize) {
        self.max_recursion_depth = depth;
    }

    /// Install a custom [`IdResolver`], replacing the default direct lookup.
    ///
    /// Affects every `[[id]]` document reference resolved after this call.
    pub fn set_id_resolver(&mut self, resolver: Box<dyn IdResolver>) {
        self.resolver = resolver;
    }

    /// Resolve access for `scope` from `book`'s perspective. Reserved wins
    /// and is uniform; otherwise the calling book's own declaration. `None`
    /// means the scope is undeclared — invalid.
    fn namespace_access_for(&self, scope: &str, book: Option<BookId>) -> Option<NamespaceAccess> {
        if let Some(access) = self.reserved_namespaces.get(scope) {
            return Some(*access);
        }
        book.and_then(|b| self.namespace_access.get(scope)?.get(&b).copied())
    }
}

// ── EvalContext implementation ───────────────────────────────────────────

impl EvalContext for WeaverHost {
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        // ── Undeclared scopes are invalid: reads yield nothing ──────
        let book = self.eval_stack.last().map(|(b, _)| *b);
        if self.namespace_access_for(scope, book).is_none() {
            return Ok(None);
        }

        // ── Stored variables ────────────────────────────────────────
        if let Some(ns) = self.variables.get(scope) {
            if let Some(val) = ns.get(name) {
                return Ok(Some(val.clone()));
            }
        }
        if scope == "state" {
            if let Some(val) = self.persistent_state.get(name) {
                return Ok(Some(val.clone()));
            }
        }
        Ok(None)
    }

    fn set_variable(&mut self, scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
        let book = self.eval_stack.last().map(|(b, _)| *b);
        match self.namespace_access_for(scope, book) {
            None => {
                return Err(EvalError::new(
                    EvalErrorKind::HostError,
                    format!("namespace '{scope}' is not declared (cannot set {scope}:{name})"),
                ));
            }
            Some(NamespaceAccess::ReadOnly) => {
                return Err(EvalError::new(
                    EvalErrorKind::HostError,
                    format!("namespace '{scope}' is read-only (cannot set {scope}:{name})"),
                ));
            }
            Some(NamespaceAccess::ReadWrite) => {}
        }

        // Persist state namespace across sessions
        if scope == "state" {
            self.persistent_state
                .insert(name.to_string(), value.clone());
        }

        self.variables
            .entry(scope.to_string())
            .or_default()
            .insert(name.to_string(), value);

        Ok(())
    }

    fn fire_trigger(&mut self, entry_id: &str, _registry: &Registry) -> Result<String, EvalError> {
        // Triggers resolve like document refs: the calling entry's book
        // first, then the others. If the id resolves nowhere, fall back to
        // the local book (or book 0) and let filter_triggered drop it.
        let origin = self.eval_stack.last().map(|(book, _)| *book);
        let book = self
            .resolver
            .resolve(entry_id, origin, &self.book_templates)
            .map(|r| r.book)
            .or(origin)
            .unwrap_or(BookId(0));

        let key = (book, entry_id.to_string());
        if !self.active_entries.contains(&key) && !self.triggered_entries.contains(&key) {
            self.triggered_entries.push(key);
        }

        Ok(String::new())
    }

    fn resolve_document(
        &mut self,
        document_id: &str,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        // The currently-evaluating entry's book is the "local" book:
        // references resolve there first, then fall back to other books.
        let origin = self.eval_stack.last().map(|(book, _)| *book);

        // ── Resolve id → (book, template) ─────────────────────────
        let resolved = self
            .resolver
            .resolve(document_id, origin, &self.book_templates)
            .ok_or_else(|| {
                EvalError::new(
                    EvalErrorKind::DocumentNotFound,
                    format!("unknown document: {document_id}"),
                )
            })?;
        let book = resolved.book;
        let template = resolved.template.clone(); // Arc clone — cheap

        // ── Cycle detection (book-qualified) ──────────────────────
        let frame = (book, document_id.to_string());
        if self.eval_stack.contains(&frame) {
            let trail = self
                .eval_stack
                .iter()
                .map(|(_, id)| id.as_str())
                .collect::<Vec<_>>()
                .join(" → ");
            return Err(EvalError::new(
                EvalErrorKind::HostError,
                format!("document cycle detected: {trail} → {document_id}"),
            ));
        }

        // ── Depth check ───────────────────────────────────────────
        if self.eval_stack.len() >= self.max_recursion_depth {
            return Err(EvalError::new(
                EvalErrorKind::RecursionLimit,
                format!(
                    "recursion limit ({}) reached resolving [[{document_id}]]",
                    self.max_recursion_depth
                ),
            ));
        }

        // ── Evaluate ──────────────────────────────────────────────
        self.eval_stack.push(frame);
        let result = weaver_lang::evaluate(template.ast(), self, registry);
        self.eval_stack.pop();

        result
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lorebook::LorebookConfig;

    fn make_host() -> WeaverHost {
        let config = LorebookConfig::default();
        let mut host = WeaverHost::from_lorebook_config(BookId(0), &config);
        host.reserve_namespace("char", NamespaceAccess::ReadOnly);
        host.reserve_namespace("user", NamespaceAccess::ReadOnly);
        host.reserve_namespace("state", NamespaceAccess::ReadWrite);
        host.reserve_namespace("temp", NamespaceAccess::ReadWrite);
        host.set_host_variable("char", "name", Value::String("Aria".into()));
        host.set_host_variable("char", "class", Value::String("Mage".into()));
        host.set_host_variable("user", "name", Value::String("Player".into()));
        host
    }

    // ── Variable access ─────────────────────────────────────────────

    #[test]
    fn test_read_host_variable() {
        let host = make_host();
        let val = host.resolve_variable("char", "name").unwrap();
        assert_eq!(val, Some(Value::String("Aria".into())));
    }

    #[test]
    fn test_readonly_namespace_blocks_writes() {
        let mut host = make_host();
        let result = host.set_variable("char", "name", Value::String("Hacked".into()));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("read-only"));
    }

    #[test]
    fn test_writable_namespace_allows_writes() {
        let mut host = make_host();
        let result = host.set_variable("state", "visited", Value::Bool(true));
        assert!(result.is_ok());

        let val = host.resolve_variable("state", "visited").unwrap();
        assert_eq!(val, Some(Value::Bool(true)));
    }

    #[test]
    fn test_state_persists() {
        let mut host = make_host();
        host.set_variable("state", "counter", Value::Number(42.0))
            .unwrap();

        let persistent = host.export_persistent();
        let state = persistent.get("state").unwrap();

        assert_eq!(state.get("counter"), Some(&Value::Number(42.0)));
    }

    #[test]
    fn test_set_host_variable_state_persists() {
        let mut host = make_host();
        host.set_host_variable("state", "weapon", Value::String("longbow".into()));

        let val = host.resolve_variable("state", "weapon").unwrap();
        assert_eq!(val, Some(Value::String("longbow".into())));

        let persistent = host.export_persistent();
        let state = persistent.get("state").unwrap();

        assert_eq!(state.get("weapon"), Some(&Value::String("longbow".into())));
    }

    #[test]
    fn test_clear_transient_preserves_state() {
        let mut host = make_host();
        host.set_variable("temp", "test", Value::String("gone".into()))
            .unwrap();
        host.set_variable("state", "kept", Value::Bool(true))
            .unwrap();

        host.clear_transient();

        assert_eq!(host.resolve_variable("temp", "test").unwrap(), None);
        assert_eq!(
            host.resolve_variable("state", "kept").unwrap(),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn test_undeclared_namespace_is_invalid() {
        let mut host = make_host();
        // No frame, not reserved, not declared → writes error, reads yield nothing.
        let write = host.set_variable("custom", "foo", Value::String("bar".into()));
        assert!(write.is_err());
        assert_eq!(host.resolve_variable("custom", "foo").unwrap(), None);
    }

    // ── Active entry tracking ───────────────────────────────────────

    #[test]
    fn test_is_entry_active_local() {
        let mut host = make_host();
        host.set_active_entries(HashSet::from([
            (BookId(0), "entry_a".to_string()),
            (BookId(0), "entry_b".to_string()),
        ]));

        // is_entry_active_local is book-scoped: it needs a calling entry
        // to anchor "local" to.
        host.begin_entry(BookId(0), "caller");

        // Active in the calling book → true.
        assert!(host.is_entry_active_local("entry_a"));
        // Not active anywhere → false.
        assert!(!host.is_entry_active_local("entry_c"));

        host.end_entry();
    }

    // ── Trigger tests ───────────────────────────────────────────────

    #[test]
    fn test_trigger_produces_no_output() {
        let mut host = make_host();
        let registry = Registry::new();

        let result = host.fire_trigger("some_entry", &registry).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_trigger_records_entry_id() {
        let mut host = make_host();
        let registry = Registry::new();

        host.fire_trigger("entry_a", &registry).unwrap();
        host.fire_trigger("entry_b", &registry).unwrap();

        let triggered = host.drain_triggered_entries();
        assert_eq!(
            triggered,
            vec![
                (BookId(0), "entry_a".to_string()),
                (BookId(0), "entry_b".to_string())
            ]
        );
    }

    #[test]
    fn test_trigger_deduplicates() {
        let mut host = make_host();
        let registry = Registry::new();

        host.fire_trigger("entry_a", &registry).unwrap();
        host.fire_trigger("entry_a", &registry).unwrap();

        let triggered = host.drain_triggered_entries();
        assert_eq!(triggered, vec![(BookId(0), "entry_a".to_string())]);
    }

    #[test]
    fn test_trigger_skips_already_active() {
        let mut host = make_host();
        host.set_active_entries(HashSet::from([(BookId(0), "entry_a".to_string())]));
        let registry = Registry::new();

        host.fire_trigger("entry_a", &registry).unwrap();

        let triggered = host.drain_triggered_entries();
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_drain_clears_triggered() {
        let mut host = make_host();
        let registry = Registry::new();

        host.fire_trigger("entry_a", &registry).unwrap();
        let first = host.drain_triggered_entries();
        assert_eq!(first.len(), 1);

        let second = host.drain_triggered_entries();
        assert!(second.is_empty());
    }

    // ── Document resolution tests ───────────────────────────────────

    #[test]
    fn test_document_resolves_template() {
        let mut host = make_host();
        let registry = Registry::new();

        let template = Arc::new(CompiledTemplate::compile("Hello from document!").unwrap());
        host.set_entry_templates(HashMap::from([("my_doc".to_string(), template)]));

        let result = host.resolve_document("my_doc", &registry).unwrap();
        assert_eq!(result, "Hello from document!");
    }

    #[test]
    fn test_document_resolves_variables() {
        let mut host = make_host();
        let registry = Registry::new();

        let template = Arc::new(CompiledTemplate::compile("Name: {{char:name}}").unwrap());
        host.set_entry_templates(HashMap::from([("char_doc".to_string(), template)]));

        let result = host.resolve_document("char_doc", &registry).unwrap();
        assert_eq!(result, "Name: Aria");
    }

    #[test]
    fn test_document_not_found() {
        let mut host = make_host();
        let registry = Registry::new();

        let result = host.resolve_document("nonexistent", &registry);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, EvalErrorKind::DocumentNotFound);
    }

    #[test]
    fn test_document_cycle_detection() {
        let mut host = make_host();
        let registry = Registry::new();

        let template = Arc::new(CompiledTemplate::compile("self-reference").unwrap());
        host.set_entry_templates(HashMap::from([("entry_a".to_string(), template)]));

        host.eval_stack.push((BookId(0), "entry_a".to_string()));
        let result = host.resolve_document("entry_a", &registry);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("cycle"));
    }

    #[test]
    fn test_document_depth_limit() {
        let mut host = make_host();
        host.set_max_recursion_depth(2);
        let registry = Registry::new();

        host.eval_stack.push((BookId(0), "a".to_string()));
        host.eval_stack.push((BookId(0), "b".to_string()));

        let template = Arc::new(CompiledTemplate::compile("deep").unwrap());
        host.set_entry_templates(HashMap::from([("c".to_string(), template)]));

        let result = host.resolve_document("c", &registry);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, EvalErrorKind::RecursionLimit);
    }

    #[test]
    fn test_document_chains() {
        let mut host = make_host();
        let registry = Registry::new();

        let template_b = Arc::new(CompiledTemplate::compile("world").unwrap());
        let template_a = Arc::new(CompiledTemplate::compile("Hello, [[doc_b]]!").unwrap());
        host.set_entry_templates(HashMap::from([
            ("doc_a".to_string(), template_a),
            ("doc_b".to_string(), template_b),
        ]));

        let result = host.resolve_document("doc_a", &registry).unwrap();
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn test_custom_id_resolver_overrides_lookup() {
        struct AliasResolver;
        impl crate::resolver::IdResolver for AliasResolver {
            fn resolve<'a>(
                &self,
                _id: &str,
                _origin: Option<BookId>,
                books: &'a BookTemplates,
            ) -> Option<crate::resolver::ResolvedRef<'a>> {
                books
                    .get(BookId(0), "canonical")
                    .map(|template| crate::resolver::ResolvedRef {
                        book: BookId(0),
                        template,
                    })
            }
        }

        let mut host = make_host();
        host.set_id_resolver(Box::new(AliasResolver));

        let template = Arc::new(CompiledTemplate::compile("canonical content").unwrap());
        host.set_entry_templates(HashMap::from([("canonical".to_string(), template)]));

        let result = host.resolve_document("anything", &Registry::new()).unwrap();
        assert_eq!(result, "canonical content");
    }

    #[test]
    fn test_resolve_prefers_local_book() {
        let mut host = make_host();
        let mut books = BookTemplates::new();
        books.push(HashMap::from([(
            "shared".to_string(),
            Arc::new(CompiledTemplate::compile("from book 0").unwrap()),
        )]));
        books.push(HashMap::from([(
            "shared".to_string(),
            Arc::new(CompiledTemplate::compile("from book 1").unwrap()),
        )]));
        host.set_book_templates(books);

        // A reference fired from book 1 picks book 1's "shared".
        host.begin_entry(BookId(1), "caller");
        let result = host.resolve_document("shared", &Registry::new()).unwrap();
        host.end_entry();
        assert_eq!(result, "from book 1");
    }

    #[test]
    fn test_resolve_falls_back_to_other_books() {
        let mut host = make_host();
        let mut books = BookTemplates::new();
        books.push(HashMap::from([(
            "only_here".to_string(),
            Arc::new(CompiledTemplate::compile("found in book 0").unwrap()),
        )]));
        books.push(HashMap::new()); // book 1 has nothing
        host.set_book_templates(books);

        // Book 1 lacks "only_here" → falls back to book 0.
        host.begin_entry(BookId(1), "caller");
        let result = host
            .resolve_document("only_here", &Registry::new())
            .unwrap();
        host.end_entry();
        assert_eq!(result, "found in book 0");
    }

    #[test]
    fn test_fire_trigger_resolves_to_other_book() {
        let mut host = make_host();
        let mut books = BookTemplates::new();
        books.push(HashMap::new()); // book 0: empty
        books.push(HashMap::from([(
            "ambush".to_string(),
            Arc::new(CompiledTemplate::compile("x").unwrap()),
        )])); // book 1: has "ambush"
        host.set_book_templates(books);

        // Firing from book 0 with no local "ambush" → resolves into book 1.
        host.begin_entry(BookId(0), "starter");
        host.fire_trigger("ambush", &Registry::new()).unwrap();
        host.end_entry();

        let triggered = host.drain_triggered_entries();
        assert_eq!(triggered, vec![(BookId(1), "ambush".to_string())]);
    }

    #[test]
    fn test_is_active_local_vs_global() {
        let mut host = make_host();
        // "goblin" active in book 1 only; "sys" active in book 0.
        host.set_active_entries(HashSet::from([
            (BookId(0), "sys".to_string()),
            (BookId(1), "goblin".to_string()),
        ]));

        // Evaluating inside book 0:
        host.begin_entry(BookId(0), "caller");

        // Strict-local: book 0 has no "goblin" → false.
        assert!(!host.is_entry_active_local("goblin"));
        // Global: "goblin" is active in book 1 → true.
        assert!(host.is_entry_active("goblin"));
        // Local hit still works for book 0's own entry.
        assert!(host.is_entry_active_local("sys"));

        host.end_entry();
    }

    #[test]
    fn test_is_active_local_outside_any_entry() {
        let mut host = make_host();
        host.set_active_entries(HashSet::from([(BookId(0), "sys".to_string())]));
        // No eval-stack frame → no local book → strict-local is false,
        // global still sees it.
        assert!(!host.is_entry_active_local("sys"));
        assert!(host.is_entry_active("sys"));
    }

    #[test]
    fn test_reserved_overrides_book_declaration() {
        let mut host = make_host(); // char reserved ReadOnly
        // A book tries to declare char ReadWrite — recorded but never wins.
        let mut ns = HashMap::new();
        ns.insert(
            "char".to_string(),
            NamespaceConfig {
                access: NamespaceAccess::ReadWrite,
                description: String::new(),
            },
        );
        host.add_book_namespaces(BookId(1), &ns);

        host.begin_entry(BookId(1), "escalator");
        let result = host.set_variable("char", "name", Value::String("Hax".into()));
        host.end_entry();
        assert!(result.is_err());
    }

    #[test]
    fn test_book_scoped_access_is_per_book() {
        let mut host = make_host();
        // Same non-reserved namespace, opposite access per book.
        let mut a = HashMap::new();
        a.insert(
            "lore".to_string(),
            NamespaceConfig {
                access: NamespaceAccess::ReadWrite,
                description: String::new(),
            },
        );
        host.add_book_namespaces(BookId(0), &a);
        let mut b = HashMap::new();
        b.insert(
            "lore".to_string(),
            NamespaceConfig {
                access: NamespaceAccess::ReadOnly,
                description: String::new(),
            },
        );
        host.add_book_namespaces(BookId(1), &b);

        // Book 0 may write; book 1 may not — same shared storage.
        host.begin_entry(BookId(0), "writer");
        let w0 = host.set_variable("lore", "x", Value::Number(1.0));
        host.end_entry();
        assert!(w0.is_ok());

        host.begin_entry(BookId(1), "reader");
        let r = host.resolve_variable("lore", "x").unwrap();
        let w1 = host.set_variable("lore", "x", Value::Number(2.0));
        host.end_entry();
        assert_eq!(r, Some(Value::Number(1.0))); // shared storage: sees book 0's write
        assert!(w1.is_err()); // but read-only for book 1
    }

    #[test]
    fn test_undeclared_scope_unreadable_per_book() {
        let mut host = make_host();
        let mut a = HashMap::new();
        a.insert(
            "lore".to_string(),
            NamespaceConfig {
                access: NamespaceAccess::ReadWrite,
                description: String::new(),
            },
        );
        host.add_book_namespaces(BookId(0), &a);

        // Book 0 declared "lore"; book 1 did not → invalid for book 1.
        host.begin_entry(BookId(0), "writer");
        host.set_variable("lore", "x", Value::Number(1.0)).unwrap();
        host.end_entry();

        host.begin_entry(BookId(1), "outsider");
        assert_eq!(host.resolve_variable("lore", "x").unwrap(), None);
        assert!(host.set_variable("lore", "x", Value::Number(2.0)).is_err());
        host.end_entry();
    }

    // ── Persistence ─────────────────────────────────────────────────

    fn writable_lore_ns() -> HashMap<String, NamespaceConfig> {
        HashMap::from([(
            "lore".to_string(),
            NamespaceConfig {
                access: NamespaceAccess::ReadWrite,
                description: String::new(),
            },
        )])
    }

    #[test]
    fn test_export_covers_all_writable_namespaces() {
        let mut host = make_host();
        host.add_book_namespaces(BookId(0), &writable_lore_ns());

        host.begin_entry(BookId(0), "writer");
        host.set_variable("state", "counter", Value::Number(1.0))
            .unwrap();
        host.set_variable("lore", "faction", Value::String("Rebels".into()))
            .unwrap();
        host.set_variable("temp", "scratch", Value::Bool(true))
            .unwrap();
        host.end_entry();

        let snap = host.export_persistent();

        // The previously-lost book namespace is now captured...
        assert_eq!(
            snap.get("lore").unwrap().get("faction"),
            Some(&Value::String("Rebels".into()))
        );
        // ...alongside state.
        assert_eq!(
            snap.get("state").unwrap().get("counter"),
            Some(&Value::Number(1.0))
        );
        // Transient and host-provided read-only scopes are excluded.
        assert!(!snap.contains_key("temp"));
        assert!(!snap.contains_key("char"));
        assert!(!snap.contains_key("user"));
    }

    #[test]
    fn test_persistent_round_trip_multi_namespace() {
        let mut host = make_host();
        host.add_book_namespaces(BookId(0), &writable_lore_ns());

        host.begin_entry(BookId(0), "writer");
        host.set_variable("state", "hp", Value::Number(7.0))
            .unwrap();
        host.set_variable("lore", "faction", Value::String("Rebels".into()))
            .unwrap();
        host.end_entry();

        let snap = host.export_persistent();

        // Reload into a fresh host and confirm both namespaces survive.
        let mut fresh = make_host();
        fresh.add_book_namespaces(BookId(0), &writable_lore_ns());
        fresh.restore_persistent(snap);

        fresh.begin_entry(BookId(0), "reader");
        assert_eq!(
            fresh.resolve_variable("state", "hp").unwrap(),
            Some(Value::Number(7.0))
        );
        assert_eq!(
            fresh.resolve_variable("lore", "faction").unwrap(),
            Some(Value::String("Rebels".into()))
        );
        fresh.end_entry();

        let persistent = fresh.export_persistent();
        let state = persistent.get("state").unwrap();
        // The state mirror is kept in sync on restore.
        assert_eq!(state.get("hp"), Some(&Value::Number(7.0)));
    }
}
