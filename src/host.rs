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

use crate::lorebook::LorebookConfig;

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
    /// Access control rules per namespace.
    namespace_access: HashMap<String, NamespaceAccess>,

    /// Variable storage: namespace → (name → value).
    variables: HashMap<String, HashMap<String, Value>>,

    /// Persistent state that survives across evaluation passes.
    /// This is the `state:` namespace — it gets serialized between sessions.
    persistent_state: HashMap<String, Value>,

    /// Set of entry IDs currently being evaluated — used for cycle detection
    /// in recursive document resolution.
    eval_stack: Vec<String>,

    /// Maximum recursion depth for document chains.
    max_recursion_depth: usize,

    /// Set of entry IDs that are currently active (populated before evaluation).
    active_entries: HashSet<String>,

    /// The entry currently being evaluated (for diagnostics).
    current_entry: Option<String>,

    /// Compiled entry templates, keyed by entry ID. Set before each
    /// evaluation pass so that `resolve_document` can evaluate entries
    /// inline.
    entry_templates: HashMap<String, Arc<CompiledTemplate>>,

    /// Entry IDs activated via `<trigger>` during the current evaluation
    /// pass. The engine drains this after evaluation to feed the next
    /// activation pass.
    triggered_entries: Vec<String>,
}

impl WeaverHost {
    /// Create a WeaverHost from lorebook configuration.
    pub fn from_lorebook_config(config: &LorebookConfig) -> Self {
        let namespace_access: HashMap<String, NamespaceAccess> = config
            .namespaces
            .iter()
            .map(|(k, v)| (k.clone(), v.access))
            .collect();

        Self {
            namespace_access,
            variables: HashMap::new(),
            persistent_state: HashMap::new(),
            eval_stack: Vec::new(),
            max_recursion_depth: 10,
            active_entries: HashSet::new(),
            current_entry: None,
            entry_templates: HashMap::new(),
            triggered_entries: Vec::new(),
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
    /// Also populates the `_active` namespace so that the `is_active`
    /// command can check entry status from within templates.
    pub fn set_active_entries(&mut self, ids: HashSet<String>) {
        // Clear previous _active namespace
        self.variables.remove("_active");
        let mut active_ns = HashMap::new();
        for id in &ids {
            active_ns.insert(id.clone(), Value::Bool(true));
        }
        self.variables.insert("_active".to_string(), active_ns);
        self.active_entries = ids;
    }

    /// Check if an entry is currently active.
    pub fn is_entry_active(&self, id: &str) -> bool {
        self.active_entries.contains(id)
    }

    // ── Entry template management ───────────────────────────────────

    /// Provide compiled templates for document resolution.
    ///
    /// Called by the engine before each evaluation pass. Templates are
    /// shared via `Arc` to avoid cloning ASTs.
    pub fn set_entry_templates(&mut self, templates: HashMap<String, Arc<CompiledTemplate>>) {
        self.entry_templates = templates;
    }

    // ── Trigger collection ──────────────────────────────────────────

    /// Drain the list of entry IDs that were triggered during evaluation.
    ///
    /// Called by the engine after an evaluation pass. The returned IDs
    /// become candidates for the next activation pass, subject to
    /// cooldown, budget, and other constraints.
    pub fn drain_triggered_entries(&mut self) -> Vec<String> {
        std::mem::take(&mut self.triggered_entries)
    }

    // ── Eval stack management ───────────────────────────────────────

    /// Called before evaluating an entry — pushes onto the eval stack
    /// for cycle detection.
    pub fn begin_entry(&mut self, entry_id: &str) {
        self.current_entry = Some(entry_id.to_string());
        self.eval_stack.push(entry_id.to_string());
    }

    /// Called after evaluating an entry — pops from the eval stack.
    pub fn end_entry(&mut self) {
        self.eval_stack.pop();
        self.current_entry = self.eval_stack.last().cloned();
    }

    // ── Persistent state ────────────────────────────────────────────

    /// Get the persistent state map (for serialization between sessions).
    pub fn persistent_state(&self) -> &HashMap<String, Value> {
        &self.persistent_state
    }

    /// Restore persistent state (e.g. loaded from a save file).
    pub fn restore_persistent_state(&mut self, state: HashMap<String, Value>) {
        self.persistent_state = state;
        // Mirror into the variables map so templates can access it
        self.variables
            .insert("state".to_string(), self.persistent_state.clone());
    }

    /// Clear temporary (non-persistent) variables. Called between turns.
    pub fn clear_transient(&mut self) {
        // Keep: state, host-provided namespaces
        // Clear: local, triggered entries, _active
        self.variables.remove("local");
        self.variables.remove("_active");
        self.triggered_entries.clear();
    }

    /// Set the maximum recursion depth for document resolution chains.
    pub fn set_max_recursion_depth(&mut self, depth: usize) {
        self.max_recursion_depth = depth;
    }
}

// ── EvalContext implementation ───────────────────────────────────────────

impl EvalContext for WeaverHost {
    fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
        // Check variables map
        if let Some(ns) = self.variables.get(scope) {
            if let Some(val) = ns.get(name) {
                return Ok(Some(val.clone()));
            }
        }

        // For the state namespace, also check persistent state
        if scope == "state" {
            if let Some(val) = self.persistent_state.get(name) {
                return Ok(Some(val.clone()));
            }
        }

        Ok(None)
    }

    fn set_variable(&mut self, scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
        // Check access control
        if let Some(access) = self.namespace_access.get(scope) {
            if *access == NamespaceAccess::ReadOnly {
                return Err(EvalError::new(
                    EvalErrorKind::HostError,
                    format!("namespace '{scope}' is read-only (cannot set {scope}:{name})"),
                ));
            }
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
        // In ContextWeaver, triggers don't produce output. They record
        // the entry ID for the engine to pick up after evaluation and
        // feed into the next activation pass.
        if !self.active_entries.contains(entry_id)
            && !self.triggered_entries.contains(&entry_id.to_string())
        {
            self.triggered_entries.push(entry_id.to_string());
        }

        // Triggers produce no output
        Ok(String::new())
    }

    fn resolve_document(
        &mut self,
        document_id: &str,
        registry: &Registry,
    ) -> Result<String, EvalError> {
        // ── Cycle detection ────────────────────────────────────────
        if self.eval_stack.contains(&document_id.to_string()) {
            return Err(EvalError::new(
                EvalErrorKind::HostError,
                format!(
                    "document cycle detected: {} → {document_id}",
                    self.eval_stack.join(" → ")
                ),
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

        // ── Look up and evaluate ──────────────────────────────────
        let template = self
            .entry_templates
            .get(document_id)
            .ok_or_else(|| {
                EvalError::new(
                    EvalErrorKind::DocumentNotFound,
                    format!("unknown document: {document_id}"),
                )
            })?
            .clone(); // Arc clone — cheap

        self.eval_stack.push(document_id.to_string());
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
        let mut host = WeaverHost::from_lorebook_config(&config);
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

        assert_eq!(
            host.persistent_state().get("counter"),
            Some(&Value::Number(42.0))
        );
    }

    #[test]
    fn test_set_host_variable_state_persists() {
        let mut host = make_host();
        host.set_host_variable("state", "weapon", Value::String("longbow".into()));

        let val = host.resolve_variable("state", "weapon").unwrap();
        assert_eq!(val, Some(Value::String("longbow".into())));

        assert_eq!(
            host.persistent_state().get("weapon"),
            Some(&Value::String("longbow".into()))
        );
    }

    #[test]
    fn test_clear_transient_preserves_state() {
        let mut host = make_host();
        host.set_variable("local", "temp", Value::String("gone".into()))
            .unwrap();
        host.set_variable("state", "kept", Value::Bool(true))
            .unwrap();

        host.clear_transient();

        assert_eq!(host.resolve_variable("local", "temp").unwrap(), None);
        assert_eq!(
            host.resolve_variable("state", "kept").unwrap(),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn test_unknown_namespace_is_writable() {
        let mut host = make_host();
        let result = host.set_variable("custom", "foo", Value::String("bar".into()));
        assert!(result.is_ok());
    }

    // ── Active entry tracking ───────────────────────────────────────

    #[test]
    fn test_active_entries_populate_namespace() {
        let mut host = make_host();
        host.set_active_entries(HashSet::from([
            "entry_a".to_string(),
            "entry_b".to_string(),
        ]));

        // Should be readable via _active namespace
        let val = host.resolve_variable("_active", "entry_a").unwrap();
        assert_eq!(val, Some(Value::Bool(true)));

        let val = host.resolve_variable("_active", "entry_c").unwrap();
        assert_eq!(val, None);
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
        assert_eq!(triggered, vec!["entry_a", "entry_b"]);
    }

    #[test]
    fn test_trigger_deduplicates() {
        let mut host = make_host();
        let registry = Registry::new();

        host.fire_trigger("entry_a", &registry).unwrap();
        host.fire_trigger("entry_a", &registry).unwrap();

        let triggered = host.drain_triggered_entries();
        assert_eq!(triggered, vec!["entry_a"]);
    }

    #[test]
    fn test_trigger_skips_already_active() {
        let mut host = make_host();
        host.set_active_entries(HashSet::from(["entry_a".to_string()]));
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

        host.eval_stack.push("entry_a".to_string());
        let result = host.resolve_document("entry_a", &registry);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("cycle"));
    }

    #[test]
    fn test_document_depth_limit() {
        let mut host = make_host();
        host.set_max_recursion_depth(2);
        let registry = Registry::new();

        host.eval_stack.push("a".to_string());
        host.eval_stack.push("b".to_string());

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
}
