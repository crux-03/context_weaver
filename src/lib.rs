//! # ContextWeaver
//!
//! A lorebook engine for LLM role-playing applications, built on
//! [weaver-lang](../weaver_lang). ContextWeaver manages a collection of
//! entries that are selectively activated based on conversation context
//! and assembled into the final prompt sent to the model.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │  Host Application (LLM frontend)                    │
//! │                                                     │
//! │  Provides: chat history, character data, user prefs │
//! │  Receives: assembled context blocks for the prompt  │
//! └────────────────────────┬────────────────────────────┘
//!                          │
//! ┌────────────────────────▼────────────────────────────┐
//! │  ContextWeaver                                      │
//! │                                                     │
//! │  ┌─────────────┐  ┌────────────┐  ┌─────────────┐   │
//! │  │  Lorebook   │  │ Activation │  │  Assembler  │   │
//! │  │  (entries)  │──│  Engine    │──│  (ordering, │   │
//! │  │             │  │            │  │   budgeting)│   │
//! │  └─────────────┘  └────────────┘  └──────┬──────┘   │
//! │                                          │          │
//! │  ┌───────────────────────────────────────▼──────┐   │
//! │  │  WeaverHost (EvalContext impl)               │   │
//! │  │  - namespace management                      │   │
//! │  │  - read-only enforcement                     │   │
//! │  │  - trigger collection (no output)            │   │
//! │  │  - document → recursive entry evaluation     │   │
//! │  └──────────────────────────────────────────────┘   │
//! │                                                     │
//! │  ┌──────────────────────────────────────────────┐   │
//! │  │  Plugin Interface                            │   │
//! │  │  - custom processors & commands              │   │
//! │  │  - activation hooks                          │   │
//! │  └──────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────┘
//!                          │
//! ┌────────────────────────▼────────────────────────────┐
//! │  weaver-lang (template evaluation)                  │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use context_weaver::{ContextWeaver, Lorebook, ChatMessage, Slot};
//!
//! // Load a lorebook from disk
//! let book = Lorebook::load_from_directory("./my_character/lorebook")?;
//!
//! // Configure the engine
//! let mut weaver = ContextWeaver::new(book);
//! weaver.set_variable("char", "name", "Aria");
//! weaver.set_variable("char", "class", "Mage");
//! weaver.set_variable("user", "name", "Player");
//!
//! // Provide conversation context
//! let messages = vec![
//!     ChatMessage::user("I walk into the dark forest"),
//!     ChatMessage::assistant("The trees close in around you..."),
//! ];
//!
//! // Assemble activated entries into context blocks
//! let blocks = weaver.assemble(&messages)?;
//! for block in &blocks {
//!     println!("[{}] {}", block.slot, block.content);
//! }
//! ```

pub mod activation;
pub mod assembler;
pub mod entry;
pub mod host;
pub mod lifecycle;
pub mod lorebook;
pub mod plugin;
pub mod resolver;
#[cfg(feature = "stdlib")]
pub mod stdlib;

pub use activation::{ActivationEngine, ActivationReason, ActivationResult, ActivationState};
pub use assembler::{
    AssembledBlock, ContextAssembler, GuesstimationTokenizer, Slot, TokenBudget, Tokenizer,
};
pub use entry::{Entry, EntryMeta};
pub use host::{NamespaceAccess, NamespaceConfig, WeaverHost};
pub use lifecycle::{
    FnLifecycle, HookError, LifecyclePlugin, PostActivationCtx, PostAssembleCtx, PostEvaluateCtx,
    PreActivationCtx, PreEvaluateCtx, TriggerCtx, TurnAdvanceCtx,
};
pub use lorebook::{BookId, Lorebook, LorebookConfig, LorebookSet};
pub use plugin::Plugin;
pub use resolver::{BookTemplates, DefaultIdResolver, IdResolver, ResolvedRef};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use weaver_lang::registry::{CommandSignature, ParamDef, WeaverCommand};
use weaver_lang::{CompiledTemplate, EvalContext, EvalError, Registry, Value};

// ── Top-level engine ────────────────────────────────────────────────────

/// The main entry point for ContextWeaver.
///
/// Owns a [`Lorebook`], manages the [`Registry`] and [`WeaverHost`],
/// and orchestrates the activation → evaluation → assembly pipeline.
pub struct ContextWeaver {
    books: LorebookSet,
    registry: Registry,
    host: WeaverHost,
    /// Per-book activation state, indexed by `BookId.0`, parallel to `books`.
    activation_states: Vec<ActivationState>,
    config: EngineConfig,
    /// The tokenizer used for budget estimation.
    tokenizer: Box<dyn Tokenizer>,
    /// Which slots are available in the host's ContextDefinition template.
    /// Entries targeting unavailable slots (with no matching fallback) are dropped.
    available_slots: HashSet<Slot>,
    /// Lifecycle plugins registered to hook into the assembly pipeline.
    /// Fired in registration order; see the [`lifecycle`] module for details.
    lifecycle_plugins: Vec<Box<dyn LifecyclePlugin>>,
}

/// A chat message provided by the host application for activation scanning.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }
}

/// Top-level engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum recursion depth for document chains.
    pub max_recursion_depth: usize,
    /// Maximum number of entries that can activate per assembly pass.
    pub max_active_entries: usize,
    /// Number of trigger-resolution passes (trigger output activating
    /// further entries).
    pub max_trigger_passes: usize,
    /// Whether to use lenient evaluation (pass through errors as raw syntax).
    pub lenient: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_recursion_depth: 10,
            max_active_entries: 100,
            max_trigger_passes: 3,
            lenient: false,
        }
    }
}

impl ContextWeaver {
    pub fn new(lorebook: Lorebook) -> Self {
        let mut host = WeaverHost::from_lorebook_config(BookId(0), &lorebook.config);
        let mut registry = Registry::new();

        host.reserve_namespace("state", NamespaceAccess::ReadWrite);
        host.reserve_namespace("local", NamespaceAccess::ReadWrite);

        // Register built-in commands and processors
        register_builtins(&mut registry);

        // Set max recursion depth from default config
        host.set_max_recursion_depth(EngineConfig::default().max_recursion_depth);

        Self {
            books: LorebookSet::single(lorebook),
            registry,
            host,
            activation_states: vec![ActivationState::new()],
            config: EngineConfig::default(),
            tokenizer: Box::new(GuesstimationTokenizer),
            available_slots: Slot::standard_slots().into_iter().collect(),
            lifecycle_plugins: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: EngineConfig) -> Self {
        self.host
            .set_max_recursion_depth(config.max_recursion_depth);
        self.config = config;
        self
    }

    /// Set a custom tokenizer for accurate token budget enforcement.
    ///
    /// The default is [`GuesstimationTokenizer`] which estimates ~4 chars
    /// per token. For production use, provide a tokenizer that matches
    /// your target model (tiktoken, sentencepiece, etc.).
    pub fn set_tokenizer(&mut self, tokenizer: Box<dyn Tokenizer>) {
        self.tokenizer = tokenizer;
    }

    /// Set which slots are available in the host's ContextDefinition.
    ///
    /// Entries targeting slots not in this set (and with no matching
    /// fallback) will be silently dropped during assembly. By default,
    /// all standard slots are available.
    pub fn set_available_slots(&mut self, slots: impl IntoIterator<Item = Slot>) {
        self.available_slots = slots.into_iter().collect();
    }

    /// Set a variable in a host-provided namespace.
    ///
    /// This is how the host application feeds data into the lorebook:
    /// character attributes, user preferences, world state, etc.
    pub fn set_variable(&mut self, scope: &str, name: &str, value: impl Into<Value>) {
        self.host.set_host_variable(scope, name, value.into());
    }

    /// Add a lorebook to the set, returning its [`BookId`]. Books are
    /// evaluated together; ids reference across books local-first, then fall
    /// back to the other books in registration order.
    pub fn add_lorebook(&mut self, book: Lorebook) -> BookId {
        let id = self.books.add(book);
        self.activation_states.push(ActivationState::new());
        if let Some(added) = self.books.get(id) {
            self.host.add_book_namespaces(id, &added.config.namespaces);
        }
        id
    }

    /// Reserve a namespace with fixed access, uniform across all books and
    /// not overridable by any book. Host applications use this to expose
    /// their own scopes — character data, user info, chat metadata, etc.
    pub fn reserve_namespace(&mut self, name: impl Into<String>, access: NamespaceAccess) {
        self.host.reserve_namespace(name, access);
    }

    /// Register a plugin, adding its processors and commands to the registry.
    pub fn register_plugin(&mut self, plugin: impl Plugin) -> Result<(), plugin::PluginError> {
        plugin.register(&mut self.registry);
        plugin.init()
    }

    pub fn register_lifecycle<P: LifecyclePlugin + 'static>(&mut self, plugin: P) {
        self.lifecycle_plugins.push(Box::new(plugin));
    }

    /// Access a book's activation state (for serialization / inspection).
    pub fn activation_state(&self, book: BookId) -> Option<&ActivationState> {
        self.activation_states.get(book.0)
    }

    /// Access all books' activation states, in id order.
    pub fn activation_states(&self) -> &[ActivationState] {
        &self.activation_states
    }

    /// Restore a book's activation state (e.g. from a save file).
    pub fn restore_activation_state(&mut self, book: BookId, state: ActivationState) {
        if let Some(slot) = self.activation_states.get_mut(book.0) {
            *slot = state;
        }
    }

    /// Access the host's persistent state (for serialization).
    pub fn persistent_state(&self) -> &HashMap<String, Value> {
        self.host.persistent_state()
    }

    /// Restore persistent state (e.g. from a save file).
    pub fn restore_persistent_state(&mut self, state: HashMap<String, Value>) {
        self.host.restore_persistent_state(state);
    }

    /// Advance the turn counter. Call this once per conversation turn,
    /// before `assemble`. Decrements sticky counters and clears
    /// transient variables.
    pub fn advance_turn(&mut self) -> Result<(), ContextWeaverError> {
        for state in &mut self.activation_states {
            state.advance_turn();
        }
        self.host.clear_transient();

        // Fire on_turn_advance once per book, with that book's state.
        for state in &mut self.activation_states {
            for plugin in &mut self.lifecycle_plugins {
                let plugin_name = plugin.name().to_string();
                let mut ctx = TurnAdvanceCtx { state };
                plugin
                    .on_turn_advance(&mut ctx)
                    .map_err(|e| ContextWeaverError::PluginHook {
                        plugin: plugin_name,
                        hook: "on_turn_advance",
                        source: e,
                    })?;
            }
        }

        Ok(())
    }

    /// Run the full pipeline: activate → evaluate → assemble.
    ///
    /// Returns ordered context blocks ready for prompt insertion.
    pub fn assemble(
        &mut self,
        messages: &[ChatMessage],
    ) -> Result<Vec<AssembledBlock>, ContextWeaverError> {
        // Clone messages so pre_activation hooks can mutate.
        let mut messages_owned: Vec<ChatMessage> = messages.to_vec();
        let turn = self
            .activation_states
            .first()
            .map(|s| s.current_turn())
            .unwrap_or(0);

        // ── Lifecycle: pre_activation ───────────────────────────────
        for plugin in &mut self.lifecycle_plugins {
            let plugin_name = plugin.name().to_string();
            let mut ctx = PreActivationCtx {
                messages: &mut messages_owned,
                turn,
            };
            plugin
                .pre_activation(&mut ctx)
                .map_err(|e| ContextWeaverError::PluginHook {
                    plugin: plugin_name,
                    hook: "pre_activation",
                    source: e,
                })?;
        }

        // ── Prepare host for evaluation ─────────────────────────────
        let book_templates = self.build_book_templates();
        self.host.set_book_templates(book_templates);

        // ── Phase 1: Activation scan ────────────────────────────────
        let mut results = ActivationEngine::scan(
            &self.books,
            messages,
            &mut self.host,
            &self.registry,
            &self.activation_states,
        );

        // ── Lifecycle: post_activation ──────────────────────────────
        {
            let lifecycle_plugins = &mut self.lifecycle_plugins;
            let lorebook = self.books.primary();
            for plugin in lifecycle_plugins {
                let plugin_name = plugin.name().to_string();
                let mut ctx = PostActivationCtx {
                    results: &mut results,
                    lorebook,
                    turn,
                };
                plugin
                    .post_activation(&mut ctx)
                    .map_err(|e| ContextWeaverError::PluginHook {
                        plugin: plugin_name,
                        hook: "post_activation",
                        source: e,
                    })?;
            }
        }

        // Enforce max active entries
        if results.len() > self.config.max_active_entries {
            results.truncate(self.config.max_active_entries);
        }

        let mut active_ids: Vec<(BookId, String)> = results
            .iter()
            .map(|r| (r.book, r.entry_id.clone()))
            .collect();

        // Tell the host which entries are active (for trigger dedup and
        // the is_active command via the _active namespace)
        self.host
            .set_active_entries(active_ids.iter().cloned().collect());

        // ── Phase 2: Evaluate + trigger resolution passes ───────────
        //
        // Each pass evaluates entries and captures both their output AND
        // any trigger side effects. Already-evaluated entries are NOT
        // re-evaluated — their cached output is reused. This prevents
        // side effects (inc_var, set_var, push_var) from running twice.
        //
        // Each pass:
        //   1. Evaluate un-evaluated entries, caching output
        //   2. Drain triggered IDs from host
        //   3. Filter through cooldown/conditions
        //   4. Add newly activated entries to the list, repeat
        let mut evaluated_cache: HashMap<(BookId, String), EvaluatedEntry> = HashMap::new();

        // Evaluate the initial batch and cache results
        for (key, entry) in self.evaluate_entries(&active_ids)? {
            evaluated_cache.insert(key, entry);
        }

        for pass_number in 0..self.config.max_trigger_passes {
            // Drain trigger activations collected during evaluation
            let mut triggered = self.host.drain_triggered_entries();
            if triggered.is_empty() {
                break;
            }

            // ── Lifecycle: on_trigger_fired ─────────────────────────
            for plugin in &mut self.lifecycle_plugins {
                let plugin_name = plugin.name().to_string();
                let mut ctx = TriggerCtx {
                    triggered_ids: &mut triggered,
                    pass_number,
                };
                plugin
                    .on_trigger_fired(&mut ctx)
                    .map_err(|e| ContextWeaverError::PluginHook {
                        plugin: plugin_name,
                        hook: "on_trigger_fired",
                        source: e,
                    })?;
            }

            // Filter through activation rules
            let new_results = ActivationEngine::filter_triggered(
                &self.books,
                &triggered,
                &active_ids,
                &mut self.host,
                &self.registry,
                &self.activation_states,
            );

            if new_results.is_empty() {
                break;
            }

            // Collect truly new entry IDs (skip entries already in
            // active_ids — they may be sticky refreshes that don't need
            // re-evaluation or duplicate list entries)
            let new_ids: Vec<(BookId, String)> = new_results
                .iter()
                .map(|r| (r.book, r.entry_id.clone()))
                .filter(|key| !active_ids.contains(key))
                .collect();

            for id in &new_ids {
                active_ids.push(id.clone());
            }
            results.extend(new_results);

            // Update host's active set
            self.host
                .set_active_entries(active_ids.iter().cloned().collect());

            // Evaluate ONLY the truly new entries (not sticky refreshes)
            if !new_ids.is_empty() {
                for (key, entry) in self.evaluate_entries(&new_ids)? {
                    evaluated_cache.insert(key, entry);
                }
            }

            if active_ids.len() > self.config.max_active_entries {
                active_ids.truncate(self.config.max_active_entries);
                break;
            }
        }

        // ── Phase 3: Collect evaluated entries in activation order ───
        let evaluated: Vec<EvaluatedEntry> = active_ids
            .iter()
            .filter_map(|key| evaluated_cache.remove(key))
            .collect();

        // ── Phase 4: Record activations ─────────────────────────────
        //
        // Only record FRESH activations (keyword, regex, constant,
        // triggered) — not sticky carry-forwards. This ensures that
        // carry-forwards don't reset the sticky countdown, while fresh
        // re-activations (keyword re-match, trigger refresh) DO reset it.
        //
        // When an entry appears multiple times in `results` (e.g. once
        // as Sticky carry-forward, once as Triggered refresh), the
        // non-Sticky entry takes precedence here because we iterate
        // all results and the last `record_activation` call wins.
        for result in &results {
            if matches!(result.reason, ActivationReason::Sticky { .. }) {
                continue;
            }
            let sticky_turns = self
                .books
                .get(result.book)
                .and_then(|b| b.get_entry(&result.entry_id))
                .map(|e| e.meta.sticky_turns);
            if let Some(sticky_turns) = sticky_turns {
                if let Some(state) = self.activation_states.get_mut(result.book.0) {
                    state.record_activation(&result.entry_id, sticky_turns);
                }
            }
        }

        // ── Phase 5: Assemble ───────────────────────────────────────
        let mut blocks = ContextAssembler::assemble(
            evaluated,
            &self.books.primary().config,
            &*self.tokenizer,
            &self.available_slots,
        );

        // ── Lifecycle: post_assemble ────────────────────────────────
        {
            let lifecycle_plugins = &mut self.lifecycle_plugins;
            let lorebook = self.books.primary();
            for plugin in lifecycle_plugins {
                let plugin_name = plugin.name().to_string();
                let mut ctx = PostAssembleCtx {
                    blocks: &mut blocks,
                    lorebook,
                };
                plugin
                    .post_assemble(&mut ctx)
                    .map_err(|e| ContextWeaverError::PluginHook {
                        plugin: plugin_name,
                        hook: "post_assemble",
                        source: e,
                    })?;
            }
        }

        Ok(blocks)
    }

    /// Build the per-book template store for the host, one partition per book.
    fn build_book_templates(&self) -> BookTemplates {
        let mut books = BookTemplates::new();
        for (_id, book) in self.books.iter() {
            let templates: HashMap<String, Arc<CompiledTemplate>> = book
                .entries_in_order()
                .map(|e| (e.meta.id.clone(), e.compiled.clone()))
                .collect();
            books.push(templates);
        }
        books
    }

    /// Evaluate all active entries and collect their output.
    fn evaluate_entries(
        &mut self,
        keys: &[(BookId, String)],
    ) -> Result<Vec<((BookId, String), EvaluatedEntry)>, ContextWeaverError> {
        let mut results = Vec::new();

        for (book, id) in keys {
            if let Some(entry) = self.books.get(*book).and_then(|b| b.get_entry(id)).cloned() {
                if let Some(content) = self.evaluate_single_entry(*book, &entry)? {
                    results.push((
                        (*book, id.clone()),
                        EvaluatedEntry {
                            id: id.clone(),
                            meta: entry.meta.clone(),
                            content,
                        },
                    ));
                }
            }
        }

        Ok(results)
    }

    /// Evaluate a single entry's template against the current host state.
    /// Returns `Ok(None)` if a `pre_evaluate` hook set `skip = true`.
    fn evaluate_single_entry(
        &mut self,
        book: BookId,
        entry: &Entry,
    ) -> Result<Option<String>, ContextWeaverError> {
        // ── Lifecycle: pre_evaluate ─────────────────────────────────
        let mut skip = false;
        for plugin in &mut self.lifecycle_plugins {
            let plugin_name = plugin.name().to_string();
            let mut ctx = PreEvaluateCtx {
                entry,
                skip: &mut skip,
            };
            plugin
                .pre_evaluate(&mut ctx)
                .map_err(|e| ContextWeaverError::PluginHook {
                    plugin: plugin_name,
                    hook: "pre_evaluate",
                    source: e,
                })?;
        }
        if skip {
            return Ok(None);
        }

        // ── Evaluate ────────────────────────────────────────────────
        self.host.begin_entry(book, &entry.meta.id);

        let opts = weaver_lang::EvalOptions::new()
            .max_node_evaluations(50_000)
            .max_iterations(10_000)
            .lenient(self.config.lenient);

        let result = weaver_lang::evaluate_with_options(
            entry.compiled.ast(),
            &mut self.host,
            &self.registry,
            opts,
        );

        self.host.end_entry();

        let mut content = result.map_err(|e| ContextWeaverError::Eval {
            entry_id: entry.meta.id.clone(),
            source: e,
        })?;

        // ── Lifecycle: post_evaluate ────────────────────────────────
        for plugin in &mut self.lifecycle_plugins {
            let plugin_name = plugin.name().to_string();
            let mut ctx = PostEvaluateCtx {
                entry,
                content: &mut content,
            };
            plugin
                .post_evaluate(&mut ctx)
                .map_err(|e| ContextWeaverError::PluginHook {
                    plugin: plugin_name,
                    hook: "post_evaluate",
                    source: e,
                })?;
        }

        Ok(Some(content))
    }

    /// Install a custom [`IdResolver`] for document id resolution.
    ///
    /// By default ids resolve by direct lookup. A custom resolver can
    /// override this — e.g. to resolve ids across multiple active lorebooks.
    pub fn set_id_resolver(&mut self, resolver: Box<dyn IdResolver>) {
        self.host.set_id_resolver(resolver);
    }
}

/// An entry that has been evaluated to its final string content.
pub struct EvaluatedEntry {
    pub id: String,
    pub meta: EntryMeta,
    pub content: String,
}

// ── Errors ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ContextWeaverError {
    /// Failed to parse an entry's frontmatter.
    MetaParse { entry_path: String, message: String },
    /// Failed to parse an entry's weaver-lang body.
    TemplateParse {
        entry_id: String,
        errors: Vec<weaver_lang::ParseError>,
    },
    /// Failed during template evaluation.
    Eval {
        entry_id: String,
        source: weaver_lang::EvalError,
    },
    /// A document reference hit the recursion limit.
    RecursionLimit { entry_id: String, depth: usize },
    /// I/O error loading lorebook files.
    Io(std::io::Error),
    PluginHook {
        plugin: String,
        hook: &'static str,
        source: HookError,
    },
}

impl std::fmt::Display for ContextWeaverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MetaParse {
                entry_path,
                message,
            } => {
                write!(f, "metadata parse error in {entry_path}: {message}")
            }
            Self::TemplateParse { entry_id, errors } => {
                write!(f, "template parse error in entry '{entry_id}':")?;
                for e in errors {
                    write!(f, "\n  {e}")?;
                }
                Ok(())
            }
            Self::Eval { entry_id, source } => {
                write!(f, "evaluation error in entry '{entry_id}': {source}")
            }
            Self::RecursionLimit { entry_id, depth } => {
                write!(f, "recursion limit ({depth}) hit from entry '{entry_id}'")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::PluginHook {
                plugin,
                hook,
                source,
            } => {
                write!(f, "lifecycle plugin '{plugin}' failed in {hook}: {source}")
            }
        }
    }
}

impl std::error::Error for ContextWeaverError {}

impl From<std::io::Error> for ContextWeaverError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Built-in commands & processors ──────────────────────────────────────

fn register_builtins(registry: &mut Registry) {
    // When the stdlib feature is enabled, register the full standard library.
    // Otherwise, register minimal placeholders.
    #[cfg(feature = "stdlib")]
    {
        stdlib::register(registry);
    }

    #[cfg(not(feature = "stdlib"))]
    {
        // Minimal built-in set when stdlib is disabled.
        registry.register_processor(weaver_lang::ClosureProcessor::new(
            "text",
            "upper",
            |props| {
                let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
                Ok(Value::String(text.to_uppercase()))
            },
        ));
        registry.register_processor(weaver_lang::ClosureProcessor::new(
            "text",
            "lower",
            |props| {
                let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
                Ok(Value::String(text.to_lowercase()))
            },
        ));
    }

    // $[is_active("entry_id")] — check if an entry is currently active in the local lorebook.
    // Uses the _active namespace populated by WeaverHost::set_active_entries.
    registry.register_command(IsActiveCommand);
    // $[is_active_global("entry_id")] — check if an entry is currently active in any loaded lorebook..
    // Uses the _active_global namespace populated by WeaverHost::set_active_entries.
    registry.register_command(IsActiveGlobalCommand);
}

// ── is_active command ──────────────────────────────────────────────────

/// `$[is_active("entry_id")]` — check if a lorebook entry is active
/// in the current evaluation pass.
///
/// Returns `true` if the entry ID is in the active set, `false`
/// otherwise. Works by reading the `_active` namespace which the
/// engine populates before evaluation.
struct IsActiveCommand;

impl WeaverCommand for IsActiveCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let id = args.first().and_then(|v| v.as_string()).ok_or_else(|| {
            EvalError::type_error("string", args.first().map_or("none", |v| v.type_name()))
        })?;

        let is_active = ctx
            .resolve_variable("_active", id)?
            .is_some_and(|v| v.is_truthy());

        Ok(Some(Value::Bool(is_active)))
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "is_active".to_string(),
            params: vec![ParamDef {
                name: "entry_id".to_string(),
                expected_type: Some(weaver_lang::registry::ValueType::String),
                required: true,
            }],
        }
    }
}

/// `$[is_active_global("entry_id")]` — true if an entry with this id is
/// active in ANY book. Mirrors the local-then-global resolution rule used
/// by `[[refs]]` and `<trigger>`; prefer `is_active` unless you mean to
/// reach across books.
struct IsActiveGlobalCommand;

impl WeaverCommand for IsActiveGlobalCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let id = args.first().and_then(|v| v.as_string()).ok_or_else(|| {
            EvalError::type_error("string", args.first().map_or("none", |v| v.type_name()))
        })?;

        let is_active = ctx
            .resolve_variable("_active_global", id)?
            .is_some_and(|v| v.is_truthy());

        Ok(Some(Value::Bool(is_active)))
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "is_active_global".to_string(),
            params: vec![ParamDef {
                name: "entry_id".to_string(),
                expected_type: Some(weaver_lang::registry::ValueType::String),
                required: true,
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book_with(entries: &[&str]) -> Lorebook {
        let mut book = Lorebook::new();
        for src in entries {
            book.add_entry(Entry::parse(src, None).unwrap());
        }
        book
    }

    #[test]
    fn test_multi_book_both_activate() {
        let primary = book_with(&["---\nid: sys\nconstant: true\n---\nSystem lore."]);
        let content = book_with(&["---\nid: goblin\nconstant: true\n---\nA goblin."]);

        let mut weaver = ContextWeaver::new(primary);
        weaver.add_lorebook(content);

        let blocks = weaver.assemble(&[]).unwrap();
        let ids: Vec<&str> = blocks.iter().map(|b| b.entry_id.as_str()).collect();
        assert!(ids.contains(&"sys"));
        assert!(ids.contains(&"goblin"));
    }

    #[test]
    fn test_multi_book_colliding_ids_both_activate() {
        let a = book_with(&["---\nid: goblin\nconstant: true\n---\nForest goblin."]);
        let b = book_with(&["---\nid: goblin\nconstant: true\n---\nCave goblin."]);

        let mut weaver = ContextWeaver::new(a);
        weaver.add_lorebook(b);

        let blocks = weaver.assemble(&[]).unwrap();
        let goblins: Vec<&str> = blocks
            .iter()
            .filter(|bl| bl.entry_id == "goblin")
            .map(|bl| bl.content.as_str())
            .collect();
        assert_eq!(goblins.len(), 2);
        assert!(goblins.contains(&"Forest goblin."));
        assert!(goblins.contains(&"Cave goblin."));
    }

    #[test]
    fn test_multi_book_document_reference() {
        // "goblin" lives only in the content book and never self-activates,
        // but it's still a resolution target for [[goblin]] from the primary.
        let primary = book_with(&["---\nid: intro\nconstant: true\n---\nMeet [[goblin]]."]);
        let content = book_with(&["---\nid: goblin\n---\nthe goblin"]);

        let mut weaver = ContextWeaver::new(primary);
        weaver.add_lorebook(content);

        let blocks = weaver.assemble(&[]).unwrap();
        let intro = blocks.iter().find(|b| b.entry_id == "intro").unwrap();
        assert_eq!(intro.content, "Meet the goblin.");
    }
}
