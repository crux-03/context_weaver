//! Activation engine for entry matching.
//!
//! Scans chat messages for keywords and regex patterns to determine which
//! lorebook entries should activate, then optionally evaluates weaver-lang
//! condition expressions for fine-grained control.
//!
//! Activation flow:
//! 1. Constant entries are always activated.
//! 2. Messages within scan depth are searched for keyword/regex matches.
//!    Sticky entries participate in this scan — a fresh match supersedes
//!    the carry-forward and resets their countdown.
//! 3. Matched entries have their optional `condition` expression evaluated.
//! 4. Cooldown is checked — recently-fired entries are suppressed.
//! 5. Sticky entries that were NOT freshly matched are carried forward
//!    with their existing countdown.
//! 6. Results are returned as a sorted list of entry IDs.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use weaver_lang::{CompiledExpr, Registry};

use crate::ChatMessage;
use crate::host::WeaverHost;
use crate::lorebook::{BookId, Lorebook, LorebookSet};

// ── Activation result ───────────────────────────────────────────────────

/// Why an entry was activated — useful for debugging and UI display.
#[derive(Debug, Clone)]
pub enum ActivationReason {
    /// Entry is marked `constant: true`.
    Constant,
    /// Matched one or more keywords.
    Keyword { matched: Vec<String> },
    /// Matched a regex pattern.
    Regex { pattern: String },
    /// Activated by a `<trigger>` in another entry's output.
    Triggered,
    /// Still active from a previous turn (sticky).
    Sticky { remaining_turns: usize },
}

/// The result of an activation scan for a single entry.
#[derive(Debug)]
pub struct ActivationResult {
    pub book: BookId,
    pub entry_id: String,
    pub reason: ActivationReason,
    pub priority: i32,
}

// ── Activation state (persisted between turns) ──────────────────────────

/// Tracks per-entry activation state across conversation turns.
///
/// Owned by [`ContextWeaver`](crate::ContextWeaver) and updated after
/// each assembly pass. Serializable alongside persistent state for
/// save/load support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActivationState {
    /// Entry ID → turn number when it was last activated.
    last_activated: HashMap<String, usize>,
    /// Entry ID → remaining sticky turns.
    sticky_remaining: HashMap<String, usize>,
    /// The current turn number, advanced by the engine.
    current_turn: usize,
}

impl ActivationState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance to the next turn. Decrements sticky counters.
    pub fn advance_turn(&mut self) {
        self.current_turn += 1;
        self.sticky_remaining.retain(|_, turns| {
            *turns = turns.saturating_sub(1);
            *turns > 0
        });
    }

    /// Current turn number.
    pub fn current_turn(&self) -> usize {
        self.current_turn
    }

    /// Record that an entry was freshly activated on the current turn.
    ///
    /// Updates the `last_activated` timestamp (used for cooldown checks)
    /// and sets (or resets) the sticky counter.
    ///
    /// This should only be called for **fresh** activations (keyword,
    /// regex, constant, triggered) — NOT for sticky carry-forwards.
    /// The caller (assemble Phase 4) is responsible for distinguishing
    /// the two cases via [`ActivationReason`].
    ///
    /// The counter is stored as `sticky_turns + 1` because `advance_turn()`
    /// is called *before* `assemble()`. Without the +1, the first
    /// advance after activation would immediately consume one count
    /// before the entry ever gets carried forward, making
    /// `sticky_turns: N` behave as N-1 additional turns.
    pub fn record_activation(&mut self, entry_id: &str, sticky_turns: usize) {
        self.last_activated
            .insert(entry_id.to_string(), self.current_turn);
        if sticky_turns > 0 {
            self.sticky_remaining
                .insert(entry_id.to_string(), sticky_turns + 1);
        }
    }

    /// Check if an entry is on cooldown.
    pub fn is_on_cooldown(&self, entry_id: &str, cooldown: usize) -> bool {
        if cooldown == 0 {
            return false;
        }
        self.last_activated
            .get(entry_id)
            .is_some_and(|last| self.current_turn - last < cooldown)
    }

    /// Check if an entry is still sticky-active. Returns remaining turns.
    pub fn is_sticky(&self, entry_id: &str) -> Option<usize> {
        self.sticky_remaining.get(entry_id).copied()
    }

    /// Get all currently-sticky entry IDs with their remaining turns.
    pub fn sticky_entries(&self) -> impl Iterator<Item = (&str, usize)> {
        self.sticky_remaining
            .iter()
            .map(|(id, turns)| (id.as_str(), *turns))
    }
}

// ── Engine ──────────────────────────────────────────────────────────────

pub struct ActivationEngine;

impl ActivationEngine {
    /// Scan chat messages and determine which entries should activate.
    ///
    /// This is the main entry point for Tier 1 activation. It handles:
    /// - Constant entries (always active)
    /// - Keyword and regex matching against recent messages
    /// - Condition expression evaluation (Tier 1.5)
    /// - Cooldown suppression
    /// - Sticky entries carried from previous turns (as fallback)
    ///
    /// Sticky entries participate in keyword/regex scanning. If they
    /// re-match, they are treated as a fresh activation (which resets
    /// their sticky counter in Phase 4 of assembly). If they don't
    /// re-match, they are carried forward with their existing countdown.
    ///
    /// Returns entry IDs in priority order (highest first).
    pub fn scan(
        lorebook: &Lorebook,
        messages: &[ChatMessage],
        host: &mut WeaverHost,
        registry: &Registry,
        state: &ActivationState,
    ) -> Vec<ActivationResult> {
        let config = &lorebook.config;
        let case_sensitive = config.case_sensitive_keywords;

        let mut activated: Vec<ActivationResult> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect sticky entries. These will be carried forward ONLY if
        // they don't get a fresh keyword/regex/constant activation below.
        let mut sticky_carry: HashMap<String, (usize, i32)> = HashMap::new();
        for (entry_id, remaining) in state.sticky_entries() {
            if let Some(entry) = lorebook.get_entry(entry_id) {
                if entry.meta.enabled {
                    sticky_carry.insert(entry_id.to_string(), (remaining, entry.meta.priority));
                }
            }
        }

        // Scan all entries — sticky entries are NOT pre-added to `seen`,
        // so they go through the normal keyword/regex path.
        for entry in lorebook.active_entries() {
            let meta = &entry.meta;

            // Skip entries on cooldown
            if state.is_on_cooldown(&meta.id, meta.cooldown) {
                continue;
            }

            // Constant entries always activate (skip keyword/regex check)
            if meta.constant {
                if check_condition(&entry.condition, host, registry) {
                    seen.insert(meta.id.clone());
                    sticky_carry.remove(&meta.id);
                    activated.push(ActivationResult {
                        book: BookId(0),
                        entry_id: meta.id.clone(),
                        reason: ActivationReason::Constant,
                        priority: meta.priority,
                    });
                }
                continue;
            }

            // Determine scan depth for this entry
            let scan_depth = meta.scan_depth.unwrap_or(config.default_scan_depth);

            // Get the messages within scan depth
            let scan_messages = if messages.len() > scan_depth {
                &messages[messages.len() - scan_depth..]
            } else {
                messages
            };

            // Build the text corpus to search
            let corpus: String = scan_messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            let corpus_normalized = if case_sensitive {
                corpus.clone()
            } else {
                corpus.to_lowercase()
            };

            // Try keyword match
            let keyword_match = check_keywords(&meta.keywords, &corpus_normalized, case_sensitive);

            // Try regex match (using pre-compiled regexes from Entry)
            let regex_match = if keyword_match.is_none() {
                check_regex_compiled(&entry.compiled_regex, &meta.regex, &corpus)
            } else {
                None
            };

            // Determine activation reason (keyword takes precedence)
            let reason = keyword_match.or(regex_match);

            if let Some(reason) = reason {
                // Check condition
                if check_condition(&entry.condition, host, registry) {
                    seen.insert(meta.id.clone());
                    sticky_carry.remove(&meta.id);
                    activated.push(ActivationResult {
                        book: BookId(0),
                        entry_id: meta.id.clone(),
                        reason,
                        priority: meta.priority,
                    });
                }
            }
        }

        // Carry forward sticky entries that weren't freshly activated
        for (id, (remaining, priority)) in sticky_carry {
            if !seen.contains(&id) {
                activated.push(ActivationResult {
                    book: BookId(0),
                    entry_id: id,
                    reason: ActivationReason::Sticky {
                        remaining_turns: remaining,
                    },
                    priority,
                });
            }
        }

        // Sort by priority (descending)
        activated.sort_by(|a, b| b.priority.cmp(&a.priority));

        activated
    }

    /// Filter triggered entry IDs through activation rules.
    ///
    /// Called after an evaluation pass drains trigger IDs from the host.
    /// Applies cooldown and condition checks to the triggered candidates.
    pub fn filter_triggered(
        books: &LorebookSet,
        triggered_ids: &[(BookId, String)],
        already_active: &[(BookId, String)],
        host: &mut WeaverHost,
        registry: &Registry,
        state: &ActivationState,
    ) -> Vec<ActivationResult> {
        let mut results = Vec::new();

        for (book, id) in triggered_ids {
            // Skip already-active entries UNLESS they are sticky.
            let key = (*book, id.clone());
            if already_active.contains(&key) && state.is_sticky(id).is_none() {
                continue;
            }

            let Some(lorebook) = books.get(*book) else {
                continue;
            };
            if let Some(entry) = lorebook.get_entry(id) {
                let meta = &entry.meta;
                if !meta.enabled {
                    continue;
                }
                if state.is_on_cooldown(&meta.id, meta.cooldown) {
                    continue;
                }
                if check_condition(&entry.condition, host, registry) {
                    results.push(ActivationResult {
                        book: *book,
                        entry_id: meta.id.clone(),
                        reason: ActivationReason::Triggered,
                        priority: meta.priority,
                    });
                }
            }
        }

        results.sort_by(|a, b| b.priority.cmp(&a.priority));
        results
    }
}

// ── Condition evaluation ────────────────────────────────────────────────

/// Evaluate an entry's optional condition expression.
///
/// Returns `true` if the condition is met (or if there is no condition).
/// Failed evaluations (type errors, missing variables, etc.) return `false`
/// — a broken condition should not silently activate an entry.
fn check_condition(
    condition: &Option<Arc<CompiledExpr>>,
    host: &mut WeaverHost,
    registry: &Registry,
) -> bool {
    match condition {
        None => true,
        Some(expr) => match expr.evaluate(host, registry) {
            Ok(val) => val.is_truthy(),
            Err(_) => false,
        },
    }
}

// ── Matching helpers ────────────────────────────────────────────────────

fn check_keywords(
    keywords: &[String],
    corpus: &str,
    case_sensitive: bool,
) -> Option<ActivationReason> {
    if keywords.is_empty() {
        return None;
    }

    let matched: Vec<String> = keywords
        .iter()
        .filter(|kw| {
            let kw_normalized = if case_sensitive {
                kw.to_string()
            } else {
                kw.to_lowercase()
            };
            corpus.contains(&kw_normalized)
        })
        .cloned()
        .collect();

    if matched.is_empty() {
        None
    } else {
        Some(ActivationReason::Keyword { matched })
    }
}

/// Check pre-compiled regex patterns against the corpus.
///
/// Uses the cached [`Regex`] objects from [`Entry`] rather than
/// recompiling from the raw pattern strings on every scan pass.
/// The raw patterns are passed alongside for the [`ActivationReason`]
/// diagnostic.
fn check_regex_compiled(
    compiled: &[Regex],
    raw_patterns: &[String],
    corpus: &str,
) -> Option<ActivationReason> {
    for (re, pattern) in compiled.iter().zip(raw_patterns.iter()) {
        if re.is_match(corpus) {
            return Some(ActivationReason::Regex {
                pattern: pattern.clone(),
            });
        }
    }
    None
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_state_cooldown() {
        let mut state = ActivationState::new();
        state.current_turn = 5;
        state.record_activation("entry_a", 0);

        state.current_turn = 6;
        assert!(state.is_on_cooldown("entry_a", 3));
        state.current_turn = 7;
        assert!(state.is_on_cooldown("entry_a", 3));
        state.current_turn = 8;
        assert!(!state.is_on_cooldown("entry_a", 3));
        assert!(!state.is_on_cooldown("entry_b", 3));
    }

    #[test]
    fn test_activation_state_sticky() {
        let mut state = ActivationState::new();
        // sticky_turns=3 → stored as 4 internally (3+1)
        state.record_activation("entry_a", 3);

        // Active for 3 advance_turn calls after activation
        assert_eq!(state.is_sticky("entry_a"), Some(4));
        state.advance_turn(); // turn after activation
        assert_eq!(state.is_sticky("entry_a"), Some(3));
        state.advance_turn(); // additional turn 1
        assert_eq!(state.is_sticky("entry_a"), Some(2));
        state.advance_turn(); // additional turn 2
        assert_eq!(state.is_sticky("entry_a"), Some(1));
        state.advance_turn(); // additional turn 3 → expired
        assert_eq!(state.is_sticky("entry_a"), None);
    }

    #[test]
    fn test_sticky_refresh_resets_countdown() {
        let mut state = ActivationState::new();
        // sticky_turns=2 → stored as 3 internally
        state.record_activation("entry_a", 2);
        assert_eq!(state.is_sticky("entry_a"), Some(3));

        state.advance_turn();
        assert_eq!(state.is_sticky("entry_a"), Some(2));

        state.advance_turn();
        assert_eq!(state.is_sticky("entry_a"), Some(1));

        // Fresh re-activation (e.g. keyword matched again) resets counter
        state.record_activation("entry_a", 2);
        assert_eq!(state.is_sticky("entry_a"), Some(3));

        // Full countdown from the refresh point
        state.advance_turn();
        assert_eq!(state.is_sticky("entry_a"), Some(2));
        state.advance_turn();
        assert_eq!(state.is_sticky("entry_a"), Some(1));
        state.advance_turn();
        assert_eq!(state.is_sticky("entry_a"), None);
    }

    #[test]
    fn test_activation_state_advance_tracks_turn() {
        let mut state = ActivationState::new();
        assert_eq!(state.current_turn(), 0);
        state.advance_turn();
        assert_eq!(state.current_turn(), 1);
        state.advance_turn();
        assert_eq!(state.current_turn(), 2);
    }

    #[test]
    fn test_activation_state_serialization() {
        let mut state = ActivationState::new();
        state.record_activation("entry_a", 3);
        state.advance_turn();

        let json = serde_json::to_string(&state).unwrap();
        let restored: ActivationState = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.current_turn(), 1);
        assert_eq!(restored.is_sticky("entry_a"), Some(3));
    }

    #[test]
    fn test_condition_none_is_true() {
        let mut host = crate::host::WeaverHost::from_lorebook_config(
            &crate::lorebook::LorebookConfig::default(),
        );
        let registry = Registry::new();
        assert!(check_condition(&None, &mut host, &registry));
    }

    #[test]
    fn test_condition_true_expression() {
        let mut host = crate::host::WeaverHost::from_lorebook_config(
            &crate::lorebook::LorebookConfig::default(),
        );
        host.set_host_variable("state", "level", weaver_lang::Value::Number(10.0));
        let registry = Registry::new();

        let expr = Arc::new(CompiledExpr::compile("{{state:level}} > 5").unwrap());
        assert!(check_condition(&Some(expr), &mut host, &registry));
    }

    #[test]
    fn test_condition_false_expression() {
        let mut host = crate::host::WeaverHost::from_lorebook_config(
            &crate::lorebook::LorebookConfig::default(),
        );
        host.set_host_variable("state", "level", weaver_lang::Value::Number(3.0));
        let registry = Registry::new();

        let expr = Arc::new(CompiledExpr::compile("{{state:level}} > 5").unwrap());
        assert!(!check_condition(&Some(expr), &mut host, &registry));
    }

    #[test]
    fn test_condition_error_returns_false() {
        let mut host = crate::host::WeaverHost::from_lorebook_config(
            &crate::lorebook::LorebookConfig::default(),
        );
        let registry = Registry::new();

        let expr = Arc::new(CompiledExpr::compile("{{state:missing}} > 5").unwrap());
        assert!(!check_condition(&Some(expr), &mut host, &registry));
    }
}
