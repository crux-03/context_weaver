//! Context assembler: orders evaluated entries and applies token budgets.
//!
//! After entries are activated and evaluated, the assembler sorts them
//! by slot and priority, then fits them within token budgets. The result
//! is an ordered list of [`AssembledBlock`]s that the host application
//! inserts into the LLM prompt.
//!
//! ## Slots
//!
//! Entries declare which slot they target. Slots represent functional
//! depth in the context window, from deep background to immediate
//! foreground. The host application's ContextDefinition declares which
//! slots exist and where they appear in the prompt.
//!
//! ```text
//! ┌─────────────────────────────┐
//! │  prelude                    │  ← world axioms, meta-instructions
//! │  preamble                   │  ← core lore, character backstory
//! │  backdrop                   │  ← supporting material, stable world state
//! │  setting                    │  ← situational scene info, current location
//! │  foreground                 │  ← active effects, urgent reminders near chat
//! │  [chat messages...]         │
//! │  coda                       │  ← post-chat notes, summaries
//! │  at_depth(N)                │  ← injected N msgs from end
//! └─────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::EvaluatedEntry;
use crate::lorebook::LorebookConfig;

// ── Slot ────────────────────────────────────────────────────────────────

/// Where in the prompt an entry's content should be inserted.
///
/// Slots describe functional depth rather than structural position.
/// The host application's template declares which slots are available
/// and where each one sits in the final prompt. Standard slot names
/// form a gradient from deep background (`Preamble`) to immediate
/// foreground (`Immediate`/`Aftermath`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Slot {
    /// The absolute top. World axioms, universal rules, meta-instructions.
    Prelude,
    /// Core background. Setting lore, character backstory, stable world state.
    Preamble,
    /// Situational context. Scene descriptions, current location, active relationships.
    #[default]
    Backdrop,
    /// Supporting material. Style guides, secondary character info, historical events.
    Setting,
    /// Sets the mode for upcoming content. "You are in combat", "this scene is quiet".
    Foreground,
    /// Instructions and constraints. Behavior rules, tone directives, quest objectives.
    Coda,
    /// Injected N messages from the end of the chat history.
    /// Handled separately from slot resolution during final prompt construction.
    #[serde(rename = "at_depth")]
    AtDepth(usize),
}

impl Slot {
    /// Numeric ordering index for sorting. Lower = earlier in prompt.
    fn order_index(&self) -> usize {
        match self {
            Self::Prelude => 0,
            Self::Preamble => 1,
            Self::Backdrop => 2,
            Self::Setting => 3,
            Self::Foreground => 4,
            Self::Coda => 5,
            Self::AtDepth(_) => 6,
        }
    }

    /// All standard slot variants (excluding `AtDepth`).
    pub fn standard_slots() -> Vec<Slot> {
        vec![
            Self::Prelude,
            Self::Preamble,
            Self::Backdrop,
            Self::Setting,
            Self::Foreground,
            Self::Coda,
        ]
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::AtDepth(a), Self::AtDepth(b)) => a.cmp(b),
            _ => self.order_index().cmp(&other.order_index()),
        }
    }
}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prelude => write!(f, "prelude"),
            Self::Preamble => write!(f, "preamble"),
            Self::Backdrop => write!(f, "backdrop"),
            Self::Setting => write!(f, "setting"),
            Self::Foreground => write!(f, "foreground"),
            Self::Coda => write!(f, "coda"),
            Self::AtDepth(n) => write!(f, "at_depth({n})"),
        }
    }
}

// ── Tokenizer ───────────────────────────────────────────────────────────

/// Trait for estimating token counts of text content.
///
/// The host application can provide a tokenizer that matches their
/// target model (e.g. tiktoken for OpenAI, sentencepiece for others).
/// The assembler uses this for budget enforcement.
pub trait Tokenizer: Send + Sync {
    /// Estimate the number of tokens in the given text.
    fn estimate_tokens(&self, text: &str) -> usize;
}

/// A rough tokenizer that estimates ~4 characters per token.
///
/// Reasonable for English text but inaccurate for code, CJK, or
/// other scripts. Use a proper tokenizer in production.
pub struct GuesstimationTokenizer;

impl Tokenizer for GuesstimationTokenizer {
    fn estimate_tokens(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

// ── Token budget ────────────────────────────────────────────────────────

/// Token budget tracking for assembled context.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Total budget across all entries.
    pub total: Option<usize>,
    /// Per-group budgets.
    pub groups: HashMap<String, usize>,
    /// Tokens consumed so far (global).
    consumed_total: usize,
    /// Tokens consumed per group.
    consumed_groups: HashMap<String, usize>,
}

impl TokenBudget {
    pub fn from_config(config: &LorebookConfig) -> Self {
        Self {
            total: config.token_budget,
            groups: config.group_budgets.clone(),
            consumed_total: 0,
            consumed_groups: HashMap::new(),
        }
    }

    /// Check if adding `tokens` would exceed the budget for the given group.
    /// Returns true if the entry fits.
    pub fn can_fit(&self, tokens: usize, group: Option<&str>) -> bool {
        // Check global budget
        if let Some(total) = self.total {
            if self.consumed_total + tokens > total {
                return false;
            }
        }

        // Check group budget
        if let Some(group_name) = group {
            if let Some(group_budget) = self.groups.get(group_name) {
                let consumed = self.consumed_groups.get(group_name).copied().unwrap_or(0);
                if consumed + tokens > *group_budget {
                    return false;
                }
            }
        }

        true
    }

    /// Record that tokens were consumed.
    pub fn consume(&mut self, tokens: usize, group: Option<&str>) {
        self.consumed_total += tokens;
        if let Some(group_name) = group {
            *self
                .consumed_groups
                .entry(group_name.to_string())
                .or_default() += tokens;
        }
    }
}

// ── Assembled output ────────────────────────────────────────────────────

/// A fully assembled context block, ready for prompt insertion.
#[derive(Debug, Clone)]
pub struct AssembledBlock {
    /// The entry that produced this block.
    pub entry_id: String,
    /// The resolved slot where this block should be inserted.
    pub slot: Slot,
    /// The evaluated content string.
    pub content: String,
    /// Priority (for ordering within the same slot).
    pub priority: i32,
    /// Insertion order (tie-breaker).
    pub insertion_order: i32,
    /// Inclusion group
    pub group: Option<String>,
    /// Approximate token count (for budget tracking).
    pub estimated_tokens: usize,
}

// ── Assembler ───────────────────────────────────────────────────────────

pub struct ContextAssembler;

impl ContextAssembler {
    /// Assemble evaluated entries into ordered, budgeted context blocks.
    ///
    /// Entries are:
    /// 1. Filtered (empty content is dropped)
    /// 2. Resolved to available slots (with fallback chain)
    /// 3. Sorted by priority (descending) for budget allocation
    /// 4. Fitted within token budgets (higher priority entries take precedence)
    /// 5. Re-sorted by slot, then priority for final prompt ordering
    ///
    /// `available_slots` is the set of slots present in the user's
    /// ContextDefinition template. Entries targeting unavailable slots
    /// walk their fallback chain; if no fallback matches, the entry is
    /// silently dropped.
    pub fn assemble(
        entries: Vec<EvaluatedEntry>,
        config: &LorebookConfig,
        tokenizer: &dyn Tokenizer,
        available_slots: &HashSet<Slot>,
    ) -> Vec<AssembledBlock> {
        let mut budget = TokenBudget::from_config(config);
        let mut blocks: Vec<AssembledBlock> = Vec::new();

        // Convert to blocks, resolve slots, and estimate tokens
        let mut candidates: Vec<AssembledBlock> = entries
            .into_iter()
            .filter(|e| !e.content.trim().is_empty())
            .filter_map(|e| {
                // Resolve the entry's target slot against available slots
                let resolved_slot = resolve_slot(&e.meta.slot, &e.meta.fallback, available_slots)?;

                let estimated_tokens = tokenizer.estimate_tokens(&e.content);
                Some(AssembledBlock {
                    entry_id: e.id,
                    slot: resolved_slot,
                    content: e.content,
                    priority: e.meta.priority,
                    insertion_order: e.meta.insertion_order,
                    group: e.meta.group,
                    estimated_tokens,
                })
            })
            .collect();

        // Sort: highest priority first (they get budget priority)
        candidates.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.insertion_order.cmp(&b.insertion_order))
        });

        // Apply budget constraints
        for block in candidates {
            // Use the entry's group from meta for budget tracking
            let group = &block.group;

            if budget.can_fit(block.estimated_tokens, group.as_deref()) {
                budget.consume(block.estimated_tokens, group.as_deref());
                blocks.push(block);
            }
            // else: entry is dropped due to budget constraints
        }

        // Final sort by slot for prompt assembly
        blocks.sort_by(|a, b| {
            a.slot
                .cmp(&b.slot)
                .then(b.priority.cmp(&a.priority))
                .then(a.insertion_order.cmp(&b.insertion_order))
        });

        blocks
    }
}

/// Resolve an entry's target slot against the available set.
///
/// Tries the primary slot first, then walks the fallback chain.
/// `AtDepth` always resolves (it's handled separately during prompt
/// construction). Returns `None` if no available slot is found.
fn resolve_slot(primary: &Slot, fallback: &[Slot], available: &HashSet<Slot>) -> Option<Slot> {
    // AtDepth bypasses slot availability — it's injected into chat history
    if matches!(primary, Slot::AtDepth(_)) {
        return Some(primary.clone());
    }

    if available.contains(primary) {
        return Some(primary.clone());
    }

    for slot in fallback {
        if matches!(slot, Slot::AtDepth(_)) {
            return Some(slot.clone());
        }
        if available.contains(slot) {
            return Some(slot.clone());
        }
    }

    None
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::EntryMeta;

    fn all_slots() -> HashSet<Slot> {
        Slot::standard_slots().into_iter().collect()
    }

    #[test]
    fn test_budget_tracking() {
        let mut budget = TokenBudget {
            total: Some(100),
            groups: HashMap::from([("combat".into(), 50)]),
            consumed_total: 0,
            consumed_groups: HashMap::new(),
        };

        assert!(budget.can_fit(30, None));
        budget.consume(30, None);

        assert!(budget.can_fit(70, None));
        assert!(!budget.can_fit(71, None));

        assert!(budget.can_fit(50, Some("combat")));
        budget.consume(50, Some("combat"));
        assert!(!budget.can_fit(1, Some("combat")));

        // Global budget also decremented
        assert!(!budget.can_fit(21, None));
    }

    #[test]
    fn test_empty_entries_filtered() {
        let config = LorebookConfig::default();
        let tokenizer = GuesstimationTokenizer;
        let entries = vec![
            EvaluatedEntry {
                id: "empty".into(),
                meta: make_meta("empty", 100),
                content: "   \n  ".into(),
            },
            EvaluatedEntry {
                id: "has_content".into(),
                meta: make_meta("has_content", 100),
                content: "Hello!".into(),
            },
        ];

        let blocks = ContextAssembler::assemble(entries, &config, &tokenizer, &all_slots());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].entry_id, "has_content");
    }

    #[test]
    fn test_slot_resolution_primary() {
        let available = HashSet::from([Slot::Backdrop, Slot::Preamble]);
        assert_eq!(
            resolve_slot(&Slot::Backdrop, &[], &available),
            Some(Slot::Backdrop)
        );
    }

    #[test]
    fn test_slot_resolution_fallback() {
        let available = HashSet::from([Slot::Backdrop]);
        assert_eq!(
            resolve_slot(
                &Slot::Coda,
                &[Slot::Backdrop, Slot::Preamble],
                &available,
            ),
            Some(Slot::Backdrop)
        );
    }

    #[test]
    fn test_slot_resolution_none_available() {
        let available = HashSet::from([Slot::Preamble]);
        assert_eq!(
            resolve_slot(&Slot::Coda, &[Slot::Backdrop], &available),
            None
        );
    }

    #[test]
    fn test_at_depth_always_resolves() {
        let available = HashSet::new(); // nothing available
        assert_eq!(
            resolve_slot(&Slot::AtDepth(3), &[], &available),
            Some(Slot::AtDepth(3))
        );
    }

    #[test]
    fn test_slot_ordering() {
        let mut slots = vec![
            Slot::Setting,
            Slot::Preamble,
            Slot::Prelude,
            Slot::Backdrop,
        ];
        slots.sort();
        assert_eq!(
            slots,
            vec![
                Slot::Prelude,
                Slot::Preamble,
                Slot::Backdrop,
                Slot::Setting
            ]
        );
    }

    #[test]
    fn test_guesstimation_tokenizer() {
        let tok = GuesstimationTokenizer;
        assert_eq!(tok.estimate_tokens(""), 0);
        assert_eq!(tok.estimate_tokens("abcd"), 1);
        assert_eq!(tok.estimate_tokens("Hello, world!"), 4); // 13 chars → (13+3)/4 = 4
    }

    fn make_meta(id: &str, priority: i32) -> EntryMeta {
        EntryMeta {
            id: id.to_string(),
            name: id.to_string(),
            keywords: vec![],
            regex: vec![],
            condition: None,
            scan_depth: None,
            constant: false,
            priority,
            slot: Slot::default(),
            fallback: vec![],
            insertion_order: 50,
            enabled: true,
            sticky_turns: 0,
            cooldown: 0,
            group: None,
            tags: vec![],
            extensions: Default::default(),
        }
    }
}
