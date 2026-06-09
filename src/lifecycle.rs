//! Lifecycle plugin system: hooks into the assembly pipeline.
//!
//! Lifecycle plugins are distinct from [`Plugin`](crate::Plugin), which
//! extends the weaver-lang registry. Where `Plugin` adds new processors and
//! commands that templates can invoke, [`LifecyclePlugin`] observes and
//! mutates the engine's state as it moves through activation, evaluation,
//! and assembly.
//!
//! ## Use cases
//!
//! - **PII redaction**: rewrite messages before activation, or strip
//!   personal data from final blocks
//! - **Analytics and tracing**: count activations, time phases, log decisions
//! - **Forced inclusion**: inject entries that didn't match any keyword
//! - **Content transformation**: post-process evaluated content (localize,
//!   spell-check, censor)
//! - **Save/load hooks**: snapshot state on every turn advance
//!
//! ## Hook ordering
//!
//! Plugins fire in registration order. Within a plugin, hooks fire in
//! pipeline order:
//!
//! ```text
//! pre_activation
//!   → activate
//!   → post_activation
//!   → for each entry:
//!       pre_evaluate → evaluate → post_evaluate
//!   → trigger pass:
//!       on_trigger_fired → filter → evaluate new entries
//!   → assemble
//!   → post_assemble
//! ```
//!
//! `on_turn_advance` fires from
//! [`ContextWeaver::advance_turn`](crate::ContextWeaver::advance_turn), which
//! is independent of the assembly pipeline.
//!
//! Do not depend on inter-plugin ordering for correctness. If your plugin
//! needs to run after another, merge them or use shared state.
//!
//! ## Errors
//!
//! Any hook returning `Err(HookError)` aborts the pipeline with
//! [`ContextWeaverError::PluginHook`](crate::ContextWeaverError::PluginHook).
//! Hosts that want resilience should swallow errors inside the hook itself
//! and return `Ok(())`.
//!
//! ## Examples
//!
//! ### Direct trait implementation (stateful)
//!
//! ```rust,ignore
//! use context_weaver::lifecycle::{LifecyclePlugin, PostAssembleCtx, HookError};
//!
//! struct BlockCounter { count: usize }
//!
//! impl LifecyclePlugin for BlockCounter {
//!     fn name(&self) -> &str { "block_counter" }
//!
//!     fn post_assemble(&mut self, ctx: &mut PostAssembleCtx<'_>)
//!         -> Result<(), HookError>
//!     {
//!         self.count += ctx.blocks.len();
//!         Ok(())
//!     }
//! }
//!
//! weaver.register_lifecycle(BlockCounter { count: 0 });
//! ```
//!
//! ### Closure adapter (stateless or simple)
//!
//! ```rust,ignore
//! use context_weaver::lifecycle::FnLifecycle;
//!
//! weaver.register_lifecycle(
//!     FnLifecycle::new("logger")
//!         .on_post_assemble(|ctx| {
//!             eprintln!("assembled {} blocks", ctx.blocks.len());
//!             Ok(())
//!         })
//! );
//! ```

use crate::activation::{ActivationResult, ActivationState};
use crate::assembler::AssembledBlock;
use crate::entry::Entry;
use crate::lorebook::Lorebook;
use crate::{BookId, ChatMessage};

// ── Errors ──────────────────────────────────────────────────────────────

/// An error returned from a lifecycle hook.
///
/// A non-`Ok` result from any hook aborts the current assembly or
/// turn-advance and surfaces to the caller as
/// [`ContextWeaverError::PluginHook`](crate::ContextWeaverError::PluginHook).
#[derive(Debug)]
pub struct HookError {
    pub message: String,
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl HookError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    pub fn with_source<E>(message: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl std::fmt::Display for HookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(src) = &self.source {
            write!(f, ": {src}")?;
        }
        Ok(())
    }
}

impl std::error::Error for HookError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|b| b.as_ref() as &(dyn std::error::Error + 'static))
    }
}

// ── Contexts ────────────────────────────────────────────────────────────

/// Context passed to [`LifecyclePlugin::pre_activation`].
///
/// Hooks may mutate `messages` to redact, inject, or transform conversation
/// content before activation matching.
pub struct PreActivationCtx<'a> {
    pub messages: &'a mut Vec<ChatMessage>,
    pub turn: usize,
}

/// Context passed to [`LifecyclePlugin::post_activation`].
///
/// Hooks may mutate `results` to filter out, reorder, or force-add
/// activations before the engine truncates to `max_active_entries`.
pub struct PostActivationCtx<'a> {
    pub results: &'a mut Vec<ActivationResult>,
    pub lorebook: &'a Lorebook,
    pub turn: usize,
}

/// Context passed to [`LifecyclePlugin::pre_evaluate`].
///
/// Hooks may set `*skip = true` to drop this entry without evaluating it.
/// All hooks fire even after another sets the flag, so each plugin sees the
/// cumulative decision. Resetting the flag to `false` is technically
/// permitted but considered bad practice.
pub struct PreEvaluateCtx<'a> {
    pub entry: &'a Entry,
    pub skip: &'a mut bool,
}

/// Context passed to [`LifecyclePlugin::post_evaluate`].
///
/// Hooks may mutate `content` to transform the evaluated output: redact,
/// translate, append boilerplate, etc.
pub struct PostEvaluateCtx<'a> {
    pub entry: &'a Entry,
    pub content: &'a mut String,
}

/// Context passed to [`LifecyclePlugin::post_assemble`].
///
/// Hooks may mutate `blocks` to reorder, filter, or transform the final
/// output. This is the last point before the engine returns to the caller.
pub struct PostAssembleCtx<'a> {
    pub blocks: &'a mut Vec<AssembledBlock>,
    pub lorebook: &'a Lorebook,
}

/// Context passed to [`LifecyclePlugin::on_turn_advance`].
///
/// Hooks may inspect or mutate the activation state. Common uses include
/// snapshotting state for save files, force-expiring sticky entries, or
/// resetting cooldowns.
pub struct TurnAdvanceCtx<'a> {
    pub state: &'a mut ActivationState,
}

/// Context passed to [`LifecyclePlugin::on_trigger_fired`].
///
/// Fires after `drain_triggered_entries` returns a non-empty batch, before
/// the engine filters those through cooldown and condition checks. Hooks may
/// mutate `triggered_ids` to add, remove, or reorder.
pub struct TriggerCtx<'a> {
    pub triggered_ids: &'a mut Vec<(BookId, String)>,
    pub pass_number: usize,
}

// ── LifecyclePlugin trait ───────────────────────────────────────────────

/// A plugin that hooks into the engine's assembly pipeline.
///
/// Distinct from [`Plugin`](crate::Plugin), which extends the registry. See
/// the [module-level documentation](crate::lifecycle) for examples and
/// ordering guarantees.
pub trait LifecyclePlugin: Send + Sync {
    /// Unique identifier for this plugin. Used in error messages.
    fn name(&self) -> &str;

    fn pre_activation(&mut self, _ctx: &mut PreActivationCtx<'_>) -> Result<(), HookError> {
        Ok(())
    }

    fn post_activation(&mut self, _ctx: &mut PostActivationCtx<'_>) -> Result<(), HookError> {
        Ok(())
    }

    fn pre_evaluate(&mut self, _ctx: &mut PreEvaluateCtx<'_>) -> Result<(), HookError> {
        Ok(())
    }

    fn post_evaluate(&mut self, _ctx: &mut PostEvaluateCtx<'_>) -> Result<(), HookError> {
        Ok(())
    }

    fn post_assemble(&mut self, _ctx: &mut PostAssembleCtx<'_>) -> Result<(), HookError> {
        Ok(())
    }

    fn on_turn_advance(&mut self, _ctx: &mut TurnAdvanceCtx<'_>) -> Result<(), HookError> {
        Ok(())
    }

    fn on_trigger_fired(&mut self, _ctx: &mut TriggerCtx<'_>) -> Result<(), HookError> {
        Ok(())
    }
}

// ── FnLifecycle adapter ─────────────────────────────────────────────────

type PreActivationFn =
    Box<dyn FnMut(&mut PreActivationCtx<'_>) -> Result<(), HookError> + Send + Sync>;
type PostActivationFn =
    Box<dyn FnMut(&mut PostActivationCtx<'_>) -> Result<(), HookError> + Send + Sync>;
type PreEvaluateFn = Box<dyn FnMut(&mut PreEvaluateCtx<'_>) -> Result<(), HookError> + Send + Sync>;
type PostEvaluateFn =
    Box<dyn FnMut(&mut PostEvaluateCtx<'_>) -> Result<(), HookError> + Send + Sync>;
type PostAssembleFn =
    Box<dyn FnMut(&mut PostAssembleCtx<'_>) -> Result<(), HookError> + Send + Sync>;
type TurnAdvanceFn = Box<dyn FnMut(&mut TurnAdvanceCtx<'_>) -> Result<(), HookError> + Send + Sync>;
type TriggerFiredFn = Box<dyn FnMut(&mut TriggerCtx<'_>) -> Result<(), HookError> + Send + Sync>;

/// Closure-based [`LifecyclePlugin`] for simple, one-off cases.
///
/// For anything stateful or complex, implement `LifecyclePlugin` directly.
/// See the module documentation for examples.
pub struct FnLifecycle {
    name: String,
    pre_activation_fn: Option<PreActivationFn>,
    post_activation_fn: Option<PostActivationFn>,
    pre_evaluate_fn: Option<PreEvaluateFn>,
    post_evaluate_fn: Option<PostEvaluateFn>,
    post_assemble_fn: Option<PostAssembleFn>,
    turn_advance_fn: Option<TurnAdvanceFn>,
    trigger_fired_fn: Option<TriggerFiredFn>,
}

macro_rules! fn_lifecycle_setter {
    ($method:ident, $field:ident, $ctx:ident) => {
        pub fn $method<F>(mut self, f: F) -> Self
        where
            F: FnMut(&mut $ctx<'_>) -> Result<(), HookError> + Send + Sync + 'static,
        {
            self.$field = Some(Box::new(f));
            self
        }
    };
}

impl FnLifecycle {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pre_activation_fn: None,
            post_activation_fn: None,
            pre_evaluate_fn: None,
            post_evaluate_fn: None,
            post_assemble_fn: None,
            turn_advance_fn: None,
            trigger_fired_fn: None,
        }
    }

    fn_lifecycle_setter!(on_pre_activation, pre_activation_fn, PreActivationCtx);
    fn_lifecycle_setter!(on_post_activation, post_activation_fn, PostActivationCtx);
    fn_lifecycle_setter!(on_pre_evaluate, pre_evaluate_fn, PreEvaluateCtx);
    fn_lifecycle_setter!(on_post_evaluate, post_evaluate_fn, PostEvaluateCtx);
    fn_lifecycle_setter!(on_post_assemble, post_assemble_fn, PostAssembleCtx);
    fn_lifecycle_setter!(on_turn_advance, turn_advance_fn, TurnAdvanceCtx);
    fn_lifecycle_setter!(on_trigger_fired, trigger_fired_fn, TriggerCtx);
}

impl LifecyclePlugin for FnLifecycle {
    fn name(&self) -> &str {
        &self.name
    }

    fn pre_activation(&mut self, ctx: &mut PreActivationCtx<'_>) -> Result<(), HookError> {
        match &mut self.pre_activation_fn {
            Some(f) => f(ctx),
            None => Ok(()),
        }
    }

    fn post_activation(&mut self, ctx: &mut PostActivationCtx<'_>) -> Result<(), HookError> {
        match &mut self.post_activation_fn {
            Some(f) => f(ctx),
            None => Ok(()),
        }
    }

    fn pre_evaluate(&mut self, ctx: &mut PreEvaluateCtx<'_>) -> Result<(), HookError> {
        match &mut self.pre_evaluate_fn {
            Some(f) => f(ctx),
            None => Ok(()),
        }
    }

    fn post_evaluate(&mut self, ctx: &mut PostEvaluateCtx<'_>) -> Result<(), HookError> {
        match &mut self.post_evaluate_fn {
            Some(f) => f(ctx),
            None => Ok(()),
        }
    }

    fn post_assemble(&mut self, ctx: &mut PostAssembleCtx<'_>) -> Result<(), HookError> {
        match &mut self.post_assemble_fn {
            Some(f) => f(ctx),
            None => Ok(()),
        }
    }

    fn on_turn_advance(&mut self, ctx: &mut TurnAdvanceCtx<'_>) -> Result<(), HookError> {
        match &mut self.turn_advance_fn {
            Some(f) => f(ctx),
            None => Ok(()),
        }
    }

    fn on_trigger_fired(&mut self, ctx: &mut TriggerCtx<'_>) -> Result<(), HookError> {
        match &mut self.trigger_fired_fn {
            Some(f) => f(ctx),
            None => Ok(()),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn fn_lifecycle_invokes_set_hook() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let mut plugin = FnLifecycle::new("counter").on_pre_activation(move |_ctx| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        let mut messages: Vec<ChatMessage> = vec![];
        let mut ctx = PreActivationCtx {
            messages: &mut messages,
            turn: 0,
        };

        plugin.pre_activation(&mut ctx).unwrap();
        plugin.pre_activation(&mut ctx).unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn fn_lifecycle_unset_hooks_are_noop() {
        let mut plugin = FnLifecycle::new("empty");
        let mut messages: Vec<ChatMessage> = vec![];
        let mut ctx = PreActivationCtx {
            messages: &mut messages,
            turn: 0,
        };
        assert!(plugin.pre_activation(&mut ctx).is_ok());

        let mut blocks: Vec<AssembledBlock> = vec![];
        let lorebook = Lorebook::new();
        let mut ctx = PostAssembleCtx {
            blocks: &mut blocks,
            lorebook: &lorebook,
        };
        assert!(plugin.post_assemble(&mut ctx).is_ok());
    }

    #[test]
    fn hook_error_display_with_source() {
        let inner = std::io::Error::new(std::io::ErrorKind::Other, "inner cause");
        let err = HookError::with_source("hook failed", inner);
        let display = err.to_string();
        assert!(display.contains("hook failed"));
        assert!(display.contains("inner cause"));
    }

    #[test]
    fn hook_error_display_without_source() {
        let err = HookError::new("simple failure");
        assert_eq!(err.to_string(), "simple failure");
    }

    #[test]
    fn fn_lifecycle_pre_activation_can_mutate_messages() {
        let mut plugin = FnLifecycle::new("injector").on_pre_activation(|ctx| {
            ctx.messages.push(ChatMessage::system("[injected]"));
            Ok(())
        });

        let mut messages = vec![ChatMessage::user("hello")];
        let mut ctx = PreActivationCtx {
            messages: &mut messages,
            turn: 0,
        };
        plugin.pre_activation(&mut ctx).unwrap();

        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn fn_lifecycle_error_propagates() {
        let mut plugin =
            FnLifecycle::new("failer").on_pre_activation(|_ctx| Err(HookError::new("nope")));

        let mut messages: Vec<ChatMessage> = vec![];
        let mut ctx = PreActivationCtx {
            messages: &mut messages,
            turn: 0,
        };
        let err = plugin.pre_activation(&mut ctx).unwrap_err();
        assert_eq!(err.message, "nope");
    }
}
