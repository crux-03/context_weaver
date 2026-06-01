//! Pluggable document/entry id resolution.
//!
//! When a template references another entry via a `[[id]]` document
//! reference, the [`WeaverHost`](crate::host::WeaverHost) must turn that bare
//! id into a compiled template to evaluate. [`IdResolver`] is the seam that
//! decides *how* an id maps to a template.
//!
//! The default, [`DefaultIdResolver`], performs a direct lookup in the
//! template map the engine hands the host each pass — the behavior
//! ContextWeaver has always had. Installing a custom resolver (via
//! [`WeaverHost::set_id_resolver`](crate::host::WeaverHost::set_id_resolver)
//! or [`ContextWeaver::set_id_resolver`](crate::ContextWeaver::set_id_resolver))
//! lets a host override that mapping — for example, to resolve ids across
//! multiple active lorebooks, or to back a command interface.
//!
//! A resolver answers only *which* template an id maps to. Cycle detection,
//! depth limiting, and recursive evaluation remain the host's responsibility,
//! per the `EvalContext` contract. Keeping the resolver a pure lookup that
//! takes the templates as a parameter (rather than owning them) is what keeps
//! it cheap to box and swap.

use std::collections::HashMap;
use std::sync::Arc;

use weaver_lang::CompiledTemplate;

/// Strategy for mapping a document/entry id to its compiled template.
///
/// Given an id and the set of templates available this pass, return the
/// template to evaluate, or `None` if the id is unknown.
pub trait IdResolver: Send + Sync {
    /// Resolve `id` against the available `templates`.
    fn resolve<'a>(
        &self,
        id: &str,
        templates: &'a HashMap<String, Arc<CompiledTemplate>>,
    ) -> Option<&'a Arc<CompiledTemplate>>;
}

/// The built-in resolver: a direct lookup by id.
///
/// Installed automatically unless a custom [`IdResolver`] is provided.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultIdResolver;

impl IdResolver for DefaultIdResolver {
    fn resolve<'a>(
        &self,
        id: &str,
        templates: &'a HashMap<String, Arc<CompiledTemplate>>,
    ) -> Option<&'a Arc<CompiledTemplate>> {
        templates.get(id)
    }
}
