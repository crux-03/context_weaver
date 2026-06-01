//! Pluggable document/entry id resolution.
//!
//! When a template references another entry via a `[[id]]` document
//! reference, the [`WeaverHost`](crate::host::WeaverHost) must turn that bare
//! id into a compiled template to evaluate. [`IdResolver`] is the seam that
//! decides *how* an id maps to a template.
//!
//! With multiple lorebooks active, an id is no longer globally unique, so
//! resolution is relative to the book the reference fired *from* — the
//! "local" book. [`DefaultIdResolver`] resolves the local book first, then
//! falls back to the remaining books in registration order (first match
//! wins). Installing a custom resolver (via
//! [`WeaverHost::set_id_resolver`](crate::host::WeaverHost::set_id_resolver)
//! or [`ContextWeaver::set_id_resolver`](crate::ContextWeaver::set_id_resolver))
//! overrides that mapping — e.g. to back a command interface.
//!
//! A resolver answers only *which* book + template an id maps to. Cycle
//! detection, depth limiting, and recursive evaluation remain the host's
//! responsibility, per the `EvalContext` contract. Keeping the resolver a
//! pure lookup that takes the templates as a parameter (rather than owning
//! them) is what keeps it cheap to box and swap.

use std::collections::HashMap;
use std::sync::Arc;

use weaver_lang::CompiledTemplate;

use crate::lorebook::BookId;

/// The compiled templates for a single book, keyed by entry id.
pub type EntryTemplates = HashMap<String, Arc<CompiledTemplate>>;

/// Compiled templates for every active book, partitioned by book and
/// preserving registration order (used for the global-fallback scan).
///
/// Owned by the host and replaced each evaluation pass; the resolver only
/// borrows it.
#[derive(Default)]
pub struct BookTemplates {
    books: Vec<EntryTemplates>,
}

impl BookTemplates {
    pub fn new() -> Self {
        Self { books: Vec::new() }
    }

    /// Append a book's templates, returning its assigned [`BookId`].
    pub fn push(&mut self, templates: EntryTemplates) -> BookId {
        let id = BookId(self.books.len());
        self.books.push(templates);
        id
    }

    pub fn len(&self) -> usize {
        self.books.len()
    }

    pub fn is_empty(&self) -> bool {
        self.books.is_empty()
    }

    /// Look up `id` within a specific book.
    pub fn get(&self, book: BookId, id: &str) -> Option<&Arc<CompiledTemplate>> {
        self.books.get(book.0)?.get(id)
    }

    /// Iterate `(BookId, templates)` for every book except `exclude`, in
    /// registration order. Used for the global-fallback step.
    pub fn iter_except(
        &self,
        exclude: Option<BookId>,
    ) -> impl Iterator<Item = (BookId, &EntryTemplates)> {
        self.books
            .iter()
            .enumerate()
            .map(|(i, m)| (BookId(i), m))
            .filter(move |(b, _)| Some(*b) != exclude)
    }
}

/// A resolved reference: which book the id resolved to, plus its template.
///
/// The host pushes `book` onto its eval stack so that nested references
/// inside the resolved entry resolve relative to *that* entry's book.
pub struct ResolvedRef<'a> {
    pub book: BookId,
    pub template: &'a Arc<CompiledTemplate>,
}

/// Strategy for mapping a document/entry id to its book and compiled template.
pub trait IdResolver: Send + Sync {
    /// Resolve `id`, where `origin` is the book of the currently-evaluating
    /// entry (`None` for a top-level resolution with no local book).
    fn resolve<'a>(
        &self,
        id: &str,
        origin: Option<BookId>,
        books: &'a BookTemplates,
    ) -> Option<ResolvedRef<'a>>;
}

/// The built-in resolver: local book first, then the remaining books in
/// registration order (first match wins).
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultIdResolver;

impl IdResolver for DefaultIdResolver {
    fn resolve<'a>(
        &self,
        id: &str,
        origin: Option<BookId>,
        books: &'a BookTemplates,
    ) -> Option<ResolvedRef<'a>> {
        // 1. Local book.
        if let Some(origin) = origin {
            if let Some(template) = books.get(origin, id) {
                return Some(ResolvedRef {
                    book: origin,
                    template,
                });
            }
        }
        // 2. General context: the other books, in registration order.
        books
            .iter_except(origin)
            .find_map(|(book, map)| map.get(id).map(|template| ResolvedRef { book, template }))
    }
}
