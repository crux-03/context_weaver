# ContextWeaver

A lorebook engine for LLM role-playing applications, built on [weaver-lang](https://github.com/crux-03/weaver_lang).

ContextWeaver manages a collection of entries that are selectively activated based on conversation context and assembled into the final prompt sent to the model. It is roughly comparable in spirit to SillyTavern's lorebook system, with first-class scripting via weaver-lang and a strict separation between activation, evaluation, and assembly phases.

## Design philosophy

ContextWeaver favors a small set of composable primitives over a large set of dedicated features. The activation engine, the template language, the namespace system, and the lifecycle hooks are designed to combine, not to cover every use case individually. When a host need does not map to a built-in field on an entry, the expectation is that it can be expressed through composition rather than waiting on engine support.

This shows up in places where the entry format does not have a dedicated field for a feature that other lorebook systems expose directly. Selective keyword logic, for example, is not a separate setting. It is what falls out of `condition` accepting any weaver-lang expression that collapses to a boolean:

```yaml
condition: '({{state:location}} == "forest" || {{state:location}} == "swamp") && !{{state:safe_zone}}'
```

The same applies to per-entry activation probability. Rather than a numeric slider, randomization is something the template expresses directly:

```
$[set_var("local:rng", @[core.rng(min: 0, max: 5)])]
{# if {{local:rng}} == 0 #}<trigger id="goblin_pack">
{# elif {{local:rng}} == 1 #}<trigger id="lone_wolf">
{# elif {{local:rng}} == 2 #}<trigger id="bandit_ambush">
{# endif #}
```

The roll is a normal variable. It can be inspected, logged through a lifecycle hook, promoted to `state:` to persist across turns, weighted by changing the branch ranges, or gated by an outer condition. The same approach extends to inclusion groups, weighted selection, conditional spawning, and similar patterns that other systems expose as opaque settings.

The tradeoff is real. Simple use cases require more typing than a checkbox, and authors must understand the primitives before they can express what they want. In exchange, the behavior of an entry is fully readable from its source. There is no separate scripting layer to install, no implicit engine logic running alongside the entry, and no settings whose interaction with the rest of the system is undocumented. Distribution stays simple because everything an entry does travels with the entry itself.

In ContextWeaver, **your entries define the system**

## Status

Pre-release. The architecture is stable and the test suite is comprehensive, but the public API may shift before v0.1.0 is tagged. See the [Roadmap](#roadmap) for what is still pending.

## Quick start

```rust
use context_weaver::{ContextWeaver, Lorebook, ChatMessage};

// Load a lorebook from disk
let book = Lorebook::load_from_directory("./my_character/lorebook")?;

let mut weaver = ContextWeaver::new(book);

// Feed host data into read-only namespaces
weaver.set_variable("char", "name", "Aria");
weaver.set_variable("char", "class", "Mage");
weaver.set_variable("user", "name", "Player");

// Provide conversation context
let messages = vec![
    ChatMessage::user("I walk into the dark forest"),
    ChatMessage::assistant("The trees close in around you..."),
];

// Run activation, evaluation, and assembly
let blocks = weaver.assemble(&messages)?;
for block in &blocks {
    println!("[{}] {}", block.slot, block.content);
}
```

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│  Host Application (LLM frontend)                    │
│  Provides: chat history, character data, user prefs │
│  Receives: assembled context blocks                 │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  ContextWeaver                                      │
│                                                     │
│  Lorebook → Activation → Evaluation → Assembly      │
│                                                     │
│  Plugin Registry (processors and commands)          │
│  Lifecycle Plugins (pipeline hooks)                 │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│  weaver-lang (template evaluation)                  │
└─────────────────────────────────────────────────────┘
```

The pipeline is four phases:

1. **Activation.** Scan recent messages for keyword or regex matches, check conditions, carry forward sticky entries, suppress entries on cooldown.
2. **Evaluation.** Run each activated entry's template against the host context. Collect any `<trigger>` activations and run a bounded number of follow-up passes.
3. **Assembly.** Resolve each entry to its target slot (with fallback chain), apply token budgets, and sort by slot then priority.
4. **Output.** Return ordered `AssembledBlock`s for the host to splice into the final prompt.

## Entry format

A `.weaver` file is YAML frontmatter followed by a weaver-lang body:

```yaml
---
id: dark_forest
name: Dark Forest Description
keywords: ["dark forest", "shadowed path"]
condition: '{{state:location}} == "forest"'
priority: 150
slot: foundation
fallback: [context]
sticky_turns: 2
---
{# if {{state:level}} > 5 #}
The ancient dark forest recognizes a seasoned adventurer.
{# else #}
The dark forest looms, its shadows deep and menacing.
{# endif #}
Threat level: @[text.upper(text: "high")]
$[set_var("state:visited_forest", true)]
```

A directory of these files plus a `lorebook.yaml` config makes a lorebook:

```text
my_character/
  lorebook.yaml
  entries/
    dark_forest.weaver
    combat_system.weaver
    npc_merchant.weaver
```

## Core features

### Activation

* **Keyword matching** against a configurable scan depth (case-insensitive by default).
* **Regex patterns**, compiled once at parse time and cached on the entry.
* **Condition expressions** in weaver-lang for fine-grained gating, e.g. `{{state:level}} > 5 && @[array.contains(items: {{state:flags}}, value: "questing")]`.
* **Constant entries** always active regardless of context.
* **Sticky entries** that persist for N turns after firing, with fresh re-matches resetting the countdown.
* **Cooldowns** that suppress entries for N turns after activation.
* **Triggers** allow one entry's evaluation to activate others, with bounded re-evaluation passes.

### Assembly

* **Slots** describe functional depth in the prompt (`preamble`, `foundation`, `context`, `reference`, `framing`, `guidance`, `emphasis`, `immediate`, `aftermath`, plus `at_depth(N)`).
* **Fallback chains** so entries gracefully degrade when their primary slot is not present in the host template.
* **Token budgets** at the global level, with optional per-group budgets for entries that share a named pool.
* **Priority and insertion order** as the tie-breakers within a slot.

### Host context

* **Namespaces** with configurable access (`ReadOnly` for host-provided data like `char`, `user`, `chat`; `ReadWrite` for template-mutable state like `state` and `local`).
* **Persistent state** in the `state:` namespace survives across turns and is exposed for save/load.
* **Recursion and cycle detection** for the `[[entry_id]]` document-inlining mechanism.
* **DoS bounds** on template evaluation via `max_node_evaluations` and `max_iterations`.

### Standard library

Enabled by the `stdlib` feature (on by default):

* **Commands** that mutate state: `set_var`, `get_var`, `inc_var`, `push_var`, `default_var`, `is_active`.
* **`text.*`** processors: `upper`, `lower`, `length`, `trim`, `capitalize`, `contains`, `starts_with`, `ends_with`, `replace`, `substr`, `join`, `repeat`.
* **`math.*`** processors: `add`, `sub`, `mul`, `div`, `mod`, `abs`, `min`, `max`, `clamp`, `floor`, `ceil`, `round`.
* **`array.*`** processors: `length`, `contains`, `first`, `last`, `reverse`, `slice`, `range`, `concat`.

### Registry plugins

Host applications extend the template surface by implementing `Plugin` and registering processors or commands. Plugins get the same registry access as the built-in stdlib.

```rust
use context_weaver::Plugin;
use weaver_lang::{Registry, ClosureProcessor, Value};

struct DicePlugin;

impl Plugin for DicePlugin {
    fn name(&self) -> &str { "dice" }

    fn register(&self, registry: &mut Registry) {
        registry.register_processor(ClosureProcessor::new("dice", "roll", |props| {
            let sides = props.get("sides").and_then(|v| v.as_number()).unwrap_or(6.0) as u32;
            let result = rand::random::<u32>() % sides + 1;
            Ok(Value::Number(result as f64))
        }));
    }
}

weaver.register_plugin(DicePlugin);
```

Templates then call `@[dice.roll(sides: 20)]`.

For type-safe processors with automatic validation, the [`weaver-macros`](https://github.com/crux-03/weaver_lang) crate provides `#[weaver_processor]` and `#[weaver_command]` attributes.

### Lifecycle plugins

Lifecycle plugins are distinct from registry plugins. Where a `Plugin` adds new processors and commands that templates invoke, a `LifecyclePlugin` observes and mutates the engine's state as it moves through the pipeline. They are the right tool for PII redaction, analytics, forced inclusion, content post-processing, and save/load snapshotting.

The trait has seven hooks, all with no-op defaults. Implement only the ones you need:

```rust
use context_weaver::{LifecyclePlugin, PostAssembleCtx, HookError};

struct BlockCounter { count: usize }

impl LifecyclePlugin for BlockCounter {
    fn name(&self) -> &str { "block_counter" }

    fn post_assemble(&mut self, ctx: &mut PostAssembleCtx<'_>)
        -> Result<(), HookError>
    {
        self.count += ctx.blocks.len();
        Ok(())
    }
}

weaver.register_lifecycle(BlockCounter { count: 0 });
```

For one-off or stateless cases, `FnLifecycle` accepts closures via a builder:

```rust
use context_weaver::FnLifecycle;

weaver.register_lifecycle(
    FnLifecycle::new("redactor")
        .on_pre_activation(|ctx| {
            for msg in ctx.messages.iter_mut() {
                msg.content = msg.content.replace("secret_token", "[REDACTED]");
            }
            Ok(())
        })
        .on_post_assemble(|ctx| {
            eprintln!("assembled {} blocks", ctx.blocks.len());
            Ok(())
        })
);
```

Plugins fire in registration order across the set, and within a plugin hooks fire in pipeline order (`pre_activation` → `post_activation` → `pre_evaluate` → `post_evaluate` → `on_trigger_fired` → `post_assemble`). `on_turn_advance` fires independently from `advance_turn()`. Any hook returning `Err(HookError)` aborts the pipeline with a `ContextWeaverError::PluginHook`.

See the [`lifecycle`](src/lifecycle.rs) module documentation for the full list of context types and their mutable surfaces.

### State persistence

Both the activation state (sticky counters, cooldown timers, turn counter) and the persistent variable map are exposed for serialization:

```rust
let activation_snapshot = weaver.activation_state().clone();
let state_snapshot = weaver.persistent_state().clone();

// ...later...

weaver.restore_activation_state(activation_snapshot);
weaver.restore_persistent_state(state_snapshot);
```

Both types derive `Serialize` and `Deserialize`, so any serde-compatible format works (JSON, YAML, MessagePack, CBOR).

## Roadmap

### v0.1.0 (current focus)

**Serialization**

* `format_version: u32` field on `LorebookConfig`. Currently absent, which makes forward-compat impossible.
* `Entry::to_source() -> String` to round-trip an entry back to its `.weaver` representation.
* `Lorebook::to_bundle() / from_bundle()` for single-blob export, with PNG tEXt embedding and database storage as primary use cases.
* `Serialize`/`Deserialize` on `ChatMessage` and `ChatRole`.
* Round-trip test specifically for `Slot::AtDepth(N)` in both YAML and JSON.

**Author experience**

* Expose `ActivationReason` and the full activation trace alongside `assemble`'s return value so authors can debug "why didn't my entry fire?" without instrumenting the engine themselves.
* `LorebookBuilder` for programmatic construction.

### v0.2.0

* **Per-entry token cap** in addition to global and group budgets.
* **Embedding-based activation hook.** A trait the host can implement to plug in vector-similarity matching alongside keyword and regex.
* **SillyTavern `character_book.json` interop.** `From`/`Into` impls for the de facto interchange format.
* **Plugin conflict detection and load ordering.** Currently last-write-wins on processor name collisions, silently.
* **Async lifecycle hooks** for plugins that need to hit external APIs from within a phase.
* **Read-only registry access from lifecycle hook contexts**, for hooks that want to call processors over content.

## License

MIT
