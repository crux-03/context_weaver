//! Plugin interface for extending ContextWeaver.
//!
//! Plugins can register custom processors and commands that become
//! available in all lorebook entries. This is how the host application
//! exposes domain-specific functionality to template authors.
//!
//! ## Implementing a plugin
//!
//! ```rust,ignore
//! use context_weaver::Plugin;
//! use weaver_lang::{Registry, ClosureProcessor, ClosureCommand, Value};
//!
//! struct DicePlugin;
//!
//! impl Plugin for DicePlugin {
//!     fn name(&self) -> &str { "dice" }
//!
//!     fn register(&self, registry: &mut Registry) {
//!         // @[dice.roll(sides: 20)]
//!         registry.register_processor(ClosureProcessor::new("dice", "roll", |props| {
//!             let sides = props.get("sides")
//!                 .and_then(|v| v.as_number())
//!                 .unwrap_or(6.0) as u32;
//!             let result = rand::random::<u32>() % sides + 1;
//!             Ok(Value::Number(result as f64))
//!         }));
//!     }
//! }
//! ```
//!
//! ## Using the weaver-macros crate
//!
//! For type-safe processors and commands with automatic validation,
//! use the `#[weaver_processor]` and `#[weaver_command]` macros:
//!
//! ```rust,ignore
//! use weaver_macros::weaver_processor;
//! use weaver_lang::{Value, EvalError};
//!
//! #[weaver_processor(namespace = "dice", name = "roll")]
//! fn roll(sides: f64) -> Result<Value, EvalError> {
//!     let result = rand::random::<u32>() % (sides as u32) + 1;
//!     Ok(Value::Number(result as f64))
//! }
//!
//! // Then in your plugin:
//! impl Plugin for DicePlugin {
//!     fn name(&self) -> &str { "dice" }
//!     fn register(&self, registry: &mut Registry) {
//!         registry.register_processor(RollProcessor);
//!     }
//! }
//! ```

use weaver_lang::Registry;

/// Trait for ContextWeaver plugins.
///
/// Plugins register processors and commands with the weaver-lang registry.
/// They are loaded before any entry evaluation begins.
pub trait Plugin: Send + Sync {
    /// A unique identifier for this plugin.
    fn name(&self) -> &str;

    /// Register processors and commands with the registry.
    fn register(&self, registry: &mut Registry);

    /// Optional: called once when the plugin is first loaded.
    /// Use this for initialization that needs to happen before
    /// any entries are evaluated.
    fn init(&self) -> Result<(), PluginError> {
        Ok(())
    }

    /// Optional: provide metadata about what this plugin offers.
    /// Useful for documentation generation and editor autocomplete.
    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            name: self.name().to_string(),
            version: "0.1.0".to_string(),
            description: String::new(),
            provides_processors: Vec::new(),
            provides_commands: Vec::new(),
        }
    }
}

/// Metadata about what a plugin provides.
///
/// Used for documentation, editor autocomplete, and validation.
#[derive(Debug, Clone)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub provides_processors: Vec<CallableDoc>,
    pub provides_commands: Vec<CallableDoc>,
}

/// Documentation for a processor or command provided by a plugin.
#[derive(Debug, Clone)]
pub struct CallableDoc {
    /// Full name (e.g. "dice.roll" for processors, "set_var" for commands).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Usage example in weaver-lang syntax.
    pub example: String,
}

#[derive(Debug)]
pub struct PluginError {
    pub plugin_name: String,
    pub message: String,
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "plugin '{}' error: {}", self.plugin_name, self.message)
    }
}

impl std::error::Error for PluginError {}

// ── Built-in plugin: core commands ──────────────────────────────────────
//
// The CorePlugin is only compiled when the `stdlib` feature is disabled.
// When `stdlib` is enabled, the standard library provides a superset of
// these commands and processors.

/// Minimal core plugin providing essential commands when the stdlib
/// feature is disabled. When stdlib is enabled, use `stdlib::register`
/// instead — it provides a superset of what CorePlugin offers.
#[cfg(not(feature = "stdlib"))]
pub struct CorePlugin;

#[cfg(not(feature = "stdlib"))]
impl Plugin for CorePlugin {
    fn name(&self) -> &str {
        "core"
    }

    fn register(&self, registry: &mut Registry) {
        use weaver_lang::{ClosureCommand, ClosureProcessor, Value};

        // ── Commands ────────────────────────────────────────────────

        // $[set_var("scope:name", value)]
        registry.register_command(ClosureCommand::new("set_var", |args| {
            let _key = args.first().and_then(|v| v.as_string()).unwrap_or("");
            let _val = args.get(1).cloned().unwrap_or(Value::None);
            Ok(None)
        }));

        // $[get_var("scope:name")]
        registry.register_command(ClosureCommand::new("get_var", |args| {
            let _key = args.first().and_then(|v| v.as_string()).unwrap_or("");
            Ok(Some(Value::None))
        }));

        // ── Processors ──────────────────────────────────────────────

        // @[text.join(items: [...], separator: ", ")]
        registry.register_processor(ClosureProcessor::new("text", "join", |props| {
            let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
            let sep = props
                .get("separator")
                .and_then(|v| v.as_string())
                .unwrap_or(", ");
            let joined: String = items
                .iter()
                .map(|v| v.to_output_string())
                .collect::<Vec<_>>()
                .join(sep);
            Ok(Value::String(joined))
        }));

        // @[text.upper(text: "...")]
        registry.register_processor(ClosureProcessor::new("text", "upper", |props| {
            let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
            Ok(Value::String(text.to_uppercase()))
        }));

        // @[text.lower(text: "...")]
        registry.register_processor(ClosureProcessor::new("text", "lower", |props| {
            let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
            Ok(Value::String(text.to_lowercase()))
        }));

        // @[text.contains(text: "...", substring: "...")]
        registry.register_processor(ClosureProcessor::new("text", "contains", |props| {
            let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
            let sub = props
                .get("substring")
                .and_then(|v| v.as_string())
                .unwrap_or("");
            Ok(Value::Bool(text.contains(sub)))
        }));

        // @[math.add(a: 1, b: 2)]
        registry.register_processor(ClosureProcessor::new("math", "add", |props| {
            let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
            let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
            Ok(Value::Number(a + b))
        }));

        // @[math.mul(a: 2, b: 3)]
        registry.register_processor(ClosureProcessor::new("math", "mul", |props| {
            let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
            let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
            Ok(Value::Number(a * b))
        }));

        // @[array.length(items: [...])]
        registry.register_processor(ClosureProcessor::new("array", "length", |props| {
            let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
            Ok(Value::Number(items.len() as f64))
        }));
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            name: "core".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Minimal built-in commands and processors (stdlib disabled)".to_string(),
            provides_processors: vec![
                CallableDoc {
                    name: "text.join".into(),
                    description: "Join array elements into a string".into(),
                    example: r#"@[text.join(items: ["a", "b", "c"], separator: ", ")]"#.into(),
                },
                CallableDoc {
                    name: "math.add".into(),
                    description: "Add two numbers".into(),
                    example: "@[math.add(a: 1, b: 2)]".into(),
                },
            ],
            provides_commands: vec![CallableDoc {
                name: "set_var".into(),
                description: "Set a variable in the given scope".into(),
                example: r#"$[set_var("state:visited", true)]"#.into(),
            }],
        }
    }
}
