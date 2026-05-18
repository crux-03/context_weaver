//! Standard library of commands and processors for ContextWeaver.
//!
//! Provides the core set of callables that lorebook authors expect. Enable
//! via the `stdlib` cargo feature (on by default).
//!
//! # Commands (mutate state via `EvalContext`)
//!
//! | Command | Syntax | Description |
//! |---------|--------|-------------|
//! | `set_var` | `$[set_var("scope:name", value)]` | Set a variable |
//! | `get_var` | `$[get_var("scope:name")]` | Read a variable (returns its value) |
//! | `inc_var` | `$[inc_var("scope:name", 1)]` | Increment a numeric variable |
//! | `push_var` | `$[push_var("scope:name", value)]` | Append to an array variable |
//! | `default_var` | `$[default_var("scope:name", value)]` | Set only if not already defined |
//!
//! # Processors (pure computation)
//!
//! ## `text.*`
//! - `upper`, `lower`, `length`, `trim`, `capitalize`
//! - `contains`, `starts_with`, `ends_with`
//! - `replace`, `substr`, `join`, `repeat`
//!
//! ## `math.*`
//! - `add`, `sub`, `mul`, `div`, `mod`
//! - `abs`, `min`, `max`, `clamp`
//! - `floor`, `ceil`, `round`
//!
//! ## `array.*`
//! - `length`, `contains`, `first`, `last`
//! - `reverse`, `slice`, `range`, `concat`

use weaver_lang::registry::{CommandSignature, ParamDef, WeaverCommand};
use weaver_lang::{ClosureProcessor, EvalContext, EvalError, EvalErrorKind, Registry, Value};

/// Register all standard library commands and processors.
pub fn register(registry: &mut Registry) {
    register_commands(registry);
    register_text_processors(registry);
    register_math_processors(registry);
    register_array_processors(registry);
}

// ═══════════════════════════════════════════════════════════════════════
// Commands
// ═══════════════════════════════════════════════════════════════════════

fn register_commands(registry: &mut Registry) {
    registry.register_command(SetVarCommand);
    registry.register_command(GetVarCommand);
    registry.register_command(IncVarCommand);
    registry.register_command(PushVarCommand);
    registry.register_command(DefaultVarCommand);
}

/// Parse a `"scope:name"` key string into (scope, name).
fn parse_var_key(key: &str) -> Result<(&str, &str), EvalError> {
    key.find(':')
        .map(|pos| (&key[..pos], &key[pos + 1..]))
        .ok_or_else(|| {
            EvalError::new(
                EvalErrorKind::HostError,
                format!("invalid variable key \"{key}\": expected \"scope:name\" format"),
            )
        })
}

// ── set_var ─────────────────────────────────────────────────────────────

/// `$[set_var("scope:name", value)]` — set a variable in any writable scope.
struct SetVarCommand;

impl WeaverCommand for SetVarCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let key = args.first().and_then(|v| v.as_string()).ok_or_else(|| {
            EvalError::type_error("string", args.first().map_or("none", |v| v.type_name()))
        })?;
        let value = args.get(1).cloned().unwrap_or(Value::None);
        let (scope, name) = parse_var_key(key)?;
        ctx.set_variable(scope, name, value)?;
        Ok(None)
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "set_var".to_string(),
            params: vec![
                ParamDef {
                    name: "key".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::String),
                    required: true,
                },
                ParamDef {
                    name: "value".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::Any),
                    required: true,
                },
            ],
        }
    }
}

// ── get_var ─────────────────────────────────────────────────────────────

/// `$[get_var("scope:name")]` — read a variable and return it.
struct GetVarCommand;

impl WeaverCommand for GetVarCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let key = args.first().and_then(|v| v.as_string()).ok_or_else(|| {
            EvalError::type_error("string", args.first().map_or("none", |v| v.type_name()))
        })?;
        let (scope, name) = parse_var_key(key)?;
        match ctx.resolve_variable(scope, name)? {
            Some(val) => Ok(Some(val)),
            None => Ok(Some(Value::None)),
        }
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "get_var".to_string(),
            params: vec![ParamDef {
                name: "key".to_string(),
                expected_type: Some(weaver_lang::registry::ValueType::String),
                required: true,
            }],
        }
    }
}

// ── inc_var ─────────────────────────────────────────────────────────────

/// `$[inc_var("scope:name", amount)]` — increment a numeric variable.
///
/// If the variable doesn't exist yet, it is initialized to `amount`.
/// If it exists but isn't a number, returns a type error.
struct IncVarCommand;

impl WeaverCommand for IncVarCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let key = args.first().and_then(|v| v.as_string()).ok_or_else(|| {
            EvalError::type_error("string", args.first().map_or("none", |v| v.type_name()))
        })?;
        let amount = args.get(1).and_then(|v| v.as_number()).unwrap_or(1.0);

        let (scope, name) = parse_var_key(key)?;

        let current = ctx.resolve_variable(scope, name)?;
        let new_val = match current {
            Some(Value::Number(n)) => n + amount,
            Some(other) => {
                return Err(EvalError::type_error("number", other.type_name()));
            }
            None => amount,
        };

        ctx.set_variable(scope, name, Value::Number(new_val))?;
        Ok(None)
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "inc_var".to_string(),
            params: vec![
                ParamDef {
                    name: "key".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::String),
                    required: true,
                },
                ParamDef {
                    name: "amount".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::Number),
                    required: false,
                },
            ],
        }
    }
}

// ── push_var ────────────────────────────────────────────────────────────

/// `$[push_var("scope:name", value)]` — append a value to an array variable.
///
/// If the variable doesn't exist yet, creates a new array containing the value.
/// If it exists but isn't an array, returns a type error.
struct PushVarCommand;

impl WeaverCommand for PushVarCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let key = args.first().and_then(|v| v.as_string()).ok_or_else(|| {
            EvalError::type_error("string", args.first().map_or("none", |v| v.type_name()))
        })?;
        let value = args.get(1).cloned().unwrap_or(Value::None);

        let (scope, name) = parse_var_key(key)?;

        let current = ctx.resolve_variable(scope, name)?;
        let new_arr = match current {
            Some(Value::Array(mut arr)) => {
                arr.push(value);
                arr
            }
            Some(other) => {
                return Err(EvalError::type_error("array", other.type_name()));
            }
            None => vec![value],
        };

        ctx.set_variable(scope, name, Value::Array(new_arr))?;
        Ok(None)
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "push_var".to_string(),
            params: vec![
                ParamDef {
                    name: "key".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::String),
                    required: true,
                },
                ParamDef {
                    name: "value".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::Any),
                    required: true,
                },
            ],
        }
    }
}

// ── default_var ─────────────────────────────────────────────────────────

/// `$[default_var("scope:name", value)]` — set only if not already defined.
///
/// If the variable already exists (even if `None`), this is a no-op.
/// Useful for initialization in constant entries.
struct DefaultVarCommand;

impl WeaverCommand for DefaultVarCommand {
    fn call(
        &self,
        args: Vec<Value>,
        ctx: &mut dyn EvalContext,
        _registry: &Registry,
    ) -> Result<Option<Value>, EvalError> {
        let key = args.first().and_then(|v| v.as_string()).ok_or_else(|| {
            EvalError::type_error("string", args.first().map_or("none", |v| v.type_name()))
        })?;
        let default = args.get(1).cloned().unwrap_or(Value::None);

        let (scope, name) = parse_var_key(key)?;

        match ctx.resolve_variable(scope, name)? {
            Some(_) => Ok(None), // already set, skip
            None => {
                ctx.set_variable(scope, name, default)?;
                Ok(None)
            }
        }
    }

    fn signature(&self) -> CommandSignature {
        CommandSignature {
            name: "default_var".to_string(),
            params: vec![
                ParamDef {
                    name: "key".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::String),
                    required: true,
                },
                ParamDef {
                    name: "value".to_string(),
                    expected_type: Some(weaver_lang::registry::ValueType::Any),
                    required: true,
                },
            ],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Processors
// ═══════════════════════════════════════════════════════════════════════

fn register_text_processors(registry: &mut Registry) {
    // @[text.upper(text: "hello")] → "HELLO"
    registry.register_processor(ClosureProcessor::new("text", "upper", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        Ok(Value::String(text.to_uppercase()))
    }));

    // @[text.lower(text: "HELLO")] → "hello"
    registry.register_processor(ClosureProcessor::new("text", "lower", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        Ok(Value::String(text.to_lowercase()))
    }));

    // @[text.length(text: "hello")] → 5
    registry.register_processor(ClosureProcessor::new("text", "length", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        Ok(Value::Number(text.len() as f64))
    }));

    // @[text.trim(text: "  hello  ")] → "hello"
    registry.register_processor(ClosureProcessor::new("text", "trim", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        Ok(Value::String(text.trim().to_string()))
    }));

    // @[text.capitalize(text: "hello world")] → "Hello world"
    registry.register_processor(ClosureProcessor::new("text", "capitalize", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        let mut chars = text.chars();
        let capitalized = match chars.next() {
            None => String::new(),
            Some(c) => c.to_uppercase().to_string() + chars.as_str(),
        };
        Ok(Value::String(capitalized))
    }));

    // @[text.contains(text: "hello world", substring: "world")] → true
    registry.register_processor(ClosureProcessor::new("text", "contains", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        let sub = props
            .get("substring")
            .and_then(|v| v.as_string())
            .unwrap_or("");
        Ok(Value::Bool(text.contains(sub)))
    }));

    // @[text.starts_with(text: "hello", prefix: "hel")] → true
    registry.register_processor(ClosureProcessor::new("text", "starts_with", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        let prefix = props
            .get("prefix")
            .and_then(|v| v.as_string())
            .unwrap_or("");
        Ok(Value::Bool(text.starts_with(prefix)))
    }));

    // @[text.ends_with(text: "hello", suffix: "llo")] → true
    registry.register_processor(ClosureProcessor::new("text", "ends_with", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        let suffix = props
            .get("suffix")
            .and_then(|v| v.as_string())
            .unwrap_or("");
        Ok(Value::Bool(text.ends_with(suffix)))
    }));

    // @[text.replace(text: "hello world", from: "world", to: "rust")]
    registry.register_processor(ClosureProcessor::new("text", "replace", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        let from = props.get("from").and_then(|v| v.as_string()).unwrap_or("");
        let to = props.get("to").and_then(|v| v.as_string()).unwrap_or("");
        Ok(Value::String(text.replace(from, to)))
    }));

    // @[text.substr(text: "hello", start: 1, length: 3)] → "ell"
    registry.register_processor(ClosureProcessor::new("text", "substr", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        let start = props
            .get("start")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0) as usize;
        let length = props.get("length").and_then(|v| v.as_number());
        let result: String = match length {
            Some(len) => text.chars().skip(start).take(len as usize).collect(),
            None => text.chars().skip(start).collect(),
        };
        Ok(Value::String(result))
    }));

    // @[text.join(items: ["a", "b", "c"], separator: ", ")] → "a, b, c"
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

    // @[text.repeat(text: "ha", count: 3)] → "hahaha"
    registry.register_processor(ClosureProcessor::new("text", "repeat", |props| {
        let text = props.get("text").and_then(|v| v.as_string()).unwrap_or("");
        let count = props
            .get("count")
            .and_then(|v| v.as_number())
            .unwrap_or(1.0) as usize;
        Ok(Value::String(text.repeat(count)))
    }));
}

fn register_math_processors(registry: &mut Registry) {
    // @[math.add(a: 1, b: 2)] → 3
    registry.register_processor(ClosureProcessor::new("math", "add", |props| {
        let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
        Ok(Value::Number(a + b))
    }));

    // @[math.sub(a: 5, b: 3)] → 2
    registry.register_processor(ClosureProcessor::new("math", "sub", |props| {
        let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
        Ok(Value::Number(a - b))
    }));

    // @[math.mul(a: 2, b: 3)] → 6
    registry.register_processor(ClosureProcessor::new("math", "mul", |props| {
        let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
        Ok(Value::Number(a * b))
    }));

    // @[math.div(a: 10, b: 3)] → 3.333...
    registry.register_processor(ClosureProcessor::new("math", "div", |props| {
        let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(1.0);
        if b == 0.0 {
            return Err(EvalError::new(
                EvalErrorKind::HostError,
                "division by zero".to_string(),
            ));
        }
        Ok(Value::Number(a / b))
    }));

    // @[math.mod(a: 10, b: 3)] → 1
    registry.register_processor(ClosureProcessor::new("math", "mod", |props| {
        let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(1.0);
        if b == 0.0 {
            return Err(EvalError::new(
                EvalErrorKind::HostError,
                "modulo by zero".to_string(),
            ));
        }
        Ok(Value::Number(a % b))
    }));

    // @[math.abs(value: -5)] → 5
    registry.register_processor(ClosureProcessor::new("math", "abs", |props| {
        let value = props
            .get("value")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0);
        Ok(Value::Number(value.abs()))
    }));

    // @[math.min(a: 3, b: 7)] → 3
    registry.register_processor(ClosureProcessor::new("math", "min", |props| {
        let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
        Ok(Value::Number(a.min(b)))
    }));

    // @[math.max(a: 3, b: 7)] → 7
    registry.register_processor(ClosureProcessor::new("math", "max", |props| {
        let a = props.get("a").and_then(|v| v.as_number()).unwrap_or(0.0);
        let b = props.get("b").and_then(|v| v.as_number()).unwrap_or(0.0);
        Ok(Value::Number(a.max(b)))
    }));

    // @[math.clamp(value: 150, min: 0, max: 100)] → 100
    registry.register_processor(ClosureProcessor::new("math", "clamp", |props| {
        let value = props
            .get("value")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0);
        let min = props.get("min").and_then(|v| v.as_number()).unwrap_or(0.0);
        let max = props.get("max").and_then(|v| v.as_number()).unwrap_or(1.0);
        Ok(Value::Number(value.max(min).min(max)))
    }));

    // @[math.floor(value: 3.7)] → 3
    registry.register_processor(ClosureProcessor::new("math", "floor", |props| {
        let value = props
            .get("value")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0);
        Ok(Value::Number(value.floor()))
    }));

    // @[math.ceil(value: 3.2)] → 4
    registry.register_processor(ClosureProcessor::new("math", "ceil", |props| {
        let value = props
            .get("value")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0);
        Ok(Value::Number(value.ceil()))
    }));

    // @[math.round(value: 3.5)] → 4
    registry.register_processor(ClosureProcessor::new("math", "round", |props| {
        let value = props
            .get("value")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0);
        Ok(Value::Number(value.round()))
    }));
}

fn register_array_processors(registry: &mut Registry) {
    // @[array.length(items: [1, 2, 3])] → 3
    registry.register_processor(ClosureProcessor::new("array", "length", |props| {
        let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
        Ok(Value::Number(items.len() as f64))
    }));

    // @[array.contains(items: ["a", "b"], value: "a")] → true
    registry.register_processor(ClosureProcessor::new("array", "contains", |props| {
        let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
        let value = props.get("value").cloned().unwrap_or(Value::None);
        Ok(Value::Bool(items.contains(&value)))
    }));

    // @[array.first(items: [1, 2, 3])] → 1
    registry.register_processor(ClosureProcessor::new("array", "first", |props| {
        let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
        Ok(items.first().cloned().unwrap_or(Value::None))
    }));

    // @[array.last(items: [1, 2, 3])] → 3
    registry.register_processor(ClosureProcessor::new("array", "last", |props| {
        let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
        Ok(items.last().cloned().unwrap_or(Value::None))
    }));

    // @[array.reverse(items: [1, 2, 3])] → [3, 2, 1]
    registry.register_processor(ClosureProcessor::new("array", "reverse", |props| {
        let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
        let mut reversed = items.to_vec();
        reversed.reverse();
        Ok(Value::Array(reversed))
    }));

    // @[array.slice(items: [1, 2, 3, 4], start: 1, end: 3)] → [2, 3]
    registry.register_processor(ClosureProcessor::new("array", "slice", |props| {
        let items = props.get("items").and_then(|v| v.as_array()).unwrap_or(&[]);
        let start = props
            .get("start")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0) as usize;
        let end = props
            .get("end")
            .and_then(|v| v.as_number())
            .map(|n| n as usize)
            .unwrap_or(items.len());
        let sliced = items
            .get(start..end.min(items.len()))
            .unwrap_or(&[])
            .to_vec();
        Ok(Value::Array(sliced))
    }));

    // @[array.range(start: 1, end: 4)] → [1, 2, 3]
    registry.register_processor(ClosureProcessor::new("array", "range", |props| {
        let start = props
            .get("start")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0) as i64;
        let end = props.get("end").and_then(|v| v.as_number()).unwrap_or(0.0) as i64;
        let range: Vec<Value> = (start..end).map(|n| Value::Number(n as f64)).collect();
        Ok(Value::Array(range))
    }));

    // @[array.concat(a: [1, 2], b: [3, 4])] → [1, 2, 3, 4]
    registry.register_processor(ClosureProcessor::new("array", "concat", |props| {
        let a = props.get("a").and_then(|v| v.as_array()).unwrap_or(&[]);
        let b = props.get("b").and_then(|v| v.as_array()).unwrap_or(&[]);
        let mut result = a.to_vec();
        result.extend(b.iter().cloned());
        Ok(Value::Array(result))
    }));
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use super::*;

    /// Simple EvalContext for testing commands in isolation.
    struct SimpleContext {
        variables: HashMap<String, HashMap<String, Value>>,
    }

    impl SimpleContext {
        fn new() -> Self {
            Self {
                variables: HashMap::new(),
            }
        }

        fn set(&mut self, scope: &str, name: &str, value: impl Into<Value>) {
            self.variables
                .entry(scope.to_string())
                .or_default()
                .insert(name.to_string(), value.into());
        }
    }

    impl EvalContext for SimpleContext {
        fn resolve_variable(&self, scope: &str, name: &str) -> Result<Option<Value>, EvalError> {
            Ok(self
                .variables
                .get(scope)
                .and_then(|ns| ns.get(name))
                .cloned())
        }

        fn set_variable(&mut self, scope: &str, name: &str, value: Value) -> Result<(), EvalError> {
            self.variables
                .entry(scope.to_string())
                .or_default()
                .insert(name.to_string(), value);
            Ok(())
        }

        fn fire_trigger(
            &mut self,
            _entry_id: &str,
            _registry: &Registry,
        ) -> Result<String, EvalError> {
            Ok(String::new())
        }

        fn resolve_document(
            &mut self,
            _document_id: &str,
            _registry: &Registry,
        ) -> Result<String, EvalError> {
            Ok(String::new())
        }
    }

    fn make_registry() -> Registry {
        let mut registry = Registry::new();
        register(&mut registry);
        registry
    }

    // ── Command tests ───────────────────────────────────────────────

    #[test]
    fn test_set_var() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();

        registry
            .call_command(
                "set_var",
                vec![
                    Value::String("global:name".into()),
                    Value::String("Kael".into()),
                ],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "name").unwrap();
        assert_eq!(val, Some(Value::String("Kael".into())));
    }

    #[test]
    fn test_get_var() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();
        ctx.set("global", "hp", 100i64);

        let result = registry
            .call_command("get_var", vec![Value::String("global:hp".into())], &mut ctx)
            .unwrap();

        assert_eq!(result, Some(Value::Number(100.0)));
    }

    #[test]
    fn test_get_var_missing_returns_none() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();

        let result = registry
            .call_command(
                "get_var",
                vec![Value::String("global:missing".into())],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result, Some(Value::None));
    }

    #[test]
    fn test_inc_var_initializes() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();

        registry
            .call_command(
                "inc_var",
                vec![Value::String("global:score".into()), Value::Number(5.0)],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "score").unwrap();
        assert_eq!(val, Some(Value::Number(5.0)));
    }

    #[test]
    fn test_inc_var_increments_existing() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();
        ctx.set("global", "score", 10i64);

        registry
            .call_command(
                "inc_var",
                vec![Value::String("global:score".into()), Value::Number(3.0)],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "score").unwrap();
        assert_eq!(val, Some(Value::Number(13.0)));
    }

    #[test]
    fn test_inc_var_default_increment() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();
        ctx.set("global", "count", 0i64);

        registry
            .call_command(
                "inc_var",
                vec![Value::String("global:count".into())],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "count").unwrap();
        assert_eq!(val, Some(Value::Number(1.0)));
    }

    #[test]
    fn test_push_var_creates_array() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();

        registry
            .call_command(
                "push_var",
                vec![
                    Value::String("global:log".into()),
                    Value::String("first".into()),
                ],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "log").unwrap();
        assert_eq!(val, Some(Value::Array(vec![Value::String("first".into())])));
    }

    #[test]
    fn test_push_var_appends() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();
        ctx.set(
            "global",
            "log",
            Value::Array(vec![Value::String("a".into())]),
        );

        registry
            .call_command(
                "push_var",
                vec![
                    Value::String("global:log".into()),
                    Value::String("b".into()),
                ],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "log").unwrap();
        assert_eq!(
            val,
            Some(Value::Array(vec![
                Value::String("a".into()),
                Value::String("b".into()),
            ]))
        );
    }

    #[test]
    fn test_default_var_sets_when_missing() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();

        registry
            .call_command(
                "default_var",
                vec![Value::String("global:x".into()), Value::Number(42.0)],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "x").unwrap();
        assert_eq!(val, Some(Value::Number(42.0)));
    }

    #[test]
    fn test_default_var_skips_when_present() {
        let registry = make_registry();
        let mut ctx = SimpleContext::new();
        ctx.set("global", "x", 10i64);

        registry
            .call_command(
                "default_var",
                vec![Value::String("global:x".into()), Value::Number(42.0)],
                &mut ctx,
            )
            .unwrap();

        let val = ctx.resolve_variable("global", "x").unwrap();
        assert_eq!(val, Some(Value::Number(10.0)));
    }

    // ── Text processor tests ────────────────────────────────────────

    #[test]
    fn test_text_processors() {
        let mut registry = Registry::new();
        register_text_processors(&mut registry);

        let upper = registry
            .call_processor(
                "text",
                "upper",
                props(&[("text", Value::String("hello".into()))]),
            )
            .unwrap();
        assert_eq!(upper, Value::String("HELLO".into()));

        let joined = registry
            .call_processor(
                "text",
                "join",
                props(&[
                    (
                        "items",
                        Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
                    ),
                    ("separator", Value::String(" - ".into())),
                ]),
            )
            .unwrap();
        assert_eq!(joined, Value::String("a - b".into()));

        let contains = registry
            .call_processor(
                "text",
                "contains",
                props(&[
                    ("text", Value::String("hello world".into())),
                    ("substring", Value::String("world".into())),
                ]),
            )
            .unwrap();
        assert_eq!(contains, Value::Bool(true));
    }

    // ── Math processor tests ────────────────────────────────────────

    #[test]
    fn test_math_processors() {
        let mut registry = Registry::new();
        register_math_processors(&mut registry);

        let clamp = registry
            .call_processor(
                "math",
                "clamp",
                props(&[
                    ("value", Value::Number(150.0)),
                    ("min", Value::Number(0.0)),
                    ("max", Value::Number(100.0)),
                ]),
            )
            .unwrap();
        assert_eq!(clamp, Value::Number(100.0));

        let floor = registry
            .call_processor("math", "floor", props(&[("value", Value::Number(3.7))]))
            .unwrap();
        assert_eq!(floor, Value::Number(3.0));
    }

    // ── Array processor tests ───────────────────────────────────────

    #[test]
    fn test_array_processors() {
        let mut registry = Registry::new();
        register_array_processors(&mut registry);

        let contains = registry
            .call_processor(
                "array",
                "contains",
                props(&[
                    (
                        "items",
                        Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
                    ),
                    ("value", Value::String("a".into())),
                ]),
            )
            .unwrap();
        assert_eq!(contains, Value::Bool(true));

        let range = registry
            .call_processor(
                "array",
                "range",
                props(&[("start", Value::Number(1.0)), ("end", Value::Number(4.0))]),
            )
            .unwrap();
        assert_eq!(
            range,
            Value::Array(vec![
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0),
            ])
        );
    }

    fn props(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }
}
