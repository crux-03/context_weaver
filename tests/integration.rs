//! Integration test: multi-turn RPG session exercising the full ContextWeaver pipeline.
//!
//! This test simulates a complete role-playing session across multiple turns,
//! covering every major feature:
//!
//! - **Activation**: constant, keyword, regex, and condition-based
//! - **Templates**: variables, if/else, foreach, processors, commands
//! - **Documents**: `[[...]]` inline expansion with variable access
//! - **Triggers**: `<trigger>` activating further entries across passes
//! - **State mutation**: `$[set_var]`, `$[inc_var]`, `$[default_var]`, `$[push_var]`
//! - **Conditional activation**: conditions that depend on state set by earlier entries
//! - **Cooldown**: entries suppressed for N turns after firing
//! - **Sticky**: entries persisting for N turns after activation
//! - **Token budgets**: entries ordered and trimmed by the assembler
//! - **Slot resolution**: entries placed in functional-depth slots
//! - **Stdlib processors**: text.*, math.*, array.*
//!
//! The scenario: a character named Kael, a Ranger, explores a world,
//! visits a merchant (triggering a quest), fights monsters, levels up,
//! checks inventory, and tests cooldown/sticky mechanics.

use context_weaver::{
    AssembledBlock, ChatMessage, ContextWeaver, Entry, Lorebook, NamespaceAccess, Slot,
};
use weaver_lang::Value;

// ── Entry sources ──────────────────────────────────────────────────────
// Each entry is a complete .weaver file as a string constant.

/// Constant entry: always active, initializes default state, includes a
/// document reference. Exercises: constant activation, default_var,
/// document resolution, variable interpolation.
const WORLD_RULES: &str = r#"---
id: world_rules
name: World Rules
constant: true
priority: 200
slot: preamble
---
$[default_var("state:level", 1)]
$[default_var("state:gold", 0)]
$[default_var("state:xp", 0)]
[World: {{char:name}} the {{char:class}}, Level {{state:level}}]
[[lore_header]]"#;

/// Document-only entry: never activated by keywords, only inlined via
/// `[[lore_header]]`. Exercises: document resolution, char namespace.
const LORE_HEADER: &str = r#"---
id: lore_header
name: Lore Header
---
The realm of Eldara awaits. Your quest begins now, {{user:name}}."#;

/// Keyword-activated with a condition gate. Only fires when the user
/// mentions the forest AND state:location == "forest".
/// Exercises: keyword activation, condition expression, if/else,
/// text.upper processor, variable interpolation.
const FOREST_DESC: &str = r#"---
id: forest_desc
name: Dark Forest Description
keywords: ["dark forest", "shadowed path"]
condition: '{{state:location}} == "forest"'
priority: 150
slot: coda
---
{# if {{state:level}} > 5 #}
The ancient dark forest recognizes a seasoned adventurer.
{# else #}
The dark forest looms, its shadows deep and menacing.
{# endif #}
Threat level: @[text.upper(text: "high")]
$[set_var("state:visited_forest", true)]"#;

/// Regex-activated entry. Fires when messages match combat-related words.
/// Exercises: regex activation, foreach loop, math processors, array
/// variable access, conditional formatting.
const COMBAT_SYSTEM: &str = r#"---
id: combat_system
name: Combat System
regex: ['\b(attack|fight|strike|slash)\b']
priority: 180
slot: preamble
---
=== COMBAT ===
{# if {{state:weapon}} == "none" #}
You have no weapon equipped!
{# else #}
Wielding: {{state:weapon}}
Base damage: @[math.mul(a: {{state:level}}, b: 3)]
{# endif #}
{# foreach item in {{state:inventory}} #}
- {{item}}
{# endforeach #}"#;

/// Keyword-activated entry that mutates state and fires a trigger.
/// The state mutation happens DURING the trigger-resolution eval pass,
/// so the triggered entry's condition can see the updated state.
/// Exercises: set_var, trigger firing, state mutation before trigger
/// condition check.
const MERCHANT: &str = r#"---
id: merchant_encounter
name: Merchant Encounter
keywords: ["merchant", "shop", "trader"]
priority: 140
slot: backdrop
---
The merchant greets you warmly.
$[set_var("state:quest_started", true)]
$[set_var("state:quest_name", "The Goblin Menace")]
$[set_var("state:quest_reward", 50)]
"Browse my wares, or hear about a job?"
<trigger id="quest_log">"#;

/// Trigger-only entry with a condition that depends on state set by the
/// merchant entry during the same turn. Contains a document reference
/// for further inline expansion.
/// Exercises: trigger activation, condition on mutated state, document
/// resolution chain.
const QUEST_LOG: &str = r#"---
id: quest_log
name: Quest Log
condition: '{{state:quest_started}}'
priority: 130
slot: backdrop
---
--- QUEST ACTIVATED ---
[[quest_details]]"#;

/// Document-only entry inlined by quest_log. Reads state variables set
/// by the merchant. Exercises: document chaining (merchant → quest_log
/// → quest_details), state variable access across document boundaries.
const QUEST_DETAILS: &str = r#"---
id: quest_details
name: Quest Details
---
Quest: {{state:quest_name}}
Reward: {{state:quest_reward}} gold"#;

/// Keyword-activated entry that modifies numeric state.
/// Exercises: inc_var, math processors (mul, add, clamp), conditional
/// content based on computed values.
const LEVEL_UP: &str = r#"---
id: level_up
name: Level Up
keywords: ["level up", "leveled up"]
priority: 160
slot: backdrop
---
$[inc_var("state:level", 1)]
$[inc_var("state:gold", 25)]
$[inc_var("state:xp", 100)]
LEVEL UP! You are now level {{state:level}}.
Power rating: @[math.mul(a: {{state:level}}, b: 10)]
Gold: {{state:gold}}
{# if {{state:level}} >= 3 #}
You've unlocked advanced abilities!
{# endif #}"#;

/// Keyword-activated entry using foreach, text, and array processors.
/// Exercises: foreach over state array, text.join, array.length,
/// text.upper processor.
const INVENTORY: &str = r#"---
id: inventory_entry
name: Inventory
keywords: ["inventory", "items", "backpack"]
priority: 120
slot: backdrop
---
=== INVENTORY (@[array.length(items: {{state:inventory}})] items) ===
{# foreach item in {{state:inventory}} #}
* @[text.upper(text: {{item}})]
{# endforeach #}
All items: @[text.join(items: {{state:inventory}}, separator: " | ")]"#;

/// Cooldown test: activates on keyword but has a 2-turn cooldown.
/// Exercises: cooldown mechanics.
const COOLDOWN_ENTRY: &str = r#"---
id: cooldown_entry
name: Special Move
keywords: ["special move"]
cooldown: 2
priority: 110
slot: coda
---
You execute a devastating special move!"#;

/// Sticky test: activates on keyword and persists for 2 additional turns.
/// Exercises: sticky_turns mechanics.
const STICKY_ENTRY: &str = r#"---
id: sticky_entry
name: Sacred Oath
keywords: ["sacred oath"]
sticky_turns: 2
priority: 105
slot: preamble
---
[Oath active] Your sacred oath empowers you."#;

/// Condition-only entry that activates when level >= 3 AND visited_forest.
/// Has no keywords — it's constant but with a compound condition.
/// Exercises: constant + condition, compound boolean expression.
const VETERAN_BONUS: &str = r#"---
id: veteran_bonus
name: Veteran Bonus
constant: true
condition: '({{state:level}} >= 3) && ({{state:visited_forest}})'
priority: 90
slot: coda
---
[Veteran bonus: +10% damage in familiar territory]"#;

/// Entry that uses push_var to build up a log array across turns.
/// Exercises: push_var, array building over time.
const EVENT_LOG: &str = r#"---
id: event_log
name: Event Logger
constant: true
priority: 50
slot: setting
---
$[push_var("state:event_log", "turn")]"#;

// ── Helpers ────────────────────────────────────────────────────────────

fn build_lorebook() -> Lorebook {
    let mut book = Lorebook::new();

    let sources = [
        WORLD_RULES,
        LORE_HEADER,
        FOREST_DESC,
        COMBAT_SYSTEM,
        MERCHANT,
        QUEST_LOG,
        QUEST_DETAILS,
        LEVEL_UP,
        INVENTORY,
        COOLDOWN_ENTRY,
        STICKY_ENTRY,
        VETERAN_BONUS,
        EVENT_LOG,
    ];

    for source in sources {
        let entry = Entry::parse(source, None).unwrap_or_else(|e| {
            panic!(
                "Failed to parse entry:\n{}\nError: {}",
                source.lines().take(5).collect::<Vec<_>>().join("\n"),
                e
            )
        });
        book.add_entry(entry);
    }

    book
}

fn build_engine() -> ContextWeaver {
    let book = build_lorebook();
    let mut engine = ContextWeaver::new(book);

    engine.reserve_namespace("char", NamespaceAccess::ReadOnly);
    engine.reserve_namespace("user", NamespaceAccess::ReadOnly);

    // Set up the character and user (read-only namespaces)
    engine.set_variable("char", "name", "Kael");
    engine.set_variable("char", "class", "Ranger");
    engine.set_variable("user", "name", "Alex");

    // Initialize some state
    engine.set_variable("state", "location", "town");
    engine.set_variable("state", "weapon", "none");
    engine.set_variable(
        "state",
        "inventory",
        Value::Array(vec![
            Value::String("rope".into()),
            Value::String("torch".into()),
            Value::String("rations".into()),
        ]),
    );

    engine
}

/// Find a block by entry_id in the assembled output.
fn find_block<'a>(blocks: &'a [AssembledBlock], entry_id: &str) -> Option<&'a AssembledBlock> {
    blocks.iter().find(|b| b.entry_id == entry_id)
}

/// Check that a specific entry is present and its content contains a substring.
fn assert_block_contains(blocks: &[AssembledBlock], entry_id: &str, substring: &str) {
    let block = find_block(blocks, entry_id)
        .unwrap_or_else(|| panic!("expected entry '{}' to be active", entry_id));
    assert!(
        block.content.contains(substring),
        "entry '{}' content does not contain '{}'\nactual content:\n{}",
        entry_id,
        substring,
        block.content
    );
}

/// Check that an entry is NOT in the assembled output.
fn assert_block_absent(blocks: &[AssembledBlock], entry_id: &str) {
    assert!(
        find_block(blocks, entry_id).is_none(),
        "expected entry '{}' to NOT be active, but it was.\ncontent: {}",
        entry_id,
        find_block(blocks, entry_id).map_or("", |b| &b.content),
    );
}

/// Collect all active entry IDs for debugging.
fn active_ids(blocks: &[AssembledBlock]) -> Vec<&str> {
    blocks.iter().map(|b| b.entry_id.as_str()).collect()
}

// ── The test ───────────────────────────────────────────────────────────

#[test]
#[cfg_attr(not(feature = "stdlib"), ignore)]
fn test_multi_turn_rpg_session() {
    let mut engine = build_engine();

    // ═══════════════════════════════════════════════════════════════════
    // TURN 1: Arrive at the dark forest
    // ═══════════════════════════════════════════════════════════════════
    //
    // The user mentions "dark forest". forest_desc has that keyword,
    // but its condition requires state:location == "forest". We set it
    // before assembling.
    //
    // Expected active:
    //   - world_rules (constant) — includes [[lore_header]]
    //   - forest_desc (keyword + condition)
    //   - event_log (constant)
    //   - NOT veteran_bonus (level < 3, visited_forest not set)

    engine.set_variable("state", "location", "forest");

    let messages = vec![ChatMessage::user("I walk into the dark forest")];

    let blocks = engine.assemble(&messages).expect("turn 1 failed");
    println!("=== TURN 1 ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    // world_rules should be active with document resolution
    assert_block_contains(&blocks, "world_rules", "Kael the Ranger");
    assert_block_contains(&blocks, "world_rules", "Level 1");
    // Document [[lore_header]] should have been inlined
    assert_block_contains(&blocks, "world_rules", "The realm of Eldara");
    assert_block_contains(&blocks, "world_rules", "Alex");

    // forest_desc should be active (keyword "dark forest" + condition met)
    assert_block_contains(&blocks, "forest_desc", "dark forest looms");
    // Level is 1, so the "else" branch fires
    assert_block_contains(&blocks, "forest_desc", "Threat level: HIGH");

    // veteran_bonus should NOT be active (level 1 < 3)
    assert_block_absent(&blocks, "veteran_bonus");

    // Verify slot assignments
    let world_block = find_block(&blocks, "world_rules").unwrap();
    assert_eq!(world_block.slot, Slot::Preamble);
    let forest_block = find_block(&blocks, "forest_desc").unwrap();
    assert_eq!(forest_block.slot, Slot::Coda);

    // state:visited_forest should have been set by forest_desc
    let persistent = engine.export_persistent();
    let state = persistent.get("state").unwrap();
    assert_eq!(
        state.get("visited_forest"),
        Some(&Value::Bool(true)),
        "forest_desc should have set state:visited_forest"
    );

    // ═══════════════════════════════════════════════════════════════════
    // TURN 2: Visit the merchant (triggers quest)
    // ═══════════════════════════════════════════════════════════════════
    //
    // Merchant entry fires, sets quest state, triggers quest_log.
    // quest_log's condition (state:quest_started) should pass because
    // the merchant set it during the trigger-resolution eval pass.
    // quest_log includes [[quest_details]] which reads the quest vars.
    //
    // Expected active:
    //   - world_rules (constant)
    //   - merchant_encounter (keyword "merchant")
    //   - quest_log (triggered by merchant, condition passes)
    //   - event_log (constant)

    engine.advance_turn().unwrap();
    engine.set_variable("state", "location", "town");

    let messages = vec![
        ChatMessage::user("I walk into the dark forest"),
        ChatMessage::user("I head back to town and visit the merchant"),
    ];

    let blocks = engine.assemble(&messages).expect("turn 2 failed");
    println!("\n=== TURN 2 ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    assert_block_contains(&blocks, "merchant_encounter", "merchant greets you");
    assert_block_contains(&blocks, "merchant_encounter", "Browse my wares");

    // The trigger chain: merchant → quest_log → [[quest_details]]
    assert_block_contains(&blocks, "quest_log", "QUEST ACTIVATED");
    assert_block_contains(&blocks, "quest_log", "The Goblin Menace");
    assert_block_contains(&blocks, "quest_log", "50 gold");

    // forest_desc should NOT fire (location is now "town")
    assert_block_absent(&blocks, "forest_desc");

    // Quest state should be persisted
    let persistent = engine.export_persistent();
    let state = persistent.get("state").unwrap();
    assert_eq!(state.get("quest_started"), Some(&Value::Bool(true)));
    assert_eq!(
        state.get("quest_name"),
        Some(&Value::String("The Goblin Menace".into()))
    );

    // ═══════════════════════════════════════════════════════════════════
    // TURN 3: Combat encounter
    // ═══════════════════════════════════════════════════════════════════
    //
    // User says "attack" → combat_system triggers via regex.
    // We equip a weapon first to test the if/else branch.
    //
    // Expected active:
    //   - world_rules (constant)
    //   - combat_system (regex match on "attack")
    //   - event_log (constant)

    engine.advance_turn().unwrap();
    engine.set_variable("state", "weapon", "longbow");

    let messages = vec![
        ChatMessage::user("I head back to town and visit the merchant"),
        ChatMessage::user("A goblin appears! I attack with my bow!"),
    ];

    let blocks = engine.assemble(&messages).expect("turn 3 failed");
    println!("\n=== TURN 3 ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    assert_block_contains(&blocks, "combat_system", "COMBAT");
    assert_block_contains(&blocks, "combat_system", "Wielding: longbow");
    // Level is 1, base damage = 1 * 3 = 3
    assert_block_contains(&blocks, "combat_system", "Base damage: 3");
    // Inventory foreach
    assert_block_contains(&blocks, "combat_system", "- rope");
    assert_block_contains(&blocks, "combat_system", "- torch");
    assert_block_contains(&blocks, "combat_system", "- rations");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 4: Level up (twice!) to reach level 3
    // ═══════════════════════════════════════════════════════════════════
    //
    // Expected active:
    //   - world_rules (constant) — now shows Level 2 (after first level up)
    //     Actually, default_var won't overwrite existing values, so level
    //     stays as set by level_up
    //   - level_up (keyword "level up")
    //   - event_log (constant)

    engine.advance_turn().unwrap();

    let messages = vec![
        ChatMessage::user("A goblin appears! I attack with my bow!"),
        ChatMessage::user("Victory! I level up!"),
    ];

    let blocks = engine.assemble(&messages).expect("turn 4 failed");
    println!("\n=== TURN 4 ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    assert_block_contains(&blocks, "level_up", "LEVEL UP!");
    // Level was 1, inc_var adds 1 → level is now 2
    assert_block_contains(&blocks, "level_up", "level 2");
    // Power rating: 2 * 10 = 20
    assert_block_contains(&blocks, "level_up", "Power rating: 20");
    // Gold: was 0, +25 = 25
    assert_block_contains(&blocks, "level_up", "Gold: 25");
    // Level 2 < 3, so no "advanced abilities" message
    assert!(
        !find_block(&blocks, "level_up")
            .unwrap()
            .content
            .contains("advanced abilities"),
        "should not have advanced abilities at level 2"
    );

    // Level up a second time to reach level 3
    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("Training pays off. I level up again!")];

    let blocks = engine.assemble(&messages).expect("turn 4b failed");

    println!("\n=== TURN 4b ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    assert_block_contains(&blocks, "level_up", "level 3");
    // Power rating: 3 * 10 = 30
    assert_block_contains(&blocks, "level_up", "Power rating: 30");
    // Gold: 25 + 25 = 50
    assert_block_contains(&blocks, "level_up", "Gold: 50");
    // NOW level >= 3, so we should see the advanced abilities message
    assert_block_contains(&blocks, "level_up", "advanced abilities");

    // world_rules should reflect the new level
    let persistent = engine.export_persistent();
    let state = persistent.get("state").unwrap();
    assert_eq!(Some(&Value::Number(3.0)), state.get("level"));

    // ═══════════════════════════════════════════════════════════════════
    // TURN 5: Check inventory (uses text and array processors)
    // ═══════════════════════════════════════════════════════════════════
    //
    // Expected active:
    //   - world_rules (constant)
    //   - inventory_entry (keyword "inventory")
    //   - veteran_bonus (constant, condition: level >= 3 AND visited_forest)
    //   - event_log (constant)

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("Let me check my inventory")];

    let blocks = engine.assemble(&messages).expect("turn 5 failed");
    println!("\n=== TURN 5 ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    assert_block_contains(&blocks, "inventory_entry", "INVENTORY");
    // array.length should show 3
    assert_block_contains(&blocks, "inventory_entry", "3 items");
    // text.upper on each item
    assert_block_contains(&blocks, "inventory_entry", "ROPE");
    assert_block_contains(&blocks, "inventory_entry", "TORCH");
    assert_block_contains(&blocks, "inventory_entry", "RATIONS");
    // text.join should produce the joined string
    assert_block_contains(&blocks, "inventory_entry", "rope | torch | rations");

    // veteran_bonus should NOW be active (level 3 >= 3 AND visited_forest == true)
    assert_block_contains(&blocks, "veteran_bonus", "Veteran bonus");
    assert_block_contains(&blocks, "veteran_bonus", "+10% damage");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 6: Cooldown test — first activation
    // ═══════════════════════════════════════════════════════════════════

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("I perform my special move!")];

    let blocks = engine.assemble(&messages).expect("turn 6 failed");
    println!("\n=== TURN 6 ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    assert_block_contains(&blocks, "cooldown_entry", "devastating special move");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 7: Cooldown test — should be suppressed (1 turn since fire)
    // ═══════════════════════════════════════════════════════════════════

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("I try my special move again!")];

    let blocks = engine.assemble(&messages).expect("turn 7 failed");
    println!("\n=== TURN 7 ===");
    println!("Active: {:?}", active_ids(&blocks));

    // cooldown_entry should NOT be active (cooldown: 2, only 1 turn passed)
    assert_block_absent(&blocks, "cooldown_entry");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 8: Cooldown expired — should fire again
    // ═══════════════════════════════════════════════════════════════════
    // cooldown means: is_on_cooldown returns true when current_turn - last < cooldown
    // If fired at turn T, on cooldown at T+1 (diff=1 < 2) and T+2-1? Let me check...
    // actually: record_activation saves current_turn. is_on_cooldown checks
    // current_turn - last_activated < cooldown. So if cooldown=2:
    //   - Fired at turn N: last_activated = N
    //   - Turn N+1: N+1 - N = 1 < 2 → on cooldown
    //   - Turn N+2: N+2 - N = 2, not < 2 → OFF cooldown
    // So turn 8 (one more advance) should be available.

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("special move, now!")];

    let blocks = engine.assemble(&messages).expect("turn 8 failed");
    println!("\n=== TURN 8 ===");
    println!("Active: {:?}", active_ids(&blocks));

    // Cooldown expired — should fire again
    assert_block_contains(&blocks, "cooldown_entry", "devastating special move");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 9: Sticky test — first activation
    // ═══════════════════════════════════════════════════════════════════

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user(
        "I swear a sacred oath to protect the realm!",
    )];

    let blocks = engine.assemble(&messages).expect("turn 9 failed");
    println!("\n=== TURN 9 ===");
    println!("Active: {:?}", active_ids(&blocks));

    assert_block_contains(&blocks, "sticky_entry", "Oath active");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 10: Sticky persists (turn 1 of 2)
    // ═══════════════════════════════════════════════════════════════════

    engine.advance_turn().unwrap();

    // No keywords that match sticky_entry — but it should still be active
    let messages = vec![ChatMessage::user("The weather is pleasant today.")];

    let blocks = engine.assemble(&messages).expect("turn 10 failed");
    println!("\n=== TURN 10 ===");
    println!("Active: {:?}", active_ids(&blocks));

    assert_block_contains(&blocks, "sticky_entry", "Oath active");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 11: Sticky persists (turn 2 of 2)
    // ═══════════════════════════════════════════════════════════════════

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("Just walking around the market.")];

    let blocks = engine.assemble(&messages).expect("turn 11 failed");
    println!("\n=== TURN 11 ===");
    println!("Active: {:?}", active_ids(&blocks));

    assert_block_contains(&blocks, "sticky_entry", "Oath active");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 12: Sticky expired
    // ═══════════════════════════════════════════════════════════════════

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("Still walking around.")];

    let blocks = engine.assemble(&messages).expect("turn 12 failed");
    println!("\n=== TURN 12 ===");
    println!("Active: {:?}", active_ids(&blocks));

    assert_block_absent(&blocks, "sticky_entry");

    // ═══════════════════════════════════════════════════════════════════
    // TURN 13: Combat at higher level — verify state accumulated
    // ═══════════════════════════════════════════════════════════════════
    //
    // After two level-ups, level is 3. Combat damage should be 3 * 3 = 9.
    // veteran_bonus should still be active.

    engine.advance_turn().unwrap();

    let messages = vec![ChatMessage::user("An orc attacks! I fight back!")];

    let blocks = engine.assemble(&messages).expect("turn 13 failed");
    println!("\n=== TURN 13 ===");
    for b in &blocks {
        println!("[{}] {}: {}", b.slot, b.entry_id, b.content.trim());
    }

    assert_block_contains(&blocks, "combat_system", "Wielding: longbow");
    // Level 3, damage = 3 * 3 = 9
    assert_block_contains(&blocks, "combat_system", "Base damage: 9");
    // veteran_bonus should be active (level >= 3 AND visited_forest)
    assert_block_contains(&blocks, "veteran_bonus", "Veteran bonus");

    // ═══════════════════════════════════════════════════════════════════
    // FINAL: Verify accumulated persistent state
    // ═══════════════════════════════════════════════════════════════════

    let persistent = engine.export_persistent();
    let state = persistent.get("state").unwrap();

    assert_eq!(state.get("level"), Some(&Value::Number(3.0)));
    assert_eq!(state.get("gold"), Some(&Value::Number(50.0)));
    assert_eq!(state.get("xp"), Some(&Value::Number(200.0)));
    assert_eq!(state.get("quest_started"), Some(&Value::Bool(true)));
    assert_eq!(
        state.get("quest_name"),
        Some(&Value::String("The Goblin Menace".into()))
    );
    assert_eq!(state.get("visited_forest"), Some(&Value::Bool(true)));
    assert_eq!(state.get("weapon"), Some(&Value::String("longbow".into())));

    // The event_log should have accumulated entries from push_var
    // (one "turn" string per turn where event_log was active)
    match state.get("event_log") {
        Some(Value::Array(log)) => {
            assert!(
                log.len() >= 5,
                "event log should have accumulated multiple entries, got {}",
                log.len()
            );
            // Every element should be "turn"
            for entry in log {
                assert_eq!(entry, &Value::String("turn".into()));
            }
        }
        other => panic!("expected event_log to be an array, got {:?}", other),
    }

    println!("\n=== ALL TURNS PASSED ===");
    println!("Final state: {:#?}", state);
}

// ═══════════════════════════════════════════════════════════════════════
// Focused tests for specific features
// ═══════════════════════════════════════════════════════════════════════

/// Verify that document resolution works across multiple levels of nesting
/// and that variables set by commands in one entry are visible in documents.
#[test]
#[cfg_attr(not(feature = "stdlib"), ignore)]
fn test_document_chain_with_state_mutation() {
    let mut book = Lorebook::new();

    // Entry A is constant, sets a var, then includes [[doc_b]]
    let entry_a = Entry::parse(
        r#"---
id: entry_a
constant: true
---
$[set_var("state:greeting", "Hello")]
From A: [[doc_b]]"#,
        None,
    )
    .unwrap();

    // doc_b reads the var set by entry_a and includes [[doc_c]]
    let entry_b = Entry::parse(
        r#"---
id: doc_b
---
{{state:greeting}} from B! [[doc_c]]"#,
        None,
    )
    .unwrap();

    // doc_c is the leaf
    let entry_c = Entry::parse(
        r#"---
id: doc_c
---
End."#,
        None,
    )
    .unwrap();

    book.add_entry(entry_a);
    book.add_entry(entry_b);
    book.add_entry(entry_c);

    let mut engine = ContextWeaver::new(book);
    let blocks = engine.assemble(&[]).unwrap();

    let block = find_block(&blocks, "entry_a").expect("entry_a should be active");
    assert!(
        block.content.contains("Hello from B!"),
        "doc_b should see state:greeting set by entry_a. Got: {}",
        block.content
    );
    assert!(
        block.content.contains("End."),
        "doc_c should be inlined through doc_b. Got: {}",
        block.content
    );
}

/// Verify that a condition referencing state set in the SAME turn by
/// another entry works correctly through the trigger mechanism.
#[test]
#[cfg_attr(not(feature = "stdlib"), ignore)]
fn test_trigger_with_state_dependent_condition() {
    let mut book = Lorebook::new();

    // Entry that sets state and fires a trigger
    let setter = Entry::parse(
        r#"---
id: the_setter
keywords: ["go"]
priority: 200
---
$[set_var("state:flag", true)]
Setter ran.
<trigger id="the_gated">"#,
        None,
    )
    .unwrap();

    // Gated entry: only activates if state:flag is truthy
    let gated = Entry::parse(
        r#"---
id: the_gated
condition: '{{state:flag}}'
---
Gate opened!"#,
        None,
    )
    .unwrap();

    book.add_entry(setter);
    book.add_entry(gated);

    let mut engine = ContextWeaver::new(book);
    let messages = vec![ChatMessage::user("let's go")];
    let blocks = engine.assemble(&messages).unwrap();

    assert_block_contains(&blocks, "the_setter", "Setter ran");
    assert_block_contains(&blocks, "the_gated", "Gate opened");
}

/// Verify that a condition prevents activation even when keywords match.
#[test]
fn test_condition_blocks_keyword_activation() {
    let mut book = Lorebook::new();

    let entry = Entry::parse(
        r#"---
id: gated
keywords: ["magic"]
condition: '{{state:mana}} > 0'
---
You cast a spell!"#,
        None,
    )
    .unwrap();

    book.add_entry(entry);

    let mut engine = ContextWeaver::new(book);
    // mana not set → condition evaluates to error → false
    let messages = vec![ChatMessage::user("I use magic!")];
    let blocks = engine.assemble(&messages).unwrap();

    assert_block_absent(&blocks, "gated");

    // Now set mana > 0 and try again
    engine.advance_turn().unwrap();
    engine.set_variable("state", "mana", 10i64);

    let blocks = engine.assemble(&messages).unwrap();
    assert_block_contains(&blocks, "gated", "cast a spell");
}

/// Verify that regex patterns work for activation.
#[test]
fn test_regex_activation() {
    let mut book = Lorebook::new();

    let entry = Entry::parse(
        r#"---
id: regex_test
regex: ['\d{3,}']
---
Large number detected!"#,
        None,
    )
    .unwrap();

    book.add_entry(entry);

    let mut engine = ContextWeaver::new(book);

    // No match
    let messages = vec![ChatMessage::user("just 42 gold")];
    let blocks = engine.assemble(&messages).unwrap();
    assert_block_absent(&blocks, "regex_test");

    // Match: 3+ digit number
    engine.advance_turn().unwrap();
    let messages = vec![ChatMessage::user("I found 1000 gold!")];
    let blocks = engine.assemble(&messages).unwrap();
    assert_block_contains(&blocks, "regex_test", "Large number detected");
}

/// Verify that disabled entries are completely skipped.
#[test]
fn test_disabled_entry_skipped() {
    let mut book = Lorebook::new();

    let entry = Entry::parse(
        r#"---
id: disabled_one
keywords: ["hello"]
enabled: false
---
You should never see this."#,
        None,
    )
    .unwrap();

    book.add_entry(entry);

    let mut engine = ContextWeaver::new(book);
    let messages = vec![ChatMessage::user("hello there!")];
    let blocks = engine.assemble(&messages).unwrap();
    assert_block_absent(&blocks, "disabled_one");
}

/// Verify that state persists across advance_turn calls.
#[test]
#[cfg_attr(not(feature = "stdlib"), ignore)]
fn test_state_persistence_across_turns() {
    let mut book = Lorebook::new();

    // Turn 1: sets a state variable
    let setter = Entry::parse(
        r#"---
id: state_setter
keywords: ["set"]
---
$[set_var("state:persistent_val", 42)]
Set."#,
        None,
    )
    .unwrap();

    // Turn 2: reads the state variable
    let reader = Entry::parse(
        r#"---
id: state_reader
keywords: ["read"]
---
Value is {{state:persistent_val}}."#,
        None,
    )
    .unwrap();

    book.add_entry(setter);
    book.add_entry(reader);

    let mut engine = ContextWeaver::new(book);

    // Turn 1: set the value
    let blocks = engine
        .assemble(&[ChatMessage::user("please set the value")])
        .unwrap();
    assert_block_contains(&blocks, "state_setter", "Set");

    // Turn 2: read the value
    engine.advance_turn().unwrap();
    let blocks = engine
        .assemble(&[ChatMessage::user("now read it back")])
        .unwrap();
    assert_block_contains(&blocks, "state_reader", "Value is 42");
}

/// Verify that entries are ordered by slot, then priority.
#[test]
fn test_assembly_ordering() {
    let mut book = Lorebook::new();

    let high = Entry::parse(
        r#"---
id: high_prio
constant: true
priority: 200
slot: backdrop
---
HIGH"#,
        None,
    )
    .unwrap();

    let low = Entry::parse(
        r#"---
id: low_prio
constant: true
priority: 50
slot: backdrop
---
LOW"#,
        None,
    )
    .unwrap();

    let early_slot = Entry::parse(
        r#"---
id: early_slot
constant: true
priority: 100
slot: preamble
---
FIRST"#,
        None,
    )
    .unwrap();

    book.add_entry(high);
    book.add_entry(low);
    book.add_entry(early_slot);

    let mut engine = ContextWeaver::new(book);
    let blocks = engine.assemble(&[]).unwrap();

    // preamble should come before context
    let positions: Vec<_> = blocks.iter().map(|b| b.entry_id.as_str()).collect();
    let early_idx = positions.iter().position(|&id| id == "early_slot");
    let high_idx = positions.iter().position(|&id| id == "high_prio");

    assert!(
        early_idx < high_idx,
        "preamble should precede backdrop in output. Order: {:?}",
        positions
    );

    // Within context, high_prio should come before low_prio
    let low_idx = positions.iter().position(|&id| id == "low_prio");
    assert!(
        high_idx < low_idx,
        "higher priority should come first within same slot. Order: {:?}",
        positions
    );
}

/// Verify that slot resolution with fallback works correctly.
#[test]
fn test_slot_fallback_resolution() {
    let mut book = Lorebook::new();

    // Entry targeting 'reference' with fallback to 'context'
    let entry = Entry::parse(
        r#"---
id: fallback_entry
constant: true
slot: coda
fallback: [backdrop]
---
Fallback content"#,
        None,
    )
    .unwrap();

    book.add_entry(entry);

    let mut engine = ContextWeaver::new(book);

    // Only make preamble and context available (no coda)
    engine.set_available_slots(vec![Slot::Preamble, Slot::Backdrop, Slot::Setting]);

    let blocks = engine.assemble(&[]).unwrap();

    let block = find_block(&blocks, "fallback_entry").expect("entry should be active via fallback");
    assert_eq!(
        block.slot,
        Slot::Backdrop,
        "entry should have fallen back to context slot"
    );
}

/// Verify that entries targeting unavailable slots with no fallback are dropped.
#[test]
fn test_unavailable_slot_drops_entry() {
    let mut book = Lorebook::new();

    let entry = Entry::parse(
        r#"---
id: no_slot_entry
constant: true
slot: setting
---
This should be dropped"#,
        None,
    )
    .unwrap();

    book.add_entry(entry);

    let mut engine = ContextWeaver::new(book);

    // Only make preamble available — emphasis is not included
    engine.set_available_slots(vec![Slot::Preamble]);

    let blocks = engine.assemble(&[]).unwrap();

    assert_block_absent(&blocks, "no_slot_entry");
}

/// Verify that AtDepth entries always resolve regardless of available slots.
#[test]
fn test_at_depth_always_resolves() {
    let mut book = Lorebook::new();

    let entry = Entry::parse(
        r#"---
id: depth_entry
constant: true
slot: !at_depth 3
---
Injected at depth"#,
        None,
    )
    .unwrap();

    book.add_entry(entry);

    let mut engine = ContextWeaver::new(book);

    // Empty available slots — but AtDepth should still work
    engine.set_available_slots(vec![]);

    let blocks = engine.assemble(&[]).unwrap();

    assert_block_contains(&blocks, "depth_entry", "Injected at depth");
    let block = find_block(&blocks, "depth_entry").unwrap();
    assert_eq!(block.slot, Slot::AtDepth(3));
}
