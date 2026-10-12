# Forked chat template provenance
## `qwen3_5_preserve_thinking.jinja`

- **Source repo:** `Qwen/Qwen3.5-0.8B`
- **Source revision:** `2fc06364715b967f1860aea9cf38778875588b17`
- **Source field:** `chat_template.jinja`
- **Behavior relative to upstream:**
  - `<think>...</think>` blocks render for every assistant turn with `reasoning_content`, regardless of its position relative to the last user query. The fork does not track `last_query_index`.
  - Message sequences without a user turn render for training and scoring. The fork does not enforce upstream's `No user query found in messages.` guard.
  - The upstream reverse scan and `ns.multi_step_tool` remain in the template but do not affect rendering.
  - Upstream validation for empty messages, misplaced system messages, and unexpected role/content remains intact.

Qwen3.6 and Qwen3.8 use their own templates, which accept `preserve_thinking` natively. `Qwen35Codec` forwards that flag; those templates require a user turn.
