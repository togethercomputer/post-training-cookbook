# Vendored encoder provenance

Files here are byte-for-byte copies of a model release's own encoder. Never edit or reformat them: ruff excludes each
encoder file, and a test pins each file's sha256. To update one, copy the new release file unchanged, update its row
here and the pinned hash, and re-run the codec's goldens.

## `deepseek_v41_encoding.py`

| Field | Value |
|---|---|
| Source repo | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| Revision | `08c8657f02291e11b6f6b63e0e16014f8e748518` |
| Source path | `encoding/encoding.py` |
| Git blob | `3c7c357dc9ae3ed37adacb8572a2e9498b7d1bcb` (37,962 bytes) |
| sha256 | `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2` |
| License | MIT, `LICENSE` at the same revision, copied as `LICENSE.deepseek-v41` (blob `d62e3bef9f054f21b7fc616365850fbf879a99ff`) |
| Imports | stdlib only (`typing`, `copy`, `json`, `re`); its one `open()` is in `load_cases`, which the codec never calls |

`deepseek-ai/DeepSeek-V4.1-Flash@314113f8a7dadb388215396ba5fbee2c610a04d6` ships the same `encoding/encoding.py`,
`tokenizer.json` (`6a15814d…`) and `tokenizer_config.json` (`f3dad388…`). Only `chat_template.jinja` differs between the
two repos.

`DeepSeekV41Codec` (`post_training_cookbook/codecs/deepseek_v41.py`) renders every prompt with this file's `encode_messages`.

### Where the 1017 `chat_template.jinja` differs from this file

The release has no tokenizer-config template; `chat_template.jinja` (blob `35b5c611…`) is a port of this file. The
codec follows this file, the release's reference encoder, so the jinja deltas below do not apply to it:

| Behaviour | `encoding.py` (the codec) | 1017 `chat_template.jinja` |
|---|---|---|
| Thinking mode | `thinking_mode` is required | defaults to `thinking`; `thinking` and `enable_thinking` are aliases; `reasoning_effort="none"` turns thinking off |
| Reasoning effort | int 1–100, or `low`/`high`/`max` (50/75/100); anything else asserts | also `none`/`minimal`/`medium`/`xhigh`, mapped to 75/50/75/75 (V4.1-Flash's jinja mapped `medium` to 62) |
| Tool-call arguments | dict, or a JSON string it parses (up to two levels; non-JSON becomes one `arguments` parameter) | dict only; a JSON string raises |
| Assistant reasoning field | `reasoning_content` | `reasoning_content`, or `thinking` as an alias |
| Trailing user or tool turn | always appends `<｜Assistant｜>` plus `<think>` or `</think>` | appends it only with `add_generation_prompt` |
| Tools | read from a message's `tools` field | a top-level `tools` argument attaches to the leading system message, or to a new empty one |

On the card-application path the two render the same text: the codec attaches the tools to the system message as the
jinja does, every prompt ends in a user or tool turn and asks for the generation header, and the encoder parses the
canonical JSON-string arguments into the dicts the jinja requires.

The codec honours `add_generation_prompt=False` by removing the encoder's trailing generation header.
Explicit `preserve_thinking=False` strips history reasoning before rendering, including when tools are declared.
