# Post Training Cookbook skills

[Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills) bundled with this repo. A skill is a `SKILL.md` (plus optional `references/` and `scripts/`) that Claude Code loads on demand when your request matches what the skill does — so the model gets repo-specific expertise without you pasting it in each time.

| Skill | What it covers |
|---|---|
| [`together-post-training`](together-post-training/) | The full post-training fine-tuning workflow on Together AI — GRPO (single- and multi-turn), SFT, DPO, custom losses, LoRA / full-weight training, checkpointing, and endpoint deployment via the Post Training Cookbook and Together Post Training SDK. |

## Install

Skills are only auto-discovered under a `.claude/skills/` directory. Run the installer to wire these in:

```bash
./skills/install.sh            # for your user — ~/.claude/skills (available in any repo)
./skills/install.sh --project  # for this repo only — .claude/skills
```

By default it **symlinks** each skill, so edits in the repo stay live. Use `--copy` for a frozen snapshot, or `--uninstall` to remove the links. Then restart Claude Code (or start a new session) to pick them up.

> The skill's examples (`python recipes/grpo/gsm8k.py`, `from post_training_cookbook...`) assume you're working inside a checkout of this repo with its `.venv` active. The skill supplies the *know-how*; the cookbook supplies the code it drives.

## Use

Once installed, you don't call a skill explicitly — Claude Code reads each skill's description at startup and invokes the matching one when your request fits. For example:

- *"Set up GRPO training on Qwen3.8-27B for a custom trivia dataset"*
- *"Write a reward function and register a new dataset"*
- *"Deploy my fine-tuned checkpoint to a dedicated endpoint"*

You can also invoke it by name with `/together-post-training`.

## Add a skill

Create a new `skills/<skill-name>/SKILL.md` with `name` and `description` frontmatter, add it to the table above, and re-run `./skills/install.sh`.
