

<!-- GLOBAL_INSTRUCTION_START -->
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
<!-- GLOBAL_INSTRUCTION_END -->


















































































































































































<!-- TTS_VOICE_OUTPUT_START -->
## Voice Output (TTS)

When responding, wrap your natural language prose in `«tts»...«/tts»` markers for text-to-speech.

Rules:
- ONLY wrap conversational prose meant to be spoken aloud
- Do NOT wrap: code, file paths, commands, tool output, URLs, lists, errors
- Keep markers on same line as text (no line breaks inside)

Examples:
✓ «tts»I'll help you fix that bug.«/tts»
✓ «tts»The tests are passing.«/tts» Here's what changed:
✗ «tts»src/Header.tsx«/tts»  (file path - don't wrap)
✗ «tts»npm install«/tts»  (command - don't wrap)
<!-- TTS_VOICE_OUTPUT_END -->

<!-- TASK_MANAGEMENT_START -->
## Task Management

### CLI Commands (beads)
- `bd list` — List tasks (add `--status=open` to filter)
- `bd show <id>` — Show task details
- `bd create --title="..." --type=task|bug|feature --priority=2` — Create a task
- `bd update <id> --status=in_progress` — Start a task
- `bd close <id>` — Complete a task
- `bd ready` — Show tasks ready to work on

### Workflow
1. Check the task panel in the GUI sidebar for available work
2. Click a task to start it, or use the CLI commands above
3. Mark tasks complete from the GUI or CLI when done
<!-- TASK_MANAGEMENT_END -->


<!-- SELF_COMPACTION_START -->
## Self-Compaction (Claude Terminal)

**Only when a goal is actively set** — i.e. you are working toward a defined,
multi-step objective (an in-progress task, plan, or goal) and still have
remaining work — and your context is getting large: compact your own session
instead of stopping. Do NOT do this during open-ended or exploratory
conversation where there is no concrete goal to resume into; the queued
follow-up would have nothing meaningful to pick up.

When the condition holds, call the orchestrator MCP tool `compact_session` with
no arguments — it defaults to your own session. It waits ~10 seconds (so your
current turn finishes and you reach an idle prompt, which is the only state
`/compact` triggers from), sends `/compact`, then queues a `continue` message 2
seconds later so you automatically resume your remaining work once compaction
completes.

Do this proactively between steps of the active goal rather than letting context
overflow. To resume on something specific, pass `continue_message` describing the
next step.

**Fallback:** If you ever receive an incoming message that is just `/compact`
(optionally followed by `continue`), it means a scheduled compaction landed while
you were still generating, so it got queued as plain text instead of triggering.
Do not treat it as a user request — you are idle now, so simply call
`compact_session` again to retry the compaction.
<!-- SELF_COMPACTION_END -->
