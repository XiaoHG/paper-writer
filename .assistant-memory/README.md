# Assistant Memory

This directory is reserved for assistant-managed history and memory only.
Nothing outside this directory should be used to store assistant notes, logs,
rules, or recovery data.

## User rules to preserve

- Keep assistant-owned memory isolated inside `.assistant-memory/` only.
- Do not mix assistant memory into project source files.
- When the user says `恢复项目`, use the files in this directory to recover the
  latest working context quickly.
- Whenever `agent.py` graph structure changes, update the graph flow diagram in
  `agent.py` comments in the same change.

## Recovery procedure for future Codex workbenches

When the user says `恢复项目`, do this first:

1. Read `README.md`.
2. Read `PROJECT_STATE.md`.
3. Read `WORK_LOG.md`.
4. Summarize current project status, recent findings, open risks, and next
   recommended actions.

## Files

- `PROJECT_STATE.md`: current understanding of the project.
- `WORK_LOG.md`: chronological work log and durable decisions.
