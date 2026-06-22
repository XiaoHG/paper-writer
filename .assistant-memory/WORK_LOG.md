# Work Log

## 2026-06-22

- Created `.assistant-memory/` as the only assistant-owned storage area in this
  repository workspace.
- Recorded the user's standing rule:
  - assistant memory must stay inside `.assistant-memory/` only
  - the recovery command is `恢复项目`
- Completed an initial read-only scan of the project.
- Captured current architecture, flow, and risks in `PROJECT_STATE.md`.
- Added comprehensive code comments/docstrings across the Python modules to make
  the project easier to read and maintain without changing behavior.
- Added an always-maintained graph flow diagram to `agent.py` and recorded the
  rule that future graph changes must update that diagram in the same change.
