# Development Notes

## Active Docs First

For current Entry/Exit AI work, read
`docs/ACTIVE_SUPER_AI_BOT_GOAL_20260702.md`,
`docs/ENTRY_FOUNDATION_AUDIT_20260628.md` and
`docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md` before editing.
Those files define the current fail-closed gates, smart520 roadmap, RAM
discipline and no-shadow/no-live/no-promotion rule.

## Code Quality Checklist

**IMPORTANT: Before committing any changes, always verify:**

1. ✅ **Check for terminal errors:**
   ```bash
   python3 -m py_compile <modified_files>
   ```
   Ensure no syntax errors or import issues.

2. ✅ **Check linter warnings:**
   - Review IDE/linter warnings
   - Fix critical issues (errors, not warnings)
   - Document expected warnings (e.g., pytest imports if pytest not installed)

3. ✅ **Run tests (if applicable):**
   ```bash
   pytest -q  # If pytest is installed
   ```

4. ✅ **Verify imports work:**
   ```bash
   python3 -c "import <module>; print('OK')"
   ```

## Known Linter Warnings

### pytest imports
Some test files may show linter warnings about `pytest` imports:
- **Reason:** pytest may not be installed in all development environments
- **Action:** This is expected and safe to ignore. Tests will run fine if pytest is installed.
- **Fix:** Add `# type: ignore[reportMissingImports]` comment if needed

### Missing type stubs
Some third-party libraries may not have type stubs:
- **Action:** Use `# type: ignore` comments for known false positives
- **Document:** Add notes in code comments explaining why

## Pre-Commit Checklist

- [ ] No syntax errors (`py_compile` passes)
- [ ] No critical linter errors
- [ ] Imports resolve correctly
- [ ] Code follows project style (PEP 8)
- [ ] Tests pass (if applicable)
- [ ] Documentation updated (if needed)

## Git Workflow

1. Make changes
2. **Check terminal for errors** ← CRITICAL STEP
3. Fix any issues found
4. Stage changes: `git add <files>`
5. Commit with descriptive message
6. Push to GitHub
