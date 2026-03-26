# SkillPill Task Plan

## Goal
Create a public GitHub repository `skillpill`, implement Phase 1 of the PRD with production Python code for `SchemaExtractor` and `TrajectoryGenerator`, then upload the code.

## Phases
- [complete] 1. Initialize project structure and planning files
- [complete] 2. Create public GitHub repository
- [complete] 3. Implement project files and Phase 1 code
- [complete] 4. Validate code structure
- [in_progress] 5. Upload code to GitHub

## Decisions
- Focus strictly on Phase 1
- Use Python package layout under `src/skillpill`
- Use Pydantic models for schemas and generated trajectories
- Use AST parsing as the primary extraction mechanism, with import/inspect fallback when safe
