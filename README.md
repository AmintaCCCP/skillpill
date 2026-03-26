# SkillPill

SkillPill distills classical agent skills, Python scripts, and API tools into ultra-lightweight MLX LoRA adapters, called Pills, for local execution on Apple Silicon.

## Phase 1: Skill-Forge

This repository currently focuses on the highest-priority part of the PRD: converting Python tools into high-quality training trajectories for LoRA fine-tuning.

Implemented in this phase:

- `SchemaExtractor`: parses Python source files, extracts function signatures, type hints, defaults, docstrings, and emits a strict JSON-schema-like tool definition.
- `TrajectoryGenerator`: calls a cloud LLM to generate three categories of tool-use trajectories:
  - Standard
  - Missing Args
  - Negative/Chat
- Strict Pydantic models for tool schemas and generated trajectories.

## Project Layout

```text
skillpill/
  datasets/
  examples/
  src/skillpill/
    forge/
      extractor.py
      generator.py
      models.py
      prompts.py
```

## Requirements

- Python 3.11+
- `pydantic>=2`
- `openai>=1.0`
- `instructor>=1.0`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Extract a schema

```bash
python -m skillpill.forge.extractor path/to/tool.py --readme README.md --function get_weather
```

### Generate trajectories

```bash
export OPENAI_API_KEY=...
python -m skillpill.forge.generator path/to/tool.py --readme README.md --function get_weather --model gpt-4o-mini --count 3
```

## Notes

- AST extraction is the default and safest path.
- Runtime import inspection is supported as a fallback when AST metadata is incomplete.
- Generated trajectories are validated against the extracted parameter schema before being returned.
