# GitHub Skill Example Dataset

This folder contains a concrete end-to-end SkillPill example based on a GitHub-style tool.

## Files

- `github_skill_tool.py`: example Python tool definition
- `github_skill_README.md`: semantic context used during generation
- `../datasets/examples/github_skill_example.jsonl`: generated training dataset

## What this example demonstrates

This example was generated to help contributors understand what a useful Phase 1 output should look like.

It includes three trajectory categories:

- `standard`: the user provides enough information, so the assistant calls the tool
- `missing_args`: the user intent matches the tool, but required fields are missing, so the assistant asks a clarification question
- `negative_chat`: the user request is unrelated, so the assistant replies normally without using the tool

## Dataset summary

The current dataset contains 18 records total:

- 6 `standard`
- 6 `missing_args`
- 6 `negative_chat`

## How it was generated

The dataset was generated with the local SkillPill generator against an OpenAI-compatible endpoint, using the `gpt-5.4` model for reliable JSON output during testing.

Sensitive local test configuration, including API base URL and API key, is intentionally not stored in this repository.

## Why this example exists

Most repositories stop at vague architecture diagrams and hand-wavy promises. This example is here so contributors can inspect:

- the tool source
- the extracted schema shape
- the final JSONL training data

and then build their own examples without guessing what “good” is supposed to look like.
