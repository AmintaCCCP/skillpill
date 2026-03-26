from __future__ import annotations

from textwrap import dedent


def base_context_prompt(tool_name: str, schema_json: str, readme_summary: str | None) -> str:
    readme_block = readme_summary or "No README summary provided."
    return dedent(
        f"""
        You are generating supervised training data for a tool-using assistant.

        Tool name: {tool_name}
        Tool schema:
        {schema_json}

        README summary:
        {readme_block}

        All outputs must be realistic, concise, and internally consistent.
        Assistant behavior rules:
        - Only call the tool when the user intent clearly matches the tool.
        - If required arguments are missing, ask a clarification question instead of fabricating values.
        - If the user request is unrelated, do not call the tool.
        - Final responses should be natural and helpful.
        """
    ).strip()


def standard_prompt(count: int) -> str:
    return dedent(
        f"""
        Generate exactly {count} trajectories of type `standard`.
        Each trajectory must include:
        1. system message
        2. user message with a valid request that should trigger the tool
        3. assistant message with a tool_call containing fully valid arguments
        4. tool message with a realistic observation/result
        5. assistant message with a concise final user-facing response
        """
    ).strip()


def missing_args_prompt(count: int) -> str:
    return dedent(
        f"""
        Generate exactly {count} trajectories of type `missing_args`.
        Each trajectory must include:
        1. system message
        2. user message that implies the tool should be used but omits one or more required arguments
        3. assistant message asking a clear clarification question

        Do not include a tool call if required fields are missing.
        Do not invent user parameters.
        """
    ).strip()


def negative_chat_prompt(count: int) -> str:
    return dedent(
        f"""
        Generate exactly {count} trajectories of type `negative_chat`.
        Each trajectory must include:
        1. system message
        2. user message unrelated to the tool
        3. assistant message that replies normally without any tool call

        Make the unrelated conversations varied and realistic.
        """
    ).strip()
