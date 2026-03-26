from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .extractor import SchemaExtractor
from .models import ChatMessage, Role, ToolCall, ToolSchema, Trajectory, TrajectoryBatch, TrajectoryKind
from .prompts import base_context_prompt, missing_args_prompt, negative_chat_prompt, standard_prompt


class TrajectoryGenerator:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = client or OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.model = model

    def generate(self, tool_schema: ToolSchema, count_per_kind: int = 3) -> list[Trajectory]:
        schema_json = json.dumps(tool_schema.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
        trajectories: list[Trajectory] = []

        batches = [
            self._request_batch(tool_schema, schema_json, standard_prompt(count_per_kind), TrajectoryKind.STANDARD),
            self._request_batch(tool_schema, schema_json, missing_args_prompt(count_per_kind), TrajectoryKind.MISSING_ARGS),
            self._request_batch(tool_schema, schema_json, negative_chat_prompt(count_per_kind), TrajectoryKind.NEGATIVE_CHAT),
        ]

        for batch in batches:
            for trajectory in batch.trajectories:
                self._validate_trajectory(tool_schema, trajectory)
                trajectories.append(trajectory)
        return trajectories

    def save_jsonl(self, trajectories: list[Trajectory], output_file: str | Path) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for trajectory in trajectories:
                chatml_record = {
                    "messages": [self._message_to_chatml(m) for m in trajectory.messages],
                    "metadata": {"kind": trajectory.kind.value},
                }
                f.write(json.dumps(chatml_record, ensure_ascii=False) + "\n")

    def _request_batch(
        self,
        tool_schema: ToolSchema,
        schema_json: str,
        task_prompt: str,
        expected_kind: TrajectoryKind,
    ) -> TrajectoryBatch:
        base_prompt = base_context_prompt(tool_schema.name, schema_json, tool_schema.readme_summary)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        base_prompt
                        + "\n\nReturn ONLY valid JSON with this shape: "
                        + '{"trajectories":[{"kind":"standard|missing_args|negative_chat","messages":[{"role":"system|user|assistant|tool","content":"...","tool_call":{"name":"tool_name","arguments":{}}}] }]}'
                        + "\nDo not use markdown fences. Do not add explanations outside JSON."
                    ),
                },
                {"role": "user", "content": task_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = self._extract_response_text(response)
        batch = self._parse_batch(raw_text, expected_kind)
        for trajectory in batch.trajectories:
            trajectory.kind = expected_kind
        return batch

    def _extract_response_text(self, response: Any) -> str:
        choice = response.choices[0].message
        if getattr(choice, "content", None):
            return choice.content
        tool_calls = getattr(choice, "tool_calls", None) or []
        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            arguments = getattr(function, "arguments", None) if function else None
            if arguments:
                return arguments
        raise ValueError("Model response did not include parseable JSON content")

    def _parse_batch(self, raw_text: str, expected_kind: TrajectoryKind) -> TrajectoryBatch:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {exc}\nRaw text: {raw_text[:500]}") from exc

        trajectories_payload = payload.get("trajectories")
        if not isinstance(trajectories_payload, list):
            raise ValueError("Response JSON is missing a top-level 'trajectories' list")

        normalized: list[dict[str, Any]] = []
        for item in trajectories_payload:
            if "kind" not in item:
                item["kind"] = expected_kind.value
            normalized.append(item)

        try:
            return TrajectoryBatch.model_validate({"trajectories": normalized})
        except ValidationError as exc:
            raise ValueError(f"Trajectory batch validation failed: {exc}") from exc

    def _validate_trajectory(self, tool_schema: ToolSchema, trajectory: Trajectory) -> None:
        properties = tool_schema.parameters.properties
        required = set(tool_schema.parameters.required)

        for message in trajectory.messages:
            if message.tool_call is None:
                continue
            if message.tool_call.name != tool_schema.name:
                raise ValueError(f"Unexpected tool call name: {message.tool_call.name} != {tool_schema.name}")
            arguments = message.tool_call.arguments
            for req in required:
                if req not in arguments:
                    raise ValueError(f"Missing required argument '{req}' in tool call")
            for key, value in arguments.items():
                if key not in properties:
                    raise ValueError(f"Unexpected argument '{key}' for tool {tool_schema.name}")
                self._validate_value_against_property(key, value, properties[key].model_dump(exclude_none=True))

        if trajectory.kind == TrajectoryKind.MISSING_ARGS:
            if any(message.tool_call is not None for message in trajectory.messages):
                raise ValueError("Missing args trajectories must not contain tool calls")
        if trajectory.kind == TrajectoryKind.NEGATIVE_CHAT:
            if any(message.tool_call is not None for message in trajectory.messages):
                raise ValueError("Negative chat trajectories must not contain tool calls")

    def _validate_value_against_property(self, key: str, value: Any, prop: dict[str, Any]) -> None:
        expected_type = prop.get("type")
        if expected_type is None:
            return
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if any(self._matches_json_type(value, t) for t in allowed_types):
            enum_values = prop.get("enum")
            if enum_values is not None and value not in enum_values:
                raise ValueError(f"Argument '{key}' must be one of {enum_values}, got {value!r}")
            return
        raise ValueError(f"Argument '{key}' has invalid type. Expected {allowed_types}, got {type(value).__name__}")

    def _matches_json_type(self, value: Any, json_type: str) -> bool:
        return {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "null": lambda v: v is None,
        }.get(json_type, lambda _: True)(value)

    def _message_to_chatml(self, message: Any) -> dict[str, Any]:
        payload = {"role": message.role.value, "content": message.content}
        if message.tool_call is not None:
            payload["tool_call"] = {
                "name": message.tool_call.name,
                "arguments": message.tool_call.arguments,
            }
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tool-use trajectories from a Python tool")
    parser.add_argument("python_file")
    parser.add_argument("--readme", dest="readme_file")
    parser.add_argument("--function", dest="function_name")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--count", type=int, default=3, help="Trajectories per category")
    parser.add_argument("--output", help="Optional JSONL output path")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--base-url", dest="base_url")
    args = parser.parse_args()

    extractor = SchemaExtractor(args.python_file, args.readme_file)
    schema = extractor.extract(args.function_name)

    generator = TrajectoryGenerator(model=args.model, api_key=args.api_key, base_url=args.base_url)
    trajectories = generator.generate(schema, count_per_kind=args.count)

    if args.output:
        generator.save_jsonl(trajectories, args.output)
    else:
        print(json.dumps([t.model_dump(exclude_none=True) for t in trajectories], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
