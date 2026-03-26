from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import instructor
from openai import OpenAI

from .extractor import SchemaExtractor
from .models import ToolSchema, Trajectory, TrajectoryBatch, TrajectoryKind
from .prompts import base_context_prompt, missing_args_prompt, negative_chat_prompt, standard_prompt


class TrajectoryGenerator:
    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o-mini"):
        raw_client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = instructor.from_openai(raw_client)
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
        batch = self.client.chat.completions.create(
            model=self.model,
            response_model=TrajectoryBatch,
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": task_prompt},
            ],
        )
        for trajectory in batch.trajectories:
            trajectory.kind = expected_kind
        return batch

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
    args = parser.parse_args()

    extractor = SchemaExtractor(args.python_file, args.readme_file)
    schema = extractor.extract(args.function_name)

    generator = TrajectoryGenerator(model=args.model)
    trajectories = generator.generate(schema, count_per_kind=args.count)

    if args.output:
        generator.save_jsonl(trajectories, args.output)
    else:
        print(json.dumps([t.model_dump(exclude_none=True) for t in trajectories], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
