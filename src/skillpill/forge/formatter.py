from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .extractor import SchemaExtractor
from .models import ChatMessage, ToolSchema, Trajectory, TrajectoryBatch, TrajectoryKind


class DatasetFormatter:
    def __init__(self, tool_schema: ToolSchema):
        self.tool_schema = tool_schema

    def load_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        input_path = Path(path)
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {input_path}: {exc}") from exc
        return records

    def format_records(self, records: list[dict[str, Any]], dedupe: bool = True) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        seen: set[str] = set()

        for record in records:
            trajectory = self._record_to_trajectory(record)
            self._validate_trajectory(trajectory)
            chatml = {
                "messages": [self._message_to_chatml(m) for m in trajectory.messages],
                "metadata": {"kind": trajectory.kind.value},
            }
            fingerprint = json.dumps(chatml, ensure_ascii=False, sort_keys=True)
            if dedupe and fingerprint in seen:
                continue
            seen.add(fingerprint)
            formatted.append(chatml)
        return formatted

    def save_jsonl(self, records: list[dict[str, Any]], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _record_to_trajectory(self, record: dict[str, Any]) -> Trajectory:
        if "messages" in record and "metadata" in record:
            kind = record.get("metadata", {}).get("kind")
            payload = {"kind": kind, "messages": record["messages"]}
        elif "kind" in record and "messages" in record:
            payload = record
        elif "trajectories" in record:
            batch = TrajectoryBatch.model_validate(record)
            if len(batch.trajectories) != 1:
                raise ValueError("Formatter expects one trajectory per record when using a trajectories wrapper")
            return batch.trajectories[0]
        else:
            raise ValueError("Record must contain either {kind,messages}, {messages,metadata}, or {trajectories}")

        try:
            return Trajectory.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid trajectory record: {exc}") from exc

    def _validate_trajectory(self, trajectory: Trajectory) -> None:
        messages = trajectory.messages
        if not messages:
            raise ValueError("Trajectory has no messages")
        if messages[0].role.value != "system":
            raise ValueError("Trajectory must start with a system message")

        tool_call_count = 0
        for idx, message in enumerate(messages):
            if message.tool_call is None:
                continue
            tool_call_count += 1
            if message.role.value != "assistant":
                raise ValueError(f"Tool call must appear on assistant message, got {message.role.value} at index {idx}")
            self._validate_tool_call(message)
            if idx + 1 >= len(messages) or messages[idx + 1].role.value != "tool":
                raise ValueError("Assistant tool call must be followed by a tool message")

        if trajectory.kind == TrajectoryKind.STANDARD:
            if tool_call_count != 1:
                raise ValueError("Standard trajectories must contain exactly one tool call")
        else:
            if tool_call_count != 0:
                raise ValueError(f"{trajectory.kind.value} trajectories must not contain tool calls")

    def _validate_tool_call(self, message: ChatMessage) -> None:
        assert message.tool_call is not None
        tool_call = message.tool_call
        if tool_call.name != self.tool_schema.name:
            raise ValueError(f"Unexpected tool name {tool_call.name}, expected {self.tool_schema.name}")

        properties = self.tool_schema.parameters.properties
        required = set(self.tool_schema.parameters.required)
        arguments = tool_call.arguments

        missing = required - set(arguments)
        if missing:
            raise ValueError(f"Missing required arguments: {sorted(missing)}")

        if not self.tool_schema.parameters.additionalProperties:
            extras = set(arguments) - set(properties)
            if extras:
                raise ValueError(f"Unexpected arguments: {sorted(extras)}")

        for key, value in arguments.items():
            if key not in properties:
                continue
            self._validate_value_against_property(key, value, properties[key].model_dump(exclude_none=True))

    def _validate_value_against_property(self, key: str, value: Any, prop: dict[str, Any]) -> None:
        expected_type = prop.get("type")
        if expected_type is None:
            return
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(self._matches_json_type(value, t) for t in allowed_types):
            raise ValueError(f"Argument '{key}' has invalid type. Expected {allowed_types}, got {type(value).__name__}")

        enum_values = prop.get("enum")
        if enum_values is not None and value is not None and value not in enum_values:
            raise ValueError(f"Argument '{key}' must be one of {enum_values}, got {value!r}")

        if isinstance(value, list) and prop.get("items"):
            item_schema = prop["items"]
            for item in value:
                self._validate_value_against_property(f"{key}[]", item, item_schema)

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

    def _message_to_chatml(self, message: ChatMessage) -> dict[str, Any]:
        payload = {"role": message.role.value, "content": message.content}
        if message.tool_call is not None:
            payload["tool_call"] = {
                "name": message.tool_call.name,
                "arguments": message.tool_call.arguments,
            }
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Format and validate generated trajectories into strict ChatML JSONL")
    parser.add_argument("input_file")
    parser.add_argument("python_file")
    parser.add_argument("--readme", dest="readme_file")
    parser.add_argument("--function", dest="function_name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    extractor = SchemaExtractor(args.python_file, args.readme_file)
    schema = extractor.extract(args.function_name)

    formatter = DatasetFormatter(schema)
    records = formatter.load_jsonl(args.input_file)
    formatted = formatter.format_records(records, dedupe=not args.no_dedupe)
    formatter.save_jsonl(formatted, args.output)

    print(json.dumps({
        "input_records": len(records),
        "output_records": len(formatted),
        "output_file": str(args.output),
        "tool": schema.name,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
