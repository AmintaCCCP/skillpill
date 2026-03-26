from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class JsonSchemaProperty(BaseModel):
    type: str | list[str]
    description: str | None = None
    enum: list[Any] | None = None
    default: Any | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None
    additionalProperties: bool | dict[str, Any] | None = None


class ToolParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, JsonSchemaProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolSchema(BaseModel):
    name: str
    description: str
    module_path: str
    function_name: str
    docstring: str | None = None
    readme_summary: str | None = None
    parameters: ToolParameters
    returns: JsonSchemaProperty | None = None


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    role: Role
    content: str
    tool_call: ToolCall | None = None


class TrajectoryKind(str, Enum):
    STANDARD = "standard"
    MISSING_ARGS = "missing_args"
    NEGATIVE_CHAT = "negative_chat"


class Trajectory(BaseModel):
    kind: TrajectoryKind
    messages: list[ChatMessage]

    @model_validator(mode="after")
    def validate_sequence(self) -> "Trajectory":
        if not self.messages:
            raise ValueError("Trajectory must include at least one message")
        if self.messages[0].role != Role.SYSTEM:
            raise ValueError("Trajectory must start with a system message")
        return self


class TrajectoryBatch(BaseModel):
    trajectories: list[Trajectory]


class GeneratedToolArguments(BaseModel):
    arguments: dict[str, Any]
