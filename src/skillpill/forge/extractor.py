from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .models import JsonSchemaProperty, ToolParameters, ToolSchema


class SchemaExtractor:
    def __init__(self, python_file: str | Path, readme_file: str | Path | None = None):
        self.python_file = Path(python_file).expanduser().resolve()
        self.readme_file = Path(readme_file).expanduser().resolve() if readme_file else None
        self.source = self.python_file.read_text(encoding="utf-8")
        self.module_ast = ast.parse(self.source, filename=str(self.python_file))

    def extract(self, function_name: str | None = None) -> ToolSchema:
        function_node = self._select_function_node(function_name)
        readme_summary = self._readme_summary()
        schema = self._build_from_ast(function_node, readme_summary)

        if self._ast_has_unresolved_annotations(function_node):
            runtime_schema = self._build_from_runtime(function_node.name, readme_summary)
            if runtime_schema:
                schema = runtime_schema

        return schema

    def _select_function_node(self, function_name: str | None) -> ast.FunctionDef | ast.AsyncFunctionDef:
        functions = [
            node
            for node in self.module_ast.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
        ]
        if not functions:
            raise ValueError(f"No top-level public function found in {self.python_file}")

        if function_name:
            for fn in functions:
                if fn.name == function_name:
                    return fn
            raise ValueError(f"Function '{function_name}' not found in {self.python_file}")

        if len(functions) == 1:
            return functions[0]

        raise ValueError(
            "Multiple public functions found. Pass --function to select one explicitly: "
            + ", ".join(fn.name for fn in functions)
        )

    def _readme_summary(self) -> str | None:
        if not self.readme_file or not self.readme_file.exists():
            return None
        text = self.readme_file.read_text(encoding="utf-8").strip()
        return text[:4000] if text else None

    def _build_from_ast(self, fn: ast.FunctionDef | ast.AsyncFunctionDef, readme_summary: str | None) -> ToolSchema:
        parameters = ToolParameters()
        docstring = ast.get_docstring(fn)
        args = fn.args.args
        defaults = list(fn.args.defaults)
        default_offset = len(args) - len(defaults)

        for idx, arg in enumerate(args):
            if arg.arg in {"self", "cls"}:
                continue
            annotation = self._annotation_to_schema(arg.annotation)
            default = None
            if idx >= default_offset:
                default = self._literal_eval_safe(defaults[idx - default_offset])
                annotation.default = default
            else:
                parameters.required.append(arg.arg)
            parameters.properties[arg.arg] = annotation

        returns = self._annotation_to_schema(fn.returns) if fn.returns is not None else None
        description = docstring.splitlines()[0].strip() if docstring else f"Tool extracted from {fn.name}"

        return ToolSchema(
            name=fn.name,
            description=description,
            module_path=str(self.python_file),
            function_name=fn.name,
            docstring=docstring,
            readme_summary=readme_summary,
            parameters=parameters,
            returns=returns,
        )

    def _build_from_runtime(self, function_name: str, readme_summary: str | None) -> ToolSchema | None:
        try:
            spec = importlib.util.spec_from_file_location("skillpill_dynamic_module", self.python_file)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules["skillpill_dynamic_module"] = module
            spec.loader.exec_module(module)
            fn = getattr(module, function_name)
            signature = inspect.signature(fn)
            type_hints = inspect.get_annotations(fn, eval_str=True)
            docstring = inspect.getdoc(fn)

            properties: dict[str, JsonSchemaProperty] = {}
            required: list[str] = []
            for name, param in signature.parameters.items():
                if name in {"self", "cls"}:
                    continue
                schema = self._python_type_to_schema(type_hints.get(name, Any))
                if param.default is inspect.Signature.empty:
                    required.append(name)
                else:
                    schema.default = param.default
                properties[name] = schema

            returns = self._python_type_to_schema(type_hints.get("return", Any)) if "return" in type_hints else None
            description = docstring.splitlines()[0].strip() if docstring else f"Tool extracted from {function_name}"
            return ToolSchema(
                name=function_name,
                description=description,
                module_path=str(self.python_file),
                function_name=function_name,
                docstring=docstring,
                readme_summary=readme_summary,
                parameters=ToolParameters(properties=properties, required=required),
                returns=returns,
            )
        except Exception:
            return None

    def _ast_has_unresolved_annotations(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        args = [a.annotation for a in fn.args.args if a.annotation is not None]
        if fn.returns is not None:
            args.append(fn.returns)
        for ann in args:
            if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
                return True
        return False

    def _annotation_to_schema(self, annotation: ast.expr | None) -> JsonSchemaProperty:
        if annotation is None:
            return JsonSchemaProperty(type=["string", "number", "boolean", "object", "array", "null"])

        if isinstance(annotation, ast.Name):
            return self._simple_name_to_schema(annotation.id)

        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            return JsonSchemaProperty(type="string", description=f"Forward annotation: {annotation.value}")

        if isinstance(annotation, ast.Subscript):
            base = self._expr_name(annotation.value)
            if base in {"list", "List"}:
                return JsonSchemaProperty(type="array", items=self._annotation_to_schema(self._slice_value(annotation)).model_dump(exclude_none=True))
            if base in {"dict", "Dict"}:
                return JsonSchemaProperty(type="object", additionalProperties=True)
            if base in {"Optional"}:
                inner = self._annotation_to_schema(self._slice_value(annotation))
                if isinstance(inner.type, list):
                    if "null" not in inner.type:
                        inner.type.append("null")
                else:
                    inner.type = [inner.type, "null"]
                return inner
            if base in {"Literal"}:
                literal_values = self._literal_values(self._slice_value(annotation))
                literal_type = list({self._python_value_to_json_type(v) for v in literal_values})
                return JsonSchemaProperty(type=literal_type[0] if len(literal_type) == 1 else literal_type, enum=literal_values)
            if base in {"Union"}:
                values = self._slice_tuple(annotation)
                types = [self._annotation_to_schema(v).type for v in values]
                flat: list[str] = []
                for t in types:
                    if isinstance(t, list):
                        flat.extend(t)
                    else:
                        flat.append(t)
                return JsonSchemaProperty(type=sorted(set(flat)))

        if isinstance(annotation, ast.Attribute):
            return self._simple_name_to_schema(annotation.attr)

        return JsonSchemaProperty(type="string", description=f"Unparsed annotation: {ast.unparse(annotation)}")

    def _simple_name_to_schema(self, name: str) -> JsonSchemaProperty:
        mapping = {
            "str": JsonSchemaProperty(type="string"),
            "int": JsonSchemaProperty(type="integer"),
            "float": JsonSchemaProperty(type="number"),
            "bool": JsonSchemaProperty(type="boolean"),
            "dict": JsonSchemaProperty(type="object", additionalProperties=True),
            "list": JsonSchemaProperty(type="array", items={"type": "string"}),
            "Any": JsonSchemaProperty(type=["string", "number", "boolean", "object", "array", "null"]),
        }
        return mapping.get(name, JsonSchemaProperty(type="string", description=f"Custom type: {name}"))

    def _python_type_to_schema(self, annotation: Any) -> JsonSchemaProperty:
        if annotation is Any:
            return JsonSchemaProperty(type=["string", "number", "boolean", "object", "array", "null"])
        if annotation is str:
            return JsonSchemaProperty(type="string")
        if annotation is int:
            return JsonSchemaProperty(type="integer")
        if annotation is float:
            return JsonSchemaProperty(type="number")
        if annotation is bool:
            return JsonSchemaProperty(type="boolean")
        if annotation is dict:
            return JsonSchemaProperty(type="object", additionalProperties=True)
        if annotation is list:
            return JsonSchemaProperty(type="array", items={"type": "string"})
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            return JsonSchemaProperty(type="object", properties=annotation.model_json_schema().get("properties", {}), additionalProperties=False)

        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in {list, tuple}:
            item_schema = self._python_type_to_schema(args[0] if args else Any)
            return JsonSchemaProperty(type="array", items=item_schema.model_dump(exclude_none=True))
        if origin is dict:
            return JsonSchemaProperty(type="object", additionalProperties=True)
        if str(origin).endswith("Literal"):
            values = list(args)
            json_types = sorted({self._python_value_to_json_type(v) for v in values})
            return JsonSchemaProperty(type=json_types[0] if len(json_types) == 1 else json_types, enum=values)
        if origin is None and hasattr(annotation, "__name__"):
            return JsonSchemaProperty(type="string", description=f"Custom type: {annotation.__name__}")
        if str(origin).endswith("Union"):
            flattened: list[str] = []
            for arg in args:
                sub = self._python_type_to_schema(arg).type
                if isinstance(sub, list):
                    flattened.extend(sub)
                else:
                    flattened.append(sub)
            return JsonSchemaProperty(type=sorted(set(flattened)))
        return JsonSchemaProperty(type="string", description=f"Custom type: {annotation}")

    def _expr_name(self, expr: ast.expr) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return ast.unparse(expr)

    def _slice_value(self, node: ast.Subscript) -> ast.expr:
        return node.slice

    def _slice_tuple(self, node: ast.Subscript) -> list[ast.expr]:
        if isinstance(node.slice, ast.Tuple):
            return list(node.slice.elts)
        return [node.slice]

    def _literal_values(self, node: ast.expr) -> list[Any]:
        if isinstance(node, ast.Tuple):
            return [self._literal_eval_safe(elt) for elt in node.elts]
        return [self._literal_eval_safe(node)]

    def _literal_eval_safe(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception:
            return ast.unparse(node)

    def _python_value_to_json_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int) and not isinstance(value, bool):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        if value is None:
            return "null"
        return "string"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a tool schema from a Python file")
    parser.add_argument("python_file")
    parser.add_argument("--readme", dest="readme_file")
    parser.add_argument("--function", dest="function_name")
    args = parser.parse_args()

    extractor = SchemaExtractor(args.python_file, args.readme_file)
    schema = extractor.extract(args.function_name)
    print(json.dumps(schema.model_dump(exclude_none=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
