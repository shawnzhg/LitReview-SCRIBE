"""Minimal JSON Schema validator for the keywords the BioLitBench schemas use, installed as
jsonschema only when that package is missing."""

from __future__ import annotations

import re
from collections import deque

SUPPORTED = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "pattern",
             "minimum", "maximum", "minItems", "maxItems", "$schema", "$id", "title", "description", "default"}
_TYPES = {
    "object": lambda x: isinstance(x, dict),
    "array": lambda x: isinstance(x, list),
    "string": lambda x: isinstance(x, str),
    "boolean": lambda x: isinstance(x, bool),
    "null": lambda x: x is None,
    "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    "integer": lambda x: (isinstance(x, int) and not isinstance(x, bool)) or (isinstance(x, float) and x.is_integer()),
}


class ValidationError(Exception):
    def __init__(self, message, path):
        super().__init__(message)
        self.message = message
        self.path = deque(path)


def _check_keywords(schema, where="#"):
    if isinstance(schema, bool):
        return
    bad = set(schema) - SUPPORTED
    if bad:
        raise NotImplementedError(f"mini jsonschema: unsupported keyword(s) {sorted(bad)} at {where}")
    for k, sub in (schema.get("properties") or {}).items():
        _check_keywords(sub, f"{where}/properties/{k}")
    if isinstance(schema.get("additionalProperties"), dict):
        _check_keywords(schema["additionalProperties"], f"{where}/additionalProperties")
    if isinstance(schema.get("items"), dict):
        _check_keywords(schema["items"], f"{where}/items")


class Draft202012Validator:
    def __init__(self, schema):
        _check_keywords(schema)
        self.schema = schema

    def iter_errors(self, instance):
        yield from self._errors(self.schema, instance, [])

    def _errors(self, s, x, path):
        if s is True:
            return
        if s is False:
            yield ValidationError("False schema does not allow the value", path)
            return
        t = s.get("type")
        if t is not None:
            ts = t if isinstance(t, list) else [t]
            if not any(_TYPES[k](x) for k in ts):
                yield ValidationError(f"{x!r:.60} is not of type {t!r}", path)
                return
        if "enum" in s and x not in s["enum"]:
            yield ValidationError(f"{x!r:.60} is not one of {s['enum']!r}", path)
        if "const" in s and x != s["const"]:
            yield ValidationError(f"{s['const']!r} was expected", path)
        if isinstance(x, str) and "pattern" in s and not re.search(s["pattern"], x):
            yield ValidationError(f"{x!r:.60} does not match {s['pattern']!r}", path)
        if _TYPES["number"](x):
            if "minimum" in s and x < s["minimum"]:
                yield ValidationError(f"{x} is less than the minimum of {s['minimum']}", path)
            if "maximum" in s and x > s["maximum"]:
                yield ValidationError(f"{x} is greater than the maximum of {s['maximum']}", path)
        if isinstance(x, list):
            if "minItems" in s and len(x) < s["minItems"]:
                yield ValidationError(f"{len(x)} items is fewer than minItems {s['minItems']}", path)
            if "maxItems" in s and len(x) > s["maxItems"]:
                yield ValidationError(f"{len(x)} items is more than maxItems {s['maxItems']}", path)
            if isinstance(s.get("items"), (dict, bool)):
                for i, v in enumerate(x):
                    yield from self._errors(s["items"], v, path + [i])
        if isinstance(x, dict):
            for k in s.get("required") or []:
                if k not in x:
                    yield ValidationError(f"{k!r} is a required property", path)
            props = s.get("properties") or {}
            for k, v in x.items():
                if k in props:
                    yield from self._errors(props[k], v, path + [k])
                elif "additionalProperties" in s:
                    ap = s["additionalProperties"]
                    if ap is False:
                        yield ValidationError(f"Additional properties are not allowed ({k!r} was unexpected)", path)
                    elif isinstance(ap, dict):
                        yield from self._errors(ap, v, path + [k])
