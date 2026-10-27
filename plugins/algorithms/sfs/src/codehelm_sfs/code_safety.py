#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

"""AST-based safety checker for LLM-generated Python code.

Detects potentially dangerous patterns before execution using AST inspection.
Intentionally non-blocking: surfaces warnings and adds diagnostic feedback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

_DANGEROUS_CALLS: list[tuple[str, str | None]] = [
    ("exec", None),
    ("eval", None),
    ("compile", None),
    ("__import__", None),
    ("importlib", "import_module"),
    ("os", "system"),
    ("os", "popen"),
    ("os", "execve"),
    ("os", "execvp"),
    ("os", "execvpe"),
    ("os", "exec"),
    ("os", "spawnl"),
    ("os", "spawnle"),
    ("os", "spawnlp"),
    ("os", "spawnlpe"),
    ("os", "spawnv"),
    ("os", "spawnve"),
    ("os", "spawnvp"),
    ("os", "spawnvpe"),
    ("os", "fork"),
    ("os", "kill"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "removedirs"),
    ("os", "rename"),
    ("os", "replace"),
    ("os", "mkdir"),
    ("os", "makedirs"),
    ("os", "chmod"),
    ("os", "chown"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("subprocess", "getoutput"),
    ("subprocess", "getstatusoutput"),
    ("shutil", "rmtree"),
    ("shutil", "move"),
    ("shutil", "copy"),
    ("shutil", "copytree"),
    ("open", None),
    ("socket", "socket"),
    ("urllib", "request"),
    ("urllib", "urlopen"),
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "request"),
    ("httpx", "get"),
    ("httpx", "post"),
    ("httpx", "Client"),
    ("aiohttp", "ClientSession"),
    ("ctypes", None),
    ("pickle", "loads"),
    ("marshal", "loads"),
]

_DANGEROUS_BUILTINS: frozenset[str] = frozenset(
    name for name, attr in _DANGEROUS_CALLS if attr is None
)
_DANGEROUS_ATTRS: frozenset[tuple[str, str]] = frozenset(
    (name, attr) for name, attr in _DANGEROUS_CALLS if attr is not None
)


@dataclass(frozen=True)
class DangerFlag:
    """A single dangerous pattern found in code."""

    pattern: str
    lineno: int
    description: str

    def __str__(self) -> str:
        return f"Line {self.lineno}: [{self.pattern}] {self.description}"


@dataclass
class SafetyReport:
    """Aggregated safety check report."""

    flags: list[DangerFlag] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return len(self.flags) == 0

    def summary(self) -> str:
        if self.is_safe:
            return "SAFETY CHECK: No dangerous patterns detected."
        lines = ["SAFETY CHECK: Dangerous patterns detected:"]
        lines.extend(f"  ⚠ {flag}" for flag in self.flags)
        return "\n".join(lines)


class _DangerVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.flags: list[DangerFlag] = []

    def _flag(self, node: ast.AST, pattern: str, description: str) -> None:
        lineno = getattr(node, "lineno", 0)
        self.flags.append(
            DangerFlag(pattern=pattern, lineno=lineno, description=description)
        )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _DANGEROUS_BUILTINS:
            self._flag(node, func.id, f"Direct call to built-in '{func.id}'")
        elif isinstance(func, ast.Attribute):
            obj_name: str | None = None
            if isinstance(func.value, ast.Name):
                obj_name = func.value.id
            elif isinstance(func.value, ast.Attribute):
                obj_name = func.value.attr
            if obj_name is not None and (obj_name, func.attr) in _DANGEROUS_ATTRS:
                self._flag(
                    node,
                    f"{obj_name}.{func.attr}",
                    f"Call to dangerous function '{obj_name}.{func.attr}'",
                )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        dangerous_modules = {name for name, _ in _DANGEROUS_CALLS}
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in dangerous_modules:
                self._flag(
                    node,
                    f"import {alias.name}",
                    f"Import of potentially dangerous module '{alias.name}'",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        dangerous_modules = {name for name, _ in _DANGEROUS_CALLS}
        module = node.module or ""
        top = module.split(".")[0]
        if top in dangerous_modules:
            self._flag(
                node,
                f"from {module} import ...",
                f"Import from potentially dangerous module '{module}'",
            )
        self.generic_visit(node)


def check_code_safety(code: str) -> SafetyReport:
    """Parse *code* and scan its AST for dangerous patterns."""
    report = SafetyReport()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return report

    visitor = _DangerVisitor()
    visitor.visit(tree)
    report.flags.extend(visitor.flags)
    return report
