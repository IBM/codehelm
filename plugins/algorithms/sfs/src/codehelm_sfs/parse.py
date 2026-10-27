#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

from __future__ import annotations

import ast
import re

DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def parse_code_block(string: str, lang: str = "python") -> str | None:
    """Extract code enclosed within markdown code blocks or fallback to first function."""
    code_pattern = rf"```{lang}\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)

    # Also match ```python ... ``` without trailing newline if at end of string
    loose_pattern = rf"```{lang}\s*\n(.*?)```"
    match = re.search(loose_pattern, string, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    loose_generic = r"```\s*\n(.*?)```"
    match = re.search(loose_generic, string, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    return parse_first_func(string, lang)


def parse_first_func(code: str, lang: str = "python") -> str | None:
    """Extract the first function definition from raw text."""
    if lang != "python":
        return code.strip() if code.strip() else None

    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    got_return = False
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == -1:
                def_i = i
            else:
                break
        elif "return" in line and def_i != -1:
            got_return = True
        if line == "" and def_i != -1 and got_return:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == -1:
        return None

    return "\n".join(code_lines[def_i : last_i + 1]).rstrip("[/PYTHON]")


def add_code_block(string: str, lang: str = "python") -> str:
    """Wrap code in a markdown code block."""
    return f"```{lang}\n{string}\n```"


def parse_updated_insights(text: str) -> str:
    """Extract updated insights section from text."""
    lines = text.split("\n")
    in_updated_insights = False
    insights_content = []

    for line in lines:
        if "Updated Insights" in line:
            in_updated_insights = True
            continue

        if in_updated_insights:
            if line.strip() == "":
                break
            insights_content.append(line.strip())

    return "\n".join(insights_content)


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    return "\n".join(["    " + line for line in split])


def fix_markdown(func_body: str) -> str:
    return re.sub(r"`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("
    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)

    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f"{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}"
        try:
            ast.parse(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = (
                handle_first_line_indent
                if cur_state == 0
                else handle_entire_body_indent
            )
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except (NameError, TypeError, ValueError):
            return f_body

    return parse_indent_rec(func_body, 0)


def py_is_syntax_valid(code: str) -> bool:
    """Check whether Python code parses cleanly with AST."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def parse_tests(tests: str) -> list[str]:
    """Parse a test block into individual test cases.

    First groups consecutive non-blank lines by blank-line separators so that
    multi-line tests (setup assignments followed by an assert) are kept
    together. Then, for groups where every line is a standalone assert,
    splits them back into individual lines.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    for line in tests.splitlines():
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        else:
            if current:
                groups.append(current)
            current = []
    if current:
        groups.append(current)

    result: list[str] = []
    for group in groups:
        if not any("assert" in line for line in group):
            continue
        if all("assert" in line for line in group):
            result.extend(group)
        else:
            result.append("\n".join(group))
    return result


def parse_improvement_ideas(string: str, num_ideas: int) -> list[str]:
    """Parse improvement ideas from LLM output (supports multiple formatting styles)."""
    patterns = [
        re.compile(
            r"(?:\*\*)?Idea \d+:(?:\*\*)?\s*(.+?)(?=(?:\*\*)?Idea \d+:|$)", re.DOTALL
        ),
        re.compile(r"^\d+\.\s+\*\*(.+?)\*\*", re.MULTILINE),
        re.compile(r"^\d+\.\s+(.+?)(?=^\d+\.|$)", re.MULTILINE | re.DOTALL),
    ]

    matches = []
    for pattern in patterns:
        matches = pattern.findall(string)
        if matches:
            matches = [m.strip().replace("\n", " ").replace("  ", " ") for m in matches]
            break

    out = matches[:num_ideas]
    if len(out) < num_ideas:
        out += ["Improve the code."] * (num_ideas - len(out))

    return out
