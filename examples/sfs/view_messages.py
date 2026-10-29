#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

"""Convert a messages.jsonl file into a self-contained HTML conversation viewer.

Usage:
    python examples/sfs/view_messages.py <path/to/messages.jsonl>
    python examples/sfs/view_messages.py <path/to/messages.jsonl> -o out.html

Output file defaults to messages.html in the same directory as the input.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

LANG_TAG = re.compile(r"[A-Za-z0-9_+#.-]*")

ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #1f2328;
    background: #ffffff;
    padding: 24px;
    max-width: 960px;
    margin: 0 auto;
}
h1 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
.meta { font-size: 12px; color: #57606a; margin-bottom: 24px; }
.request {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-bottom: 20px;
    overflow: hidden;
}
.request-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    background: #f7f8fa;
    border-bottom: 1px solid #e5e7eb;
    cursor: pointer;
    user-select: none;
}
.request-header:hover { background: #eef0f3; }
.req-num {
    font-weight: 700;
    font-size: 13px;
    color: #3b82d4;
    min-width: 24px;
}
.req-temp {
    font-size: 11px;
    color: #57606a;
    background: #e5e7eb;
    padding: 1px 6px;
    border-radius: 10px;
}
.req-preview {
    font-size: 12px;
    color: #57606a;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.toggle-icon { font-size: 11px; color: #57606a; margin-left: auto; }
.request-body { display: none; }
.request-body.open { display: block; }
.message {
    padding: 12px 16px;
    border-bottom: 1px solid #e5e7eb;
}
.message:last-child { border-bottom: none; }
.msg-role {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
}
.role-system  { background: #dce9ff; color: #1a3a8a; }
.role-user    { background: #e8edf2; color: #2c3e50; }
.role-assistant { background: #d4f0d4; color: #1a5c1a; }
.role-response  { background: #fff0d4; color: #7a4800; }
.msg-content {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 13px;
    line-height: 1.55;
}
.msg-content code, .code-block {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 12px;
    background: #f6f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 10px 12px;
    display: block;
    white-space: pre;
    overflow-x: auto;
    margin: 6px 0;
}
.response-block {
    background: #fffdf0;
}
"""

JS = """
function toggle(id) {
    var body = document.getElementById(id);
    var icon = document.getElementById(id + '-icon');
    if (body.classList.contains('open')) {
        body.classList.remove('open');
        icon.textContent = '▶';
    } else {
        body.classList.add('open');
        icon.textContent = '▼';
    }
}
function expandAll() {
    document.querySelectorAll('.request-body').forEach(function(b) {
        b.classList.add('open');
    });
    document.querySelectorAll('.toggle-icon').forEach(function(i) {
        i.textContent = '▼';
    });
}
function collapseAll() {
    document.querySelectorAll('.request-body').forEach(function(b) {
        b.classList.remove('open');
    });
    document.querySelectorAll('.toggle-icon').forEach(function(i) {
        i.textContent = '▶';
    });
}
"""


def render_content(text: str) -> str:
    """Render message content: code fences as <code> blocks, rest as escaped text."""
    parts = text.split("```")
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # plain text
            out.append(f'<span class="msg-content">{html.escape(part)}</span>')
        else:
            # inside a code fence — strip optional language tag on first line
            first, sep, rest = part.partition("\n")
            if sep and LANG_TAG.fullmatch(first.strip()):
                lang, code = first.strip(), rest
            else:
                lang, code = "", part
            label = (
                f'<span style="font-size:10px;color:#888;margin-bottom:4px;display:block">{html.escape(lang)}</span>'
                if lang
                else ""
            )
            out.append(f'<code class="code-block">{label}{html.escape(code)}</code>')
    return "".join(out)


def build_html(records: list[dict], source_path: str) -> str:
    body_parts: list[str] = []

    for rec in records:
        req_num = rec.get("request_num", "?")
        temp = rec.get("temperature", "?")
        messages = rec.get("messages", [])
        response = rec.get("response", "")

        # preview = first user message, truncated
        first_user = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        preview = html.escape(first_user[:120].replace("\n", " "))

        body_id = f"req-{req_num}"
        header = (
            f'<div class="request-header" onclick="toggle(\'{body_id}\')">'
            f'<span class="req-num">#{req_num}</span>'
            f'<span class="req-temp">T={temp}</span>'
            f'<span class="req-preview">{preview}</span>'
            f'<span class="toggle-icon" id="{body_id}-icon">▶</span>'
            f"</div>"
        )

        msg_html: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            label_cls = f"role-{role}" if role in ROLE_LABELS else "role-user"
            label_text = ROLE_LABELS.get(role, role.capitalize())
            msg_html.append(
                f'<div class="message">'
                f'<span class="msg-role {label_cls}">{label_text}</span>'
                f"<div>{render_content(content)}</div>"
                f"</div>"
            )

        # response block
        msg_html.append(
            f'<div class="message response-block">'
            f'<span class="msg-role role-response">Response</span>'
            f"<div>{render_content(response)}</div>"
            f"</div>"
        )

        body_parts.append(
            f'<div class="request">'
            f"{header}"
            f'<div class="request-body" id="{body_id}">{"".join(msg_html)}</div>'
            f"</div>"
        )

    total = len(records)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SFS Message Log — {html.escape(source_path)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>SFS Message Log</h1>
<div class="meta">
  {html.escape(source_path)} &nbsp;·&nbsp; {total} request{"s" if total != 1 else ""}
  &nbsp;|&nbsp;
  <a href="#" onclick="expandAll(); return false">expand all</a>
  &nbsp;/&nbsp;
  <a href="#" onclick="collapseAll(); return false">collapse all</a>
</div>
{"".join(body_parts)}
<script>{JS}</script>
</body>
</html>"""
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description="Render messages.jsonl as HTML")
    parser.add_argument("input", help="Path to messages.jsonl")
    parser.add_argument(
        "-o", "--output", help="Output HTML file (default: messages.html next to input)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    records: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping malformed line: {e}", file=sys.stderr)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")
    html_content = build_html(records, str(input_path))
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Written {len(records)} requests to {output_path}")


if __name__ == "__main__":
    main()
