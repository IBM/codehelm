#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

"""Render a graph.json MCTS search tree as a self-contained HTML/SVG viewer.

Usage:
    python examples/sfs/view_graph.py <path/to/graph.json>
    python examples/sfs/view_graph.py <path/to/graph.json> -o tree.html

Output defaults to graph.html in the same directory as the input.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path


# ── score → colour (green=good, red=bad, grey=root) ──────────────────────────
def score_color(score: float, is_best: bool) -> str:
    if score < 0:  # root sentinel
        return "#e5e7eb"
    if is_best:
        return "#22c55e"
    if score >= 0.99:
        return "#4ade80"
    if score >= 0.67:
        return "#a3e635"
    if score >= 0.34:
        return "#facc15"
    if score > 0:
        return "#fb923c"
    return "#fca5a5"


def score_text_color(score: float) -> str:
    return "#1f2328" if score >= 0 else "#57606a"


# ── layout: simple layered tree, nodes spread evenly per depth ───────────────


def build_layout(nodes: list[dict]) -> list[dict]:
    """Compute (x, y) pixel positions for each node using a layered approach."""
    # Group by depth
    by_depth: dict[int, list[int]] = {}
    for n in nodes:
        by_depth.setdefault(n["depth"], []).append(n["id"])

    max_depth = max(by_depth.keys(), default=0)
    NODE_W = 120
    NODE_H = 56
    V_GAP = 90
    H_GAP = 14

    # Assign horizontal positions depth by depth
    positions: dict[int, tuple[float, float]] = {}
    for depth in range(max_depth + 1):
        ids = by_depth.get(depth, [])
        total_w = len(ids) * NODE_W + (len(ids) - 1) * H_GAP
        start_x = -total_w / 2
        for i, nid in enumerate(ids):
            x = start_x + i * (NODE_W + H_GAP) + NODE_W / 2
            y = depth * (NODE_H + V_GAP) + NODE_H / 2
            positions[nid] = (x, y)

    canvas_w = max((len(by_depth.get(d, [])) for d in range(max_depth + 1)), default=1)
    canvas_w = max(canvas_w * (NODE_W + H_GAP) + 100, 800)
    canvas_h = (max_depth + 1) * (NODE_H + V_GAP) + 60

    result = []
    for n in nodes:
        x, y = positions[n["id"]]
        result.append(
            {
                **n,
                "x": x,
                "y": y,
                "w": NODE_W,
                "h": NODE_H,
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
            }
        )
    return result


def render_html(nodes_layout: list[dict], edges: list[dict], source: str) -> str:
    if not nodes_layout:
        return "<html><body>No nodes</body></html>"

    canvas_w = nodes_layout[0]["canvas_w"]
    canvas_h = nodes_layout[0]["canvas_h"]
    cx = canvas_w / 2  # centre offset

    id_to_pos = {n["id"]: (n["x"] + cx, n["y"]) for n in nodes_layout}
    nw = nodes_layout[0]["w"]
    nh = nodes_layout[0]["h"]

    # Edge paths
    edge_svg: list[str] = []
    for e in edges:
        sx, sy = id_to_pos.get(e["source"], (0, 0))
        tx, ty = id_to_pos.get(e["target"], (0, 0))
        # Curved bezier
        mid_y = (sy + ty) / 2
        edge_svg.append(
            f'<path d="M{sx:.1f},{sy + nh / 2:.1f} C{sx:.1f},{mid_y:.1f} {tx:.1f},{mid_y:.1f} {tx:.1f},{ty - nh / 2:.1f}" '
            f'stroke="#c0c4cc" stroke-width="1.5" fill="none"/>'
        )

    # Node rects + text
    node_svg: list[str] = []
    for n in nodes_layout:
        x, y = n["x"] + cx, n["y"]
        fill = score_color(n["score"], n["is_best"])
        tc = score_text_color(n["score"])
        stroke = "#22c55e" if n["is_best"] else "#c0c4cc"
        sw = "2.5" if n["is_best"] else "1"
        label = html.escape(n["label"])
        score_str = f"{n['score']:.2f}" if n["score"] >= 0 else "—"
        visits = n["visits"]
        tooltip = html.escape(
            f"depth={n['depth']} score={score_str} visits={visits}\n"
            f"action: {n['action'][:60]}\n"
            f"code: {n['code_snippet'][:80]}"
        )
        node_svg.append(
            f'<g class="node" data-id="{n["id"]}">'
            f"<title>{tooltip}</title>"
            f'<rect x="{x - nw / 2:.1f}" y="{y - nh / 2:.1f}" width="{nw}" height="{nh}" '
            f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
            f'<text x="{x:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="{tc}">{label}</text>'
            f'<text x="{x:.1f}" y="{y + 10:.1f}" text-anchor="middle" '
            f'font-size="10" fill="{tc}">score {score_str}</text>'
            f'<text x="{x:.1f}" y="{y + 22:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#57606a">visits {visits}</text>'
            f"</g>"
        )

    # Escape "<" so candidate code in a snippet cannot close or nest the <script> tag.
    node_data_json = json.dumps({n["id"]: n for n in nodes_layout}).replace(
        "<", "\\u003c"
    )

    total_nodes = len([n for n in nodes_layout if n["depth"] > 0])
    best_score = max((n["score"] for n in nodes_layout if n["score"] >= 0), default=0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MCTS Tree — {html.escape(source)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system,"Segoe UI",system-ui,sans-serif;
    font-size: 13px; background: #f7f8fa; color: #1f2328;
}}
header {{
    padding: 14px 20px; background: #fff; border-bottom: 1px solid #e5e7eb;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}}
header h1 {{ font-size: 15px; font-weight: 600; }}
.meta {{ font-size: 11px; color: #57606a; }}
.legend {{ display: flex; gap: 8px; align-items: center; margin-left: auto; flex-wrap: wrap; }}
.swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
.legend span {{ font-size: 11px; color: #57606a; }}
#canvas-wrap {{
    overflow: auto; padding: 24px;
    cursor: grab;
}}
#canvas-wrap:active {{ cursor: grabbing; }}
svg .node rect {{ cursor: pointer; }}
svg .node rect:hover {{ filter: brightness(0.92); }}
#detail {{
    position: fixed; bottom: 16px; right: 16px;
    background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
    padding: 12px 14px; font-size: 12px; max-width: 360px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08); display: none;
    white-space: pre-wrap; word-break: break-word; line-height: 1.5;
    max-height: 220px; overflow-y: auto;
}}
</style>
</head>
<body>
<header>
  <h1>MCTS Search Tree</h1>
  <span class="meta">{html.escape(source)} &nbsp;·&nbsp; {total_nodes} nodes &nbsp;·&nbsp; best score {best_score:.2f}</span>
  <div class="legend">
    <span class="swatch" style="background:#22c55e"></span><span>best</span>
    <span class="swatch" style="background:#facc15"></span><span>partial</span>
    <span class="swatch" style="background:#fca5a5"></span><span>0/N</span>
    <span class="swatch" style="background:#e5e7eb"></span><span>root</span>
  </div>
</header>
<div id="canvas-wrap">
<svg id="tree" width="{canvas_w:.0f}" height="{canvas_h:.0f}"
     xmlns="http://www.w3.org/2000/svg">
{"".join(edge_svg)}
{"".join(node_svg)}
</svg>
</div>
<div id="detail"></div>
<script>
const nodeData = {node_data_json};
document.querySelectorAll('.node').forEach(g => {{
  g.addEventListener('click', () => {{
    const id = parseInt(g.dataset.id);
    const n = nodeData[id];
    const d = document.getElementById('detail');
    if (!n) {{ d.style.display='none'; return; }}
    const score = n.score >= 0 ? n.score.toFixed(3) : '—';
    d.style.display = 'block';
    d.textContent = [
      'Node #' + id + '  depth=' + n.depth,
      'Score: ' + score + '  visits: ' + n.visits + '  value: ' + n.value.toFixed(3),
      n.action ? 'Action: ' + n.action : '',
      n.code_snippet ? 'Code: ' + n.code_snippet : '',
    ].filter(Boolean).join('\\n');
  }});
}});
document.getElementById('canvas-wrap').addEventListener('click', e => {{
  if (!e.target.closest('.node')) document.getElementById('detail').style.display='none';
}});
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render graph.json as HTML tree")
    parser.add_argument("input", help="Path to graph.json")
    parser.add_argument(
        "-o", "--output", help="Output HTML (default: graph.html next to input)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    data = json.loads(input_path.read_text())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    nodes_layout = build_layout(nodes)
    html_out = render_html(nodes_layout, edges, str(input_path))

    out_path = Path(args.output) if args.output else input_path.with_suffix(".html")
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Written {len(nodes)} nodes to {out_path}")


if __name__ == "__main__":
    main()
