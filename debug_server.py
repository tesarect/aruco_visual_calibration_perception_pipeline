#!/usr/bin/env python3
# Bare minimal local webpage to visually inspect aruco_pose.py's
# preprocessing debug sweep (run run_debug_sweep() / aruco_pose.py first —
# this just reads debug_output/results.json and serves the saved images).
#
# Debugging/inspection tool only — never runs on the production rosject
# server. Run inside YOLO-pipeline/venv only.
#
# Usage:
#   source venv/bin/activate
#   python3 debug_server.py
#   (open http://localhost:5050 in a browser)

import json
from pathlib import Path

from flask import Flask, send_from_directory

OUT_DIR = Path("debug_output")
RESULTS_FILE = OUT_DIR / "results.json"

app = Flask(__name__)


def load_results():
    if not RESULTS_FILE.exists():
        return None
    with open(RESULTS_FILE) as f:
        return json.load(f)


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(OUT_DIR, filename)


@app.route("/")
def index():
    data = load_results()
    if not data:
        return "<p>No debug_output/results.json found — run aruco_pose.py first.</p>"

    images = data["images"]
    total_time = data["total_sweep_time_s"]

    options = "".join(f'<option value="{e["image"]}">{e["image"]}</option>' for e in images)

    return f"""
    <html>
    <head>
        <title>ArUco preprocessing debug</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            select {{ font-size: 16px; padding: 4px; margin-bottom: 8px; }}
            #sweepTime {{ color: #666; margin-bottom: 20px; }}
            #tiles {{ display: flex; flex-wrap: wrap; gap: 16px; }}
            .tile {{
                border: 2px solid #ccc;
                padding: 8px;
                width: 220px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .tile .title {{ font-weight: bold; margin-bottom: 8px; text-align: center; }}
            .tile .imgbox {{
                width: 200px;
                height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #fafafa;
            }}
            .tile img {{
                image-rendering: pixelated;
                max-width: 200px;
                max-height: 200px;
            }}
            .tile.ok {{ border-color: #2a2; background: #eafbea; }}
            .tile.fail {{ border-color: #c33; background: #fdeaea; }}
            table {{ border-collapse: collapse; margin-top: 20px; }}
            td, th {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
            .tick {{ color: #2a2; font-weight: bold; }}
            #chartSection {{ margin-top: 40px; }}
            #chartSection h3 {{ margin-bottom: 4px; }}
            .bar-label {{ font-size: 11px; }}
            .bar-value {{ font-size: 11px; fill: #333; }}
        </style>
    </head>
    <body>
        <h2>ArUco preprocessing debug</h2>
        <select id="picker" onchange="render()">{options}</select>
        <div id="sweepTime">Total sweep time (all images, all variants): {total_time:.2f}s</div>
        <div id="tiles"></div>
        <table id="results"></table>

        <div id="chartSection">
            <h3>Overall: avg. detection time per process (Y axis inverted — higher point = faster)</h3>
            <div id="chart"></div>
        </div>

        <script>
            const DATA = {json.dumps(images)};

            function renderChart() {{
                // one point per pipeline variant: its average time_s (ms)
                // across every image where it successfully detected a
                // marker. Order follows PIPELINES order (as they first
                // appear in the data), so the X axis reads left-to-right
                // the same way the tiles above do.
                const attempted = DATA.filter(e => e.yolo_found_box);
                if (attempted.length === 0) {{
                    document.getElementById('chart').innerHTML = '<p>No images with a YOLO-detected box to chart.</p>';
                    return;
                }}

                const variantNames = attempted[0].variants.map(v => v.name);
                const points = variantNames.map(name => {{
                    const times = attempted
                        .flatMap(e => e.variants)
                        .filter(v => v.name === name && v.detected)
                        .map(v => v.time_s * 1000);
                    const avg = times.length ? times.reduce((a, b) => a + b, 0) / times.length : null;
                    return {{ name, avg, count: times.length }};
                }});

                const plotted = points.filter(p => p.avg !== null);
                if (plotted.length === 0) {{
                    document.getElementById('chart').innerHTML = '<p>No successful detections to chart.</p>';
                    return;
                }}

                const colWidth = 70;
                const chartHeight = 200;
                const topPad = 30;   // room for value labels above the highest (fastest) point
                const bottomPad = 50; // room for rotated variant-name labels below the axis
                const maxAvg = Math.max(...plotted.map(p => p.avg));
                const svgWidth = points.length * colWidth;

                // Y axis is inverted: 0ms at the BOTTOM, maxAvg at the TOP,
                // so a faster (smaller ms) process plots lower on the page
                // (standard axis orientation, just with fast=low/slow=high).
                const yFor = (avg) => topPad + chartHeight - (avg / maxAvg) * chartHeight;

                let dots = '';
                let pathPoints = [];
                points.forEach((p, i) => {{
                    const x = i * colWidth + colWidth / 2;
                    if (p.avg === null) {{
                        dots += `<text class="bar-label" x="${{x}}" y="${{topPad + chartHeight / 2}}" text-anchor="middle">no data</text>`;
                        return;
                    }}
                    const y = yFor(p.avg);
                    pathPoints.push(`${{x}},${{y}}`);
                    dots += `
                        <circle cx="${{x}}" cy="${{y}}" r="4" fill="#5b9bd5"></circle>
                        <text class="bar-value" x="${{x}}" y="${{y - 10}}" text-anchor="middle">${{p.avg.toFixed(2)}}ms</text>
                        <text class="bar-label" x="0" y="0" text-anchor="middle"
                              transform="translate(${{x}}, ${{topPad + chartHeight + 14}}) rotate(30)">${{p.name}}</text>
                    `;
                }});

                const curve = pathPoints.length > 1
                    ? `<polyline points="${{pathPoints.join(' ')}}" fill="none" stroke="#5b9bd5" stroke-width="2"></polyline>`
                    : '';

                document.getElementById('chart').innerHTML = `
                    <svg width="${{svgWidth + 40}}" height="${{topPad + chartHeight + bottomPad}}">
                        <line x1="0" y1="${{topPad}}" x2="${{svgWidth}}" y2="${{topPad}}" stroke="#eee"></line>
                        <text class="bar-label" x="0" y="${{topPad - 8}}">${{maxAvg.toFixed(2)}}ms</text>
                        <line x1="0" y1="${{topPad + chartHeight}}" x2="${{svgWidth}}" y2="${{topPad + chartHeight}}" stroke="#eee"></line>
                        <text class="bar-label" x="0" y="${{topPad + chartHeight + 28}}">0ms</text>
                        ${{curve}}
                        ${{dots}}
                    </svg>
                `;
            }}

            function render() {{
                const name = document.getElementById('picker').value;
                const entry = DATA.find(e => e.image === name);
                const tilesDiv = document.getElementById('tiles');
                const table = document.getElementById('results');
                const stem = entry.image.replace('.png', '');

                if (!entry.yolo_found_box) {{
                    tilesDiv.innerHTML = '<p><b>YOLO found no aruco_marker candidate box in this image.</b></p>';
                    table.innerHTML = '';
                    return;
                }}

                // Rank rule for "quickest detection": among variants that
                // actually detected a marker, rank by real time_s (1 =
                // fastest detected variant, 2 = second-fastest, ...).
                // Variants that failed to detect get no rank at all.
                const rankByName = {{}};
                entry.variants
                    .filter(v => v.detected)
                    .slice()
                    .sort((a, b) => a.time_s - b.time_s)
                    .forEach((v, i) => {{ rankByName[v.name] = i + 1; }});

                let tiles = `<div class="tile">
                    <div class="title">original crop</div>
                    <div class="imgbox"><img src="/images/${{stem}}/00_yolo_crop.png"></div>
                </div>`;
                let rows = '<tr><th>variant</th><th>detected</th><th>time (this variant)</th>' +
                           '<th>quickest detection</th><th>tvec (x,y,z m)</th></tr>';

                for (const v of entry.variants) {{
                    const cls = v.detected ? 'ok' : 'fail';
                    const imgFile = v.detected ? v.overlay_file : v.file;
                    tiles += `<div class="tile ${{cls}}">
                        <div class="title">${{v.name}}</div>
                        <div class="imgbox"><img src="/images/${{stem}}/${{imgFile}}"></div>
                    </div>`;
                    const timeMs = (v.time_s * 1000).toFixed(1) + ' ms';
                    const rank = rankByName[v.name];
                    const rankLabel = rank ? `<span class="tick">${{rank}}</span>` : '';
                    rows += `<tr><td>${{v.name}}</td><td>${{v.detected ? 'YES' : 'no'}}</td>` +
                            `<td>${{timeMs}}</td><td>${{rankLabel}}</td>` +
                            `<td>${{v.tvec_m ? v.tvec_m.map(x => x.toFixed(4)).join(', ') : '-'}}</td></tr>`;
                }}
                rows += `<tr><td colspan="5"><b>Total time for this image (all variants): ${{entry.image_time_s.toFixed(3)}}s</b></td></tr>`;

                tilesDiv.innerHTML = tiles;
                table.innerHTML = rows;
            }}
            render();
            renderChart();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)