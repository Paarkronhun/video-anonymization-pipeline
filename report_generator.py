"""
report_generator.py
=====================
Builds a self-contained HTML safety report from a completed analysis run:
traffic counts, per-class speed statistics, and flagged PET near-miss
conflicts, with severity tiers and per-conflict detail.

Output is a single .html file (Chart.js loaded from CDN) so it can be
opened directly in a browser or emailed without bundling assets.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

try:
    from .pet_analyzer import PETConflict, WorldTrajectory
except ImportError:
    from pet_analyzer import PETConflict, WorldTrajectory

SEVERITY_COLOR = {
    "critical": "#e23d28",
    "high":     "#f08c2e",
    "moderate": "#e0b03c",
    "low":      "#6b8f5e",
}

SEVERITY_LABEL = {
    "critical": "Critical",
    "high":     "High",
    "moderate": "Moderate",
    "low":      "Low",
}


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def _seconds_to_timestamp(t: float) -> str:
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


def build_safety_report(
    *,
    output_path: str,
    video_name: str,
    video_duration_s: float,
    trajectories: dict[int, WorldTrajectory],
    conflicts: list[PETConflict],
    pet_threshold_s: float,
    homography_reprojection_error_m: float | None = None,
) -> str:
    """
    Build and write the HTML report. Returns the output_path for convenience.
    """

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ------------------------------------------------------------------
    # Aggregate traffic stats
    # ------------------------------------------------------------------
    by_label_counts: dict[str, int] = {}
    by_label_speeds: dict[str, list[float]] = {}

    for traj in trajectories.values():
        by_label_counts[traj.label] = by_label_counts.get(traj.label, 0) + 1
        if len(traj.speeds) > 0:
            # Use the 85th percentile speed per track as its "characteristic"
            # speed -- standard traffic-engineering practice (85th-percentile
            # speed), more robust than mean/max to brief detection jitter.
            p85 = float(np.percentile(traj.speeds, 85))
            by_label_speeds.setdefault(traj.label, []).append(p85 * 3.6)  # m/s -> km/h

    total_vehicles = sum(c for lbl, c in by_label_counts.items()
                          if lbl in {"car", "motorcycle", "bus", "truck"})
    total_vulnerable = sum(c for lbl, c in by_label_counts.items()
                            if lbl in {"pedestrian", "cyclist"})

    speed_summary = {}
    for label, speeds in by_label_speeds.items():
        speed_summary[label] = {
            "mean": float(np.mean(speeds)),
            "p85": float(np.percentile(speeds, 85)) if len(speeds) > 1 else speeds[0],
            "max": float(np.max(speeds)),
            "n": len(speeds),
        }

    severity_counts = {"critical": 0, "high": 0, "moderate": 0, "low": 0}
    for c in conflicts:
        severity_counts[c.severity] += 1

    # ------------------------------------------------------------------
    # Chart data
    # ------------------------------------------------------------------
    traffic_chart = {
        "labels": list(by_label_counts.keys()),
        "data": list(by_label_counts.values()),
    }

    pet_histogram_bins = np.arange(0, pet_threshold_s + 0.5, 0.25)
    pet_values = [c.pet_seconds for c in conflicts]
    hist_counts, _ = np.histogram(pet_values, bins=pet_histogram_bins) if pet_values else (
        np.zeros(len(pet_histogram_bins) - 1), None
    )
    pet_histogram = {
        "labels": [f"{pet_histogram_bins[i]:.2f}-{pet_histogram_bins[i+1]:.2f}s"
                    for i in range(len(pet_histogram_bins) - 1)],
        "data": [int(x) for x in hist_counts],
    }

    speed_chart = {
        "labels": list(speed_summary.keys()),
        "mean": [round(v["mean"], 1) for v in speed_summary.values()],
        "p85": [round(v["p85"], 1) for v in speed_summary.values()],
    }

    # ------------------------------------------------------------------
    # Conflict table rows
    # ------------------------------------------------------------------
    conflict_rows_html = []
    for i, c in enumerate(conflicts, start=1):
        color = SEVERITY_COLOR[c.severity]
        conflict_rows_html.append(f"""
        <tr class="conflict-row" data-severity="{c.severity}">
          <td class="num">{i:02d}</td>
          <td><span class="sev-pill" style="--sev-color:{color}">{SEVERITY_LABEL[c.severity]}</span></td>
          <td class="mono">{_fmt(c.pet_seconds, 2)}s</td>
          <td>{c.vehicle_label.title()} <span class="track-id">#{c.vehicle_id}</span></td>
          <td>{c.vulnerable_label.title()} <span class="track-id">#{c.vulnerable_id}</span></td>
          <td class="mono">{_fmt(c.vehicle_speed_mps * 3.6, 1)} km/h</td>
          <td class="mono">{_fmt(c.vulnerable_speed_mps * 3.6, 1)} km/h</td>
          <td class="mono">{_seconds_to_timestamp(c.vehicle_arrival_s)}</td>
          <td class="mono">frame {c.frame_idx}</td>
        </tr>""")

    conflict_rows = "".join(conflict_rows_html) if conflicts else """
        <tr><td colspan="9" class="empty-state">No near-miss conflicts detected below the
        {:.1f}s PET threshold for this video.</td></tr>""".format(pet_threshold_s)

    # ------------------------------------------------------------------
    # Speed table rows
    # ------------------------------------------------------------------
    speed_rows = "".join(f"""
        <tr>
          <td>{label.title()}</td>
          <td class="mono">{v['n']}</td>
          <td class="mono">{_fmt(v['mean'], 1)} km/h</td>
          <td class="mono">{_fmt(v['p85'], 1)} km/h</td>
          <td class="mono">{_fmt(v['max'], 1)} km/h</td>
        </tr>""" for label, v in speed_summary.items())

    calibration_note = ""
    if homography_reprojection_error_m is not None:
        warn_class = "calib-warn" if homography_reprojection_error_m > 0.5 else ""
        calibration_note = f"""
        <div class="calib-note {warn_class}">
          Ground-plane calibration reprojection error: {_fmt(homography_reprojection_error_m, 3)} m
          {' — speed/PET figures may be less reliable; recheck calibration points.' if homography_reprojection_error_m > 0.5 else '(within expected tolerance).'}
        </div>"""

    html = HTML_TEMPLATE.format(
        video_name=video_name,
        generated_at=generated_at,
        video_duration=_seconds_to_timestamp(video_duration_s),
        total_vehicles=total_vehicles,
        total_vulnerable=total_vulnerable,
        total_conflicts=len(conflicts),
        critical_count=severity_counts["critical"],
        high_count=severity_counts["high"],
        moderate_count=severity_counts["moderate"],
        low_count=severity_counts["low"],
        pet_threshold=pet_threshold_s,
        conflict_rows=conflict_rows,
        speed_rows=speed_rows if speed_rows else '<tr><td colspan="5" class="empty-state">No speed data available.</td></tr>',
        calibration_note=calibration_note,
        traffic_chart_json=json.dumps(traffic_chart),
        pet_histogram_json=json.dumps(pet_histogram),
        speed_chart_json=json.dumps(speed_chart),
        severity_chart_json=json.dumps({
            "labels": ["Critical", "High", "Moderate", "Low"],
            "data": [severity_counts["critical"], severity_counts["high"],
                     severity_counts["moderate"], severity_counts["low"]],
            "colors": [SEVERITY_COLOR["critical"], SEVERITY_COLOR["high"],
                       SEVERITY_COLOR["moderate"], SEVERITY_COLOR["low"]],
        }),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ==========================================================
# HTML TEMPLATE
# ==========================================================
# Design intent: an engineering/DOT traffic-safety report, not a SaaS
# dashboard. Asphalt-dark hero with amber/red severity accents (matches
# real-world traffic signal/warning convention), monospace for all
# measured figures so columns of numbers stay legible and scannable,
# a serif display face for headings to read as a formal report rather
# than a product UI.

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Intersection Safety Report — {video_name}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root {{
    --asphalt:      #1c1e22;
    --asphalt-2:    #26292e;
    --paper:        #f6f4ee;
    --paper-dim:    #e7e3d8;
    --line:         #3a3d42;
    --amber:        #f0a93b;
    --ink:          #1c1e22;
    --ink-soft:     #5a5d63;
    --critical:     {critical_color};
    --high:         {high_color};
    --moderate:     {moderate_color};
    --low:          {low_color};
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: 'Iowan Old Style', 'Palatino Linotype', Georgia, serif;
    line-height: 1.5;
  }}

  .mono {{
    font-family: 'SF Mono', 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.92em;
    letter-spacing: -0.01em;
  }}

  header.hero {{
    background: linear-gradient(160deg, var(--asphalt) 0%, var(--asphalt-2) 100%);
    color: var(--paper);
    padding: 56px 48px 40px;
    border-bottom: 4px solid var(--amber);
  }}

  .hero-top {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 24px;
  }}

  .hero-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--amber);
    margin: 0 0 10px;
  }}

  h1 {{
    font-size: clamp(1.8rem, 3vw, 2.6rem);
    margin: 0;
    font-weight: 600;
    letter-spacing: -0.01em;
  }}

  .hero-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #b7bac0;
    text-align: right;
  }}

  .stat-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1px;
    background: var(--line);
    margin-top: 36px;
    border-top: 1px solid var(--line);
  }}

  .stat {{
    background: var(--asphalt);
    padding: 20px 18px;
  }}

  .stat .value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: var(--paper);
    line-height: 1;
  }}

  .stat .label {{
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #9a9da3;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
  }}

  .stat.flag-critical .value {{ color: var(--critical); }}
  .stat.flag-high .value {{ color: var(--high); }}

  main {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 48px 48px 80px;
  }}

  section {{ margin-bottom: 56px; }}

  h2 {{
    font-size: 1.3rem;
    font-weight: 600;
    border-bottom: 2px solid var(--ink);
    padding-bottom: 10px;
    margin-bottom: 22px;
    display: flex;
    align-items: baseline;
    gap: 10px;
  }}

  h2 .section-no {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--ink-soft);
    font-weight: 400;
  }}

  .charts-grid {{
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 28px;
  }}

  .chart-card {{
    background: white;
    border: 1px solid var(--paper-dim);
    border-radius: 4px;
    padding: 20px 22px;
  }}

  .chart-card h3 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-soft);
    margin: 0 0 14px;
    font-weight: 600;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    font-size: 0.92rem;
  }}

  th {{
    text-align: left;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ink-soft);
    padding: 10px 14px;
    border-bottom: 2px solid var(--ink);
  }}

  td {{
    padding: 11px 14px;
    border-bottom: 1px solid var(--paper-dim);
  }}

  tr.conflict-row:hover {{ background: var(--paper-dim); }}

  td.num {{ font-family: 'IBM Plex Mono', monospace; color: var(--ink-soft); }}

  .track-id {{ color: var(--ink-soft); font-size: 0.82em; }}

  .sev-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: white;
    background: var(--sev-color);
  }}

  .empty-state {{
    text-align: center;
    color: var(--ink-soft);
    font-style: italic;
    padding: 28px 0;
  }}

  .calib-note {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink-soft);
    margin-top: 14px;
    padding: 10px 14px;
    border-left: 3px solid var(--line);
    background: var(--paper-dim);
  }}
  .calib-note.calib-warn {{
    border-left-color: var(--high);
    color: #8a5a1e;
  }}

  .methodology {{
    font-size: 0.88rem;
    color: var(--ink-soft);
    background: white;
    border: 1px solid var(--paper-dim);
    border-radius: 4px;
    padding: 20px 24px;
  }}

  .methodology p {{ margin: 0 0 10px; }}
  .methodology p:last-child {{ margin-bottom: 0; }}

  footer {{
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: var(--ink-soft);
    padding: 30px 0 50px;
  }}

  @media (max-width: 800px) {{
    .charts-grid {{ grid-template-columns: 1fr; }}
    header.hero, main {{ padding-left: 22px; padding-right: 22px; }}
  }}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-top">
    <div>
      <p class="hero-eyebrow">Intersection Safety Monitor — Conflict Report</p>
      <h1>{video_name}</h1>
    </div>
    <div class="hero-meta">
      Generated {generated_at}<br>
      Footage duration: {video_duration}<br>
      PET threshold: {pet_threshold}s
    </div>
  </div>

  <div class="stat-strip">
    <div class="stat">
      <div class="value">{total_vehicles}</div>
      <div class="label">Vehicles tracked</div>
    </div>
    <div class="stat">
      <div class="value">{total_vulnerable}</div>
      <div class="label">Pedestrians / cyclists</div>
    </div>
    <div class="stat">
      <div class="value">{total_conflicts}</div>
      <div class="label">Flagged conflicts</div>
    </div>
    <div class="stat flag-critical">
      <div class="value">{critical_count}</div>
      <div class="label">Critical severity</div>
    </div>
    <div class="stat flag-high">
      <div class="value">{high_count}</div>
      <div class="label">High severity</div>
    </div>
  </div>
</header>

<main>

  <section>
    <h2><span class="section-no">01</span> Traffic &amp; Conflict Overview</h2>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Road users by type</h3>
        <canvas id="trafficChart" height="220"></canvas>
      </div>
      <div class="chart-card">
        <h3>Conflicts by severity</h3>
        <canvas id="severityChart" height="220"></canvas>
      </div>
    </div>
  </section>

  <section>
    <h2><span class="section-no">02</span> Speed Statistics</h2>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>85th-percentile speed by road-user type</h3>
        <canvas id="speedChart" height="220"></canvas>
      </div>
      <div class="chart-card">
        <h3>Speed summary</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Tracks</th><th>Mean</th><th>85th pct.</th><th>Max</th></tr>
          </thead>
          <tbody>
            {speed_rows}
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <h2><span class="section-no">03</span> PET Distribution</h2>
    <div class="chart-card">
      <h3>Post-encroachment time of flagged conflicts</h3>
      <canvas id="petHistogram" height="140"></canvas>
    </div>
  </section>

  <section>
    <h2><span class="section-no">04</span> Flagged Near-Miss Conflicts</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Severity</th><th>PET</th><th>Vehicle</th><th>Vulnerable user</th>
          <th>Vehicle speed</th><th>Vulnerable speed</th><th>Time</th><th>Frame</th>
        </tr>
      </thead>
      <tbody>
        {conflict_rows}
      </tbody>
    </table>
    {calibration_note}
  </section>

  <section>
    <h2><span class="section-no">05</span> Methodology</h2>
    <div class="methodology">
      <p><strong>Post-Encroachment Time (PET)</strong> is the time gap between
      when one road user leaves a shared point on the road and when the
      second road user arrives at that same point. A low PET means two
      paths crossed with little time to spare — a near-miss — even when no
      collision occurred and neither party reacted.</p>
      <p>This report computes PET only for <strong>vehicle ↔ pedestrian/cyclist</strong>
      pairs, since that is the safety-critical category for people outside
      vehicles. Vehicle-vehicle and pedestrian-pedestrian path overlaps are
      not flagged.</p>
      <p>Pixel trajectories are converted to real-world ground-plane meters
      using a 4+ point homography calibration. Speed and PET accuracy
      depend directly on the accuracy of that calibration.</p>
      <p>Severity tiers: <strong>Critical</strong> (PET &lt; 0.5s with high
      closing speed), <strong>High</strong> (PET &lt; 1.0s), <strong>Moderate</strong>
      (PET &lt; {pet_threshold}s), <strong>Low</strong> (above threshold but
      included for context). These tiers are a starting heuristic — adjust
      thresholds in the analysis config to match your site's own safety
      standards if needed.</p>
    </div>
  </section>

</main>

<footer>Generated by the Intersection Safety Monitor pipeline · All faces anonymized prior to analysis</footer>

<script>
  const paper = "#f6f4ee";
  const ink = "#1c1e22";
  const inkSoft = "#5a5d63";
  const gridColor = "rgba(28,30,34,0.08)";

  Chart.defaults.font.family = "'IBM Plex Mono', monospace";
  Chart.defaults.color = inkSoft;

  const trafficData = {traffic_chart_json};
  new Chart(document.getElementById('trafficChart'), {{
    type: 'bar',
    data: {{
      labels: trafficData.labels,
      datasets: [{{
        data: trafficData.data,
        backgroundColor: '#1c1e22',
        borderRadius: 2,
        maxBarThickness: 46,
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ beginAtZero: true, grid: {{ color: gridColor }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  const severityData = {severity_chart_json};
  new Chart(document.getElementById('severityChart'), {{
    type: 'doughnut',
    data: {{
      labels: severityData.labels,
      datasets: [{{
        data: severityData.data,
        backgroundColor: severityData.colors,
        borderWidth: 2,
        borderColor: paper,
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 14 }} }} }}
    }}
  }});

  const speedData = {speed_chart_json};
  new Chart(document.getElementById('speedChart'), {{
    type: 'bar',
    data: {{
      labels: speedData.labels,
      datasets: [
        {{ label: 'Mean (km/h)', data: speedData.mean, backgroundColor: '#c9c4b3' }},
        {{ label: '85th pct. (km/h)', data: speedData.p85, backgroundColor: '#f0a93b' }},
      ]
    }},
    options: {{
      plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }} }},
      scales: {{
        y: {{ beginAtZero: true, grid: {{ color: gridColor }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  const petData = {pet_histogram_json};
  new Chart(document.getElementById('petHistogram'), {{
    type: 'bar',
    data: {{
      labels: petData.labels,
      datasets: [{{
        data: petData.data,
        backgroundColor: '#e23d28',
        borderRadius: 2,
      }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ beginAtZero: true, ticks: {{ precision: 0 }}, grid: {{ color: gridColor }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
</script>

</body>
</html>
"""

# Fill in severity colors used by the CSS template before formatting elsewhere
HTML_TEMPLATE = HTML_TEMPLATE.replace("{critical_color}", SEVERITY_COLOR["critical"])
HTML_TEMPLATE = HTML_TEMPLATE.replace("{high_color}", SEVERITY_COLOR["high"])
HTML_TEMPLATE = HTML_TEMPLATE.replace("{moderate_color}", SEVERITY_COLOR["moderate"])
HTML_TEMPLATE = HTML_TEMPLATE.replace("{low_color}", SEVERITY_COLOR["low"])
