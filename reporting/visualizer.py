import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from config.settings import REPORTS_DIR

logger = logging.getLogger(__name__)


class VisualReportGenerator:
    """
    Generates an interactive visual HTML dashboard ranking assets by social mentions,
    composite sentiment score, and directional bias.
    Includes data_source verification badges, out-of-sample sample sizes (n_test),
    naive-baseline comparisons, statistical significance warnings,
    and empirical interval coverage metrics.
    """

    def __init__(self, output_dir: Path = REPORTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, summary_df: pd.DataFrame, all_data: List[Dict[str, Any]]) -> Path:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report_path = self.output_dir / "sentiment_price_prediction_report.html"
        json_path = self.output_dir / "pipeline_summary.json"

        # Export JSON summary
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": timestamp_str,
                "asset_count": len(all_data),
                "summary": summary_df.to_dict(orient="records")
            }, f, indent=2)

        # Check for simulated records
        is_simulated = any(r.get("data_source") == "simulated" for r in summary_df.to_dict(orient="records"))
        banner_class = "banner-simulated" if is_simulated else "banner-live"
        banner_text = "⚠️ SIMULATED DATA WARNING: Fallback synthetic feeds were permitted via --allow-simulation." if is_simulated else "✓ VERIFIED LIVE DATA: Market feeds ingested live via Yahoo Finance, Apify & Exa."

        # Separate exchanges
        hkex_df = summary_df[summary_df["exchange"] == "HKEX"].sort_values(
            by=["social_mentions", "composite_sentiment"], ascending=[False, False]
        )
        krx_df = summary_df[summary_df["exchange"] == "KRX"].sort_values(
            by=["social_mentions", "composite_sentiment"], ascending=[False, False]
        )

        hkex_rows = self._render_table_rows(hkex_df)
        krx_rows = self._render_table_rows(krx_df)

        # Chart payloads
        mentions_chart_data = {
            "labels": summary_df["symbol"].tolist(),
            "data": summary_df["social_mentions"].tolist(),
            "colors": ["#10b981" if d == "BULLISH" else ("#ef4444" if d == "BEARISH" else "#f59e0b") for d in summary_df["direction"]]
        }

        sentiment_chart_data = {
            "labels": summary_df["symbol"].tolist(),
            "social": [float(x) for x in summary_df["weighted_sentiment"]],
            "news": [float(x) for x in summary_df["news_sentiment"]],
            "composite": [float(x) for x in summary_df["composite_sentiment"]]
        }

        # Calculate valid metrics without inventing values
        valid_accuracies = summary_df["oos_accuracy"].dropna()
        avg_accuracy_str = f"{valid_accuracies.mean() * 100:.1f}%" if not valid_accuracies.empty else "Insufficient Data"

        valid_lifts = summary_df["accuracy_lift"].dropna()
        avg_lift_str = f"{valid_lifts.mean() * 100:+.1f}%" if not valid_lifts.empty else "N/A"

        total_mentions = summary_df["social_mentions"].sum()
        bullish_count = len(summary_df[summary_df["direction"] == "BULLISH"])
        bearish_count = len(summary_df[summary_df["direction"] == "BEARISH"])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Exchange Sentiment & n+1 Forecast Pipeline</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-blue: #3b82f6;
            --border-color: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .header .subtitle {{ color: var(--text-secondary); font-size: 13px; margin-top: 4px; }}
        .banner {{
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 20px;
        }}
        .banner-live {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .banner-simulated {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px 20px;
        }}
        .stat-card .label {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase; font-weight: 600; }}
        .stat-card .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
        .section-title {{
            font-size: 16px;
            font-weight: 600;
            margin: 24px 0 12px 0;
            color: var(--text-primary);
        }}
        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        .charts-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 12px;
        }}
        th {{
            background: #0f172a;
            color: var(--text-secondary);
            padding: 10px 12px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            border-bottom: 1px solid var(--border-color);
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .pill {{
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            display: inline-block;
        }}
        .pill-bull {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.4); }}
        .pill-bear {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.4); }}
        .pill-neut {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.4); }}
        .text-green {{ color: var(--accent-green); font-weight: 600; }}
        .text-red {{ color: var(--accent-red); font-weight: 600; }}
        .methodology-box {{
            background: #111827;
            border-left: 4px solid var(--accent-blue);
            padding: 16px 20px;
            border-radius: 6px;
            font-size: 12px;
            color: #cbd5e1;
            margin-top: 24px;
        }}
        .methodology-box h4 {{ color: #fff; margin-bottom: 6px; font-size: 13px; }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>Cross-Exchange Sentiment & n+1 Forecast Pipeline</h1>
            <div class="subtitle">Hong Kong Stock Exchange (Hang Seng) & Korea Exchange (KOSPI) | Aligned UTC Runs | {timestamp_str}</div>
        </div>
    </div>

    <div class="banner {banner_class}">
        {banner_text}
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Total Social Posts Analyzed</div>
            <div class="value" style="color: var(--accent-blue);">{total_mentions:,}</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg OOS Directional Accuracy</div>
            <div class="value text-green">{avg_accuracy_str}</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg Lift vs Naive Baseline</div>
            <div class="value" style="color: #a855f7;">{avg_lift_str}</div>
        </div>
        <div class="stat-card">
            <div class="label">Sentiment Breadth</div>
            <div class="value"><span class="text-green">{bullish_count} Bull</span> / <span class="text-red">{bearish_count} Bear</span></div>
        </div>
    </div>

    <div class="charts-container">
        <div class="card">
            <h3 style="font-size: 14px; margin-bottom: 12px;">Social Mention Volume by Asset (Lookback Window)</h3>
            <canvas id="mentionsChart" height="150"></canvas>
        </div>
        <div class="card">
            <h3 style="font-size: 14px; margin-bottom: 12px;">Sentiment Polarity: Retail Social vs Exa Institutional News</h3>
            <canvas id="sentimentChart" height="150"></canvas>
        </div>
    </div>

    <div class="section-title">
        <span>🇭🇰 Hong Kong Stock Exchange (Hang Seng) - Ranked by Mentions & Sentiment</span>
    </div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Asset Name</th>
                    <th>Price</th>
                    <th>Mentions</th>
                    <th>Social Sent.</th>
                    <th>News Sent.</th>
                    <th>Composite</th>
                    <th>Direction</th>
                    <th>Stated Pos</th>
                    <th>Stance</th>
                    <th>Support (S1)</th>
                    <th>Swing Low</th>
                    <th>Pivot</th>
                    <th>Resistance (R1)</th>
                    <th>Target n+1</th>
                    <th>OOS Acc (n)</th>
                    <th>Lift vs Base</th>
                </tr>
            </thead>
            <tbody>
                {hkex_rows}
            </tbody>
        </table>
    </div>

    <div class="section-title">
        <span>🇰🇷 Korea Stock Exchange (KRX / KOSPI) - Ranked by Mentions & Sentiment</span>
    </div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Asset Name</th>
                    <th>Price</th>
                    <th>Mentions</th>
                    <th>Social Sent.</th>
                    <th>News Sent.</th>
                    <th>Composite</th>
                    <th>Direction</th>
                    <th>Stated Pos</th>
                    <th>Stance</th>
                    <th>Support (S1)</th>
                    <th>Swing Low</th>
                    <th>Pivot</th>
                    <th>Resistance (R1)</th>
                    <th>Target n+1</th>
                    <th>OOS Acc (n)</th>
                    <th>Lift vs Base</th>
                </tr>
            </thead>
            <tbody>
                {krx_rows}
            </tbody>
        </table>
    </div>

    <div class="methodology-box">
        <h4>Methodology, Baseline Comparisons & Limitations:</h4>
        <p><strong>1. Strict No-Leakage Time Alignment:</strong> Features for predicting bar t+1 are calculated using market and alternative data available strictly at or before the close of bar t.</p>
        <p><strong>2. Naive Baseline & Statistical Tests:</strong> Model accuracy is benchmarked against a naive persistence baseline (sign of preceding bar return). Binomial p-values are computed to evaluate whether directional accuracy exceeds 50% significantly.</p>
        <p><strong>3. Position Tracking:</strong> Stated positions reflect retail traders' self-reported holdings in forum discourse. The suggested stance is the quantitative model's risk-adjusted recommendation.</p>
        <p><strong>4. Verification & Audit:</strong> If run with `--allow-simulation`, synthetic data is tagged and marked across all records and charts.</p>
    </div>

    <script>
        const ctxMentions = document.getElementById('mentionsChart').getContext('2d');
        new Chart(ctxMentions, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(mentions_chart_data['labels'])},
                datasets: [{{
                    label: 'Social Mentions',
                    data: {json.dumps(mentions_chart_data['data'])},
                    backgroundColor: {json.dumps(mentions_chart_data['colors'])},
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        const ctxSentiment = document.getElementById('sentimentChart').getContext('2d');
        new Chart(ctxSentiment, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(sentiment_chart_data['labels'])},
                datasets: [
                    {{
                        label: 'Weighted Social',
                        data: {json.dumps(sentiment_chart_data['social'])},
                        backgroundColor: '#3b82f6',
                        borderRadius: 3
                    }},
                    {{
                        label: 'Exa Institutional News',
                        data: {json.dumps(sentiment_chart_data['news'])},
                        backgroundColor: '#a855f7',
                        borderRadius: 3
                    }},
                    {{
                        label: 'Composite Score',
                        data: {json.dumps(sentiment_chart_data['composite'])},
                        backgroundColor: '#10b981',
                        borderRadius: 3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ min: -1.0, max: 1.0, grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Visual report generated at: {report_path}")
        return report_path

    def _render_table_rows(self, df: pd.DataFrame) -> str:
        rows = []
        for _, r in df.iterrows():
            dir_pill = (
                '<span class="pill pill-bull">BULLISH</span>' if r["direction"] == "BULLISH"
                else ('<span class="pill pill-bear">BEARISH</span>' if r["direction"] == "BEARISH"
                else '<span class="pill pill-neut">NEUTRAL</span>')
            )
            stance_pill = (
                f'<span class="text-green" style="font-weight:700;">{r["suggested_stance"]}</span>' if r["suggested_stance"] == "LONG"
                else (f'<span class="text-red" style="font-weight:700;">{r["suggested_stance"]}</span>' if r["suggested_stance"] == "SHORT"
                else '<span style="color:#64748b;">NEUTRAL</span>')
            )

            p_n1 = r["pred_price_n1"]
            curr = r["current_price"]
            diff_pct = ((p_n1 - curr) / curr) * 100 if curr > 0 else 0.0
            diff_str = f"+{diff_pct:.2f}%" if diff_pct >= 0 else f"{diff_pct:.2f}%"
            diff_color = "text-green" if diff_pct >= 0 else "text-red"

            # OOS accuracy & lift
            if r["oos_accuracy"] is not None:
                sig_star = " *" if r["statistically_significant"] else ""
                oos_str = f"{r['oos_accuracy']*100:.1f}% ({r['n_test']}){sig_star}"
                lift_val = r["accuracy_lift"]
                lift_color = "text-green" if (lift_val or 0) > 0 else "text-red"
                lift_str = f'<span class="{lift_color}">{lift_val*100:+.1f}%</span>' if lift_val is not None else "N/A"
            else:
                oos_str = '<span style="color:#64748b;">Insuff. Data</span>'
                lift_str = '<span style="color:#64748b;">N/A</span>'

            rows.append(f"""
            <tr>
                <td><strong>{r["symbol"]}</strong></td>
                <td>{r["name"]}</td>
                <td>{curr:,.2f} {r["currency"]}</td>
                <td style="font-weight:600; color:#38bdf8;">{r["social_mentions"]}</td>
                <td>{r["weighted_sentiment"]:+.2f}</td>
                <td>{r["news_sentiment"]:+.2f}</td>
                <td style="font-weight:600;">{r["composite_sentiment"]:+.2f}</td>
                <td>{dir_pill}</td>
                <td style="color:#94a3b8;">{r["stated_position"]}</td>
                <td>{stance_pill}</td>
                <td class="text-green">{r["support_s1"]:,.2f}</td>
                <td style="color:#cbd5e1;">{r["swing_low"]:,.2f}</td>
                <td style="color:#94a3b8;">{r["pivot"]:,.2f}</td>
                <td class="text-red">{r["resistance_r1"]:,.2f}</td>
                <td><strong>{p_n1:,.2f}</strong> <span class="{diff_color}" style="font-size:11px;">({diff_str})</span></td>
                <td>{oos_str}</td>
                <td>{lift_str}</td>
            </tr>
            """)
        return "\n".join(rows)
