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
    Generates a publication-grade interactive visual HTML dashboard
    ranking assets by social mentions, sentiment score, and directional bias,
    complete with technical support/resistance levels and n+1 predictive forecasts.
    """

    def __init__(self, output_dir: Path = REPORTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, summary_df: pd.DataFrame, all_data: List[Dict[str, Any]]) -> Path:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report_path = self.output_dir / "sentiment_price_prediction_report.html"
        json_path = self.output_dir / "pipeline_summary.json"

        # Export JSON summary for programmatic integration
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": timestamp_str,
                "asset_count": len(all_data),
                "summary": summary_df.to_dict(orient="records")
            }, f, indent=2)

        # Build Exchange Tables HTML
        hkex_df = summary_df[summary_df["exchange"] == "HKEX"].sort_values(
            by=["social_mentions", "composite_sentiment"], ascending=[False, False]
        )
        krx_df = summary_df[summary_df["exchange"] == "KRX"].sort_values(
            by=["social_mentions", "composite_sentiment"], ascending=[False, False]
        )

        hkex_rows = self._render_table_rows(hkex_df)
        krx_rows = self._render_table_rows(krx_df)

        # Chart datasets (JSON serialized for Chart.js)
        mentions_chart_data = {
            "labels": summary_df["symbol"].tolist(),
            "data": summary_df["social_mentions"].tolist(),
            "colors": ["#10b981" if d == "BULLISH" else ("#ef4444" if d == "BEARISH" else "#f59e0b") for d in summary_df["direction"]]
        }

        sentiment_chart_data = {
            "labels": summary_df["symbol"].tolist(),
            "social": summary_df["weighted_sentiment"].tolist(),
            "news": summary_df["news_sentiment"].tolist(),
            "composite": summary_df["composite_sentiment"].tolist()
        }

        # Out-of-sample validation averages
        avg_accuracy = summary_df["oos_accuracy"].mean() * 100
        avg_sharpe = summary_df["sharpe"].mean()
        total_mentions = summary_df["social_mentions"].sum()
        bullish_count = len(summary_df[summary_df["direction"] == "BULLISH"])
        bearish_count = len(summary_df[summary_df["direction"] == "BEARISH"])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Source Sentiment & n+1 Price Prediction Pipeline</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
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
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .header h1 {{ font-size: 26px; font-weight: 700; color: #fff; }}
        .header .subtitle {{ color: var(--text-secondary); font-size: 14px; margin-top: 4px; }}
        .badge-live {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
            padding: 6px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}
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
        .stat-card .label {{ color: var(--text-secondary); font-size: 13px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
        .stat-card .value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin: 28px 0 14px 0;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
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
            font-size: 13px;
        }}
        th {{
            background: #0f172a;
            color: var(--text-secondary);
            padding: 12px 14px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .pill {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }}
        .pill-bull {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.4); }}
        .pill-bear {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.4); }}
        .pill-neut {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.4); }}
        .text-green {{ color: var(--accent-green); font-weight: 600; }}
        .text-red {{ color: var(--accent-red); font-weight: 600; }}
        .text-yellow {{ color: var(--accent-amber); font-weight: 600; }}
        .methodology-box {{
            background: #111827;
            border-left: 4px solid var(--accent-blue);
            padding: 16px 20px;
            border-radius: 6px;
            font-size: 13px;
            color: #cbd5e1;
            margin-top: 24px;
        }}
        .methodology-box h4 {{ color: #fff; margin-bottom: 6px; font-size: 14px; }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>Cross-Exchange Sentiment & n+1 Price Forecast System</h1>
            <div class="subtitle">Multi-Source Alternative Data Pipeline (Apify Scrapers & Exa News Ingestion) | Generated: {timestamp_str}</div>
        </div>
        <div class="badge-live">UTC ALIGNED PIPELINE</div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Total Social Posts Scraped</div>
            <div class="value" style="color: var(--accent-blue);">{total_mentions:,}</div>
        </div>
        <div class="stat-card">
            <div class="label">OOS Directional Accuracy</div>
            <div class="value text-green">{avg_accuracy:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="label">Walk-Forward Sharpe Ratio</div>
            <div class="value" style="color: #a855f7;">{avg_sharpe:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Market Sentiment Breadth</div>
            <div class="value"><span class="text-green">{bullish_count} Bull</span> / <span class="text-red">{bearish_count} Bear</span></div>
        </div>
    </div>

    <div class="charts-container">
        <div class="card">
            <h3 style="font-size: 15px; margin-bottom: 12px;">Social Discussion Volume by Asset (10-Day Window)</h3>
            <canvas id="mentionsChart" height="150"></canvas>
        </div>
        <div class="card">
            <h3 style="font-size: 15px; margin-bottom: 12px;">Sentiment Polarity: Retail Social vs. Exa Institutional News</h3>
            <canvas id="sentimentChart" height="150"></canvas>
        </div>
    </div>

    <div class="section-title">
        <span>🇭🇰 Hong Kong Stock Exchange (Hang Seng) - Ranked by Social Mentions & Sentiment</span>
    </div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Asset Name</th>
                    <th>Current Price</th>
                    <th>Mentions</th>
                    <th>Social Sent.</th>
                    <th>News Sent.</th>
                    <th>Composite</th>
                    <th>Direction</th>
                    <th>Position</th>
                    <th>Support (S1)</th>
                    <th>Pivot</th>
                    <th>Resistance (R1)</th>
                    <th>Predicted (n+1)</th>
                    <th>OOS Hit%</th>
                </tr>
            </thead>
            <tbody>
                {hkex_rows}
            </tbody>
        </table>
    </div>

    <div class="section-title">
        <span>🇰🇷 Korea Stock Exchange (KRX / KOSPI) - Ranked by Social Mentions & Sentiment</span>
    </div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Asset Name</th>
                    <th>Current Price</th>
                    <th>Mentions</th>
                    <th>Social Sent.</th>
                    <th>News Sent.</th>
                    <th>Composite</th>
                    <th>Direction</th>
                    <th>Position</th>
                    <th>Support (S1)</th>
                    <th>Pivot</th>
                    <th>Resistance (R1)</th>
                    <th>Predicted (n+1)</th>
                    <th>OOS Hit%</th>
                </tr>
            </thead>
            <tbody>
                {krx_rows}
            </tbody>
        </table>
    </div>

    <div class="methodology-box">
        <h4>Methodology & Quantitative Model Architecture:</h4>
        <p><strong>1. Data Ingestion:</strong> Multi-source ingestion capturing retail community forums (LIHKG, Futu/Moomoo, Naver Finance, Daum Finance) via Apify actors and institutional news via Exa.com neural search across top financial outlets.</p>
        <p><strong>2. Engagement-Weighted Sentiment Index:</strong> Weights posts by author reach (logarithmic follower scale), virality (likes, reposts, comments), and exponential time decay (half-life of 3 days).</p>
        <p><strong>3. Retail vs Institutional Divergence (RSMD):</strong> Computes sentiment delta between retail crowds and institutional coverage to identify potential retail trap distributions near resistance and capitulation accumulation near support.</p>
        <p><strong>4. Out-of-Sample Walk-Forward Validation:</strong> Machine learning forecasts (blended XGBoost + L2 Ridge Regression) are evaluated strictly on out-of-sample forward folds with zero lookahead bias.</p>
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
            pos_pill = (
                f'<span class="text-green" style="font-weight:700;">{r["position"]}</span>' if r["position"] == "LONG"
                else (f'<span class="text-red" style="font-weight:700;">{r["position"]}</span>' if r["position"] == "SHORT"
                else '<span style="color:#64748b;">NEUTRAL</span>')
            )

            p_n1 = r["pred_price_n1"]
            curr = r["current_price"]
            diff_pct = ((p_n1 - curr) / curr) * 100
            diff_str = f"+{diff_pct:.2f}%" if diff_pct >= 0 else f"{diff_pct:.2f}%"
            diff_color = "text-green" if diff_pct >= 0 else "text-red"

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
                <td>{pos_pill}</td>
                <td class="text-green">{r["support_s1"]:,.2f}</td>
                <td style="color:#94a3b8;">{r["pivot"]:,.2f}</td>
                <td class="text-red">{r["resistance_r1"]:,.2f}</td>
                <td><strong>{p_n1:,.2f}</strong> <span class="{diff_color}" style="font-size:11px;">({diff_str})</span></td>
                <td>{r["oos_accuracy"]*100:.1f}%</td>
            </tr>
            """)
        return "\n".join(rows)
