from typing import List, Dict, Any
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


class SummaryTableGenerator:
    """
    Constructs sorted tabular views of ticker assets per exchange
    ranked by social mentions and composite sentiment score.
    """

    def build_summary_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        records = []
        for r in results:
            pos = r.get("recommendation", {})
            levels = pos.get("levels", {})
            pred = r.get("prediction", {})
            sent = r.get("sentiment", {})

            records.append({
                "exchange": r["exchange"],
                "symbol": r["symbol"],
                "name": r["name"],
                "current_price": r["current_price"],
                "currency": r["currency"],
                "social_mentions": sent.get("social_mentions", 0),
                "weighted_sentiment": sent.get("weighted_social_sentiment", 0.0),
                "composite_sentiment": sent.get("composite_sentiment", 0.0),
                "news_sentiment": sent.get("news_sentiment", 0.0),
                "direction": sent.get("sentiment_direction", "NEUTRAL"),
                "rs_divergence": sent.get("retail_smart_divergence", 0.0),
                "position": pos.get("position", "NEUTRAL"),
                "confidence": pos.get("confidence", "LOW"),
                "support_s1": levels.get("s1", 0.0),
                "pivot": levels.get("pivot", 0.0),
                "resistance_r1": levels.get("r1", 0.0),
                "pred_price_n1": pred.get("predicted_price_n1", r["current_price"]),
                "pred_return_n1": pred.get("predicted_return_n1", 0.0),
                "oos_accuracy": pred.get("oos_validation", {}).get("directional_accuracy", 0.0),
                "sharpe": pred.get("oos_validation", {}).get("sharpe_ratio", 0.0),
                "risk_reward": pos.get("risk_reward_ratio", 1.0),
                "rationale": pos.get("rationale", "")
            })

        df = pd.DataFrame(records)
        return df

    def display_rich_tables(self, df: pd.DataFrame):
        for exchange, group in df.groupby("exchange"):
            sorted_by_mentions = group.sort_values(by=["social_mentions", "composite_sentiment"], ascending=[False, False])

            table = Table(
                title=f"=== {exchange} Market Intelligence & Sentiment Summary (Sorted by Mentions) ===",
                box=box.ROUNDED,
                header_style="bold cyan"
            )

            table.add_column("Ticker", style="bold white", width=10)
            table.add_column("Asset Name", width=22)
            table.add_column("Price", justify="right", width=10)
            table.add_column("Mentions", justify="right", style="magenta", width=9)
            table.add_column("Sentiment", justify="right", width=10)
            table.add_column("Direction", justify="center", width=10)
            table.add_column("Position", justify="center", width=9)
            table.add_column("S1 (Support)", justify="right", style="green", width=11)
            table.add_column("R1 (Resist)", justify="right", style="red", width=11)
            table.add_column("Target n+1", justify="right", style="bold yellow", width=11)
            table.add_column("OOS Acc", justify="right", style="blue", width=8)

            for _, row in sorted_by_mentions.iterrows():
                # Direction styling
                dir_color = "green" if row["direction"] == "BULLISH" else ("red" if row["direction"] == "BEARISH" else "yellow")
                dir_str = f"[{dir_color}]{row['direction']}[/{dir_color}]"

                pos_color = "bold green" if row["position"] == "LONG" else ("bold red" if row["position"] == "SHORT" else "dim")
                pos_str = f"[{pos_color}]{row['position']}[/{pos_color}]"

                table.add_row(
                    row["symbol"],
                    row["name"][:20],
                    f"{row['current_price']:.2f}",
                    str(row["social_mentions"]),
                    f"{row['composite_sentiment']:+.2f}",
                    dir_str,
                    pos_str,
                    f"{row['support_s1']:.2f}",
                    f"{row['resistance_r1']:.2f}",
                    f"{row['pred_price_n1']:.2f}",
                    f"{row['oos_accuracy']*100:.1f}%"
                )

            console.print("\n")
            console.print(table)
