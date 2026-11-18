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
    Includes stated vs suggested stances, post-mentioned levels,
    sample sizes (n_test), and naive-baseline comparisons.
    """

    def build_summary_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        records = []
        for r in results:
            pos = r.get("recommendation", {})
            levels = pos.get("levels", {})
            pred = r.get("prediction", {})
            sent = r.get("sentiment", {})
            oos = pred.get("oos_validation", {}) or {}

            # Parse out-of-sample metrics safely without inventing numbers
            status = oos.get("status", "unknown")
            n_test = oos.get("n_test", 0)
            oos_acc = oos.get("directional_accuracy")
            base_acc = oos.get("naive_baseline_accuracy")
            lift = oos.get("accuracy_lift_vs_baseline")
            p_val = oos.get("p_value")
            is_sig = oos.get("statistically_significant", False)
            sharpe = oos.get("sharpe_ratio")

            records.append({
                "exchange": r.get("exchange"),
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "current_price": r.get("current_price"),
                "currency": r.get("currency"),
                "data_source": r.get("data_source", "live"),
                "social_mentions": sent.get("social_mentions", 0),
                "weighted_sentiment": sent.get("weighted_social_sentiment", 0.0),
                "composite_sentiment": sent.get("composite_sentiment", 0.0),
                "news_sentiment": sent.get("news_sentiment", 0.0),
                "direction": sent.get("sentiment_direction", "NEUTRAL"),
                "rs_divergence": sent.get("retail_smart_divergence", 0.0),
                "stated_position": pos.get("stated_position", "NONE"),
                "suggested_stance": pos.get("suggested_stance", "NEUTRAL"),
                "position": pos.get("suggested_stance", "NEUTRAL"),
                "confidence": pos.get("confidence", "LOW"),
                "support_s1": levels.get("s1", 0.0),
                "pivot": levels.get("pivot", 0.0),
                "resistance_r1": levels.get("r1", 0.0),
                "swing_high": levels.get("swing_high", 0.0),
                "swing_low": levels.get("swing_low", 0.0),
                "post_mentioned_levels": levels.get("post_mentioned_levels", []),
                "pred_price_n1": pred.get("predicted_price_n1", r.get("current_price")),
                "pred_return_n1": pred.get("predicted_return_n1", 0.0),
                "oos_status": status,
                "n_test": n_test,
                "oos_accuracy": oos_acc,
                "baseline_accuracy": base_acc,
                "accuracy_lift": lift,
                "p_value": p_val,
                "statistically_significant": is_sig,
                "sharpe": sharpe,
                "risk_reward": pos.get("risk_reward_ratio", 1.0),
                "rationale": pos.get("rationale", "")
            })

        df = pd.DataFrame(records)
        return df

    def display_rich_tables(self, df: pd.DataFrame):
        for exchange, group in df.groupby("exchange"):
            sorted_by_mentions = group.sort_values(by=["social_mentions", "composite_sentiment"], ascending=[False, False])

            # Check if any asset in exchange used simulation
            has_sim = any(row["data_source"] == "simulated" for _, row in sorted_by_mentions.iterrows())
            banner = "[bold red]⚠️ SIMULATED DATA WARNING[/bold red]" if has_sim else "[bold green]✓ LIVE MARKET DATA[/bold green]"

            table = Table(
                title=f"=== {exchange} Market Intelligence & Sentiment Summary ({banner}) ===",
                box=box.ROUNDED,
                header_style="bold cyan"
            )

            table.add_column("Ticker", style="bold white", width=10)
            table.add_column("Asset Name", width=20)
            table.add_column("Price", justify="right", width=9)
            table.add_column("Mentions", justify="right", style="cyan", width=8)
            table.add_column("Sentiment", justify="right", width=9)
            table.add_column("Direction", justify="center", width=9)
            table.add_column("Stated Pos", justify="center", style="dim", width=10)
            table.add_column("Stance", justify="center", width=8)
            table.add_column("S1", justify="right", style="green", width=9)
            table.add_column("R1", justify="right", style="red", width=9)
            table.add_column("Target n+1", justify="right", style="bold yellow", width=10)
            table.add_column("OOS Acc (n)", justify="right", style="blue", width=13)
            table.add_column("Lift vs Base", justify="right", width=11)

            for _, row in sorted_by_mentions.iterrows():
                dir_color = "green" if row["direction"] == "BULLISH" else ("red" if row["direction"] == "BEARISH" else "yellow")
                dir_str = f"[{dir_color}]{row['direction']}[/{dir_color}]"

                stance_color = "bold green" if row["suggested_stance"] == "LONG" else ("bold red" if row["suggested_stance"] == "SHORT" else "dim")
                stance_str = f"[{stance_color}]{row['suggested_stance']}[/{stance_color}]"

                # Stated retail position
                stated_str = f"{row['stated_position']}"

                # OOS Accuracy & lift
                if row["oos_accuracy"] is not None:
                    sig_mark = "*" if row["statistically_significant"] else ""
                    acc_str = f"{row['oos_accuracy']*100:.1f}% ({row['n_test']}){sig_mark}"
                    lift_val = row["accuracy_lift"]
                    lift_str = f"{lift_val*100:+.1f}%" if lift_val is not None else "N/A"
                    lift_color = "green" if (lift_val or 0) > 0 else "red"
                    lift_styled = f"[{lift_color}]{lift_str}[/{lift_color}]"
                else:
                    acc_str = "[dim]Insuff. Data[/dim]"
                    lift_styled = "[dim]N/A[/dim]"

                table.add_row(
                    str(row["symbol"]),
                    str(row["name"])[:18],
                    f"{row['current_price']:.2f}",
                    str(row["social_mentions"]),
                    f"{row['composite_sentiment']:+.2f}",
                    dir_str,
                    stated_str,
                    stance_str,
                    f"{row['support_s1']:.2f}",
                    f"{row['resistance_r1']:.2f}",
                    f"{row['pred_price_n1']:.2f}",
                    acc_str,
                    lift_styled
                )

            console.print("\n")
            console.print(table)
            console.print("[dim]* Indicates statistically significant directional edge at p < 0.05 (Binomial Test)[/dim]")
