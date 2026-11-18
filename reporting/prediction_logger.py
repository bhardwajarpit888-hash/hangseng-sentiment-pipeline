import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box
import yfinance as yf

from config.settings import REPORTS_DIR

logger = logging.getLogger(__name__)
console = Console()


class PredictionEvaluator:
    """
    Logs n+1 price predictions to Parquet and performs out-of-sample realization checks
    against live market closes once the n+1 bar is realized.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (REPORTS_DIR / "predictions_log.parquet")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_predictions(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        records = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for r in results:
            pred = r.get("prediction", {})
            recom = r.get("recommendation", {})
            interval = pred.get("prediction_interval", {})

            records.append({
                "timestamp_utc": now_utc,
                "exchange": r.get("exchange"),
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "current_price": r.get("current_price"),
                "predicted_price_n1": pred.get("predicted_price_n1"),
                "predicted_return_n1": pred.get("predicted_return_n1"),
                "predicted_direction": pred.get("predicted_direction_n1"),
                "suggested_stance": recom.get("suggested_stance"),
                "stated_position": recom.get("stated_position"),
                "lower_bound": interval.get("lower_bound"),
                "upper_bound": interval.get("upper_bound"),
                "realized_close": None,
                "realized_return": None,
                "direction_hit": None
            })

        new_df = pd.DataFrame(records)
        if self.log_path.exists():
            existing_df = pd.read_parquet(self.log_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Deduplicate by symbol + timestamp
            combined_df = combined_df.drop_duplicates(subset=["symbol", "timestamp_utc"], keep="last")
        else:
            combined_df = new_df

        combined_df.to_parquet(self.log_path)
        logger.info(f"Logged {len(new_df)} predictions to {self.log_path}")
        return combined_df

    def evaluate_realized_predictions(self) -> pd.DataFrame:
        """
        Loads predictions_log.parquet, checks current market prices,
        evaluates directional hit rate and forecast error on realized bars.
        """
        if not self.log_path.exists():
            console.print("[yellow]No predictions log found at reports/predictions_log.parquet[/yellow]")
            return pd.DataFrame()

        df = pd.read_parquet(self.log_path)
        if df.empty:
            console.print("[yellow]Predictions log is empty.[/yellow]")
            return df

        console.print(f"\n[bold cyan]=== Evaluating Realized Forecasts (Total Logged: {len(df)}) ===[/bold cyan]\n")

        # Fetch latest prices for distinct symbols
        symbols = df["symbol"].unique()
        latest_prices = {}
        for s in symbols:
            try:
                t = yf.Ticker(s)
                hist = t.history(period="5d", interval="1h")
                if not hist.empty:
                    latest_prices[s] = float(hist["Close"].dropna().iloc[-1])
            except Exception as e:
                logger.debug(f"Could not fetch realization for {s}: {e}")

        # Update realization metrics
        for idx, row in df.iterrows():
            sym = row["symbol"]
            if sym in latest_prices and pd.isna(row.get("realized_close")):
                realized = latest_prices[sym]
                curr = row["current_price"]
                actual_ret = (realized - curr) / curr if curr > 0 else 0.0
                pred_ret = row["predicted_return_n1"]

                # Direction hit
                hit = bool((pred_ret > 0 and actual_ret > 0) or (pred_ret <= 0 and actual_ret <= 0))

                df.at[idx, "realized_close"] = realized
                df.at[idx, "realized_return"] = actual_ret
                df.at[idx, "direction_hit"] = hit

        df.to_parquet(self.log_path)

        # Display rich evaluation table
        table = Table(
            title="Prediction Realization vs Actual Out-of-Sample Market Closes",
            box=box.ROUNDED,
            header_style="bold magenta"
        )
        table.add_column("Timestamp", style="dim", width=18)
        table.add_column("Symbol", style="bold white", width=10)
        table.add_column("Entry Price", justify="right", width=11)
        table.add_column("Pred (n+1)", justify="right", style="cyan", width=11)
        table.add_column("Realized", justify="right", style="bold", width=11)
        table.add_column("Pred %", justify="right", width=9)
        table.add_column("Real %", justify="right", width=9)
        table.add_column("Hit?", justify="center", width=8)

        realized_hits = []
        for _, r in df.tail(15).iterrows():
            hit_str = "[green]YES[/green]" if r["direction_hit"] is True else ("[red]NO[/red]" if r["direction_hit"] is False else "[dim]PENDING[/dim]")
            if r["direction_hit"] is not None and not pd.isna(r["direction_hit"]):
                realized_hits.append(int(r["direction_hit"]))

            real_str = f"{r['realized_close']:,.2f}" if pd.notna(r["realized_close"]) else "PENDING"
            real_ret_str = f"{r['realized_return']*100:+.2f}%" if pd.notna(r["realized_return"]) else "-"

            table.add_row(
                str(r["timestamp_utc"])[:16],
                str(r["symbol"]),
                f"{r['current_price']:,.2f}",
                f"{r['predicted_price_n1']:,.2f}",
                real_str,
                f"{r['predicted_return_n1']*100:+.2f}%",
                real_ret_str,
                hit_str
            )

        console.print(table)

        if realized_hits:
            hit_rate = float(np.mean(realized_hits) * 100)
            console.print(f"\n[bold green]Realized Directional Hit Rate: {hit_rate:.1f}% ({sum(realized_hits)}/{len(realized_hits)})[/bold green]\n")
        else:
            console.print("\n[yellow]Market bars for logged timestamps are still active / pending realization.[/yellow]\n")

        return df
