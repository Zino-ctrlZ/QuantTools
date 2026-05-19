"""Option retrieval to PnL orchestration for vectorized backtests.

Runs leg-level option PnL decomposition from retrieval outputs, then aggregates
results to trade, signal, and portfolio levels with a normalized equity curve.

Core Dataclasses:
        OptionVectorizedPnLRunResult: Output payload from runner execution.

Core Functions:
        OptionVectorizedPnLRunner.run: End-to-end retrieval->PnL->aggregation flow.

Processing Flow:
        1. Map selected contracts to optticks.
        2. Run load_option_pnl_data per trade segment.
        3. Remove trade-adjustment effects per configuration.
        4. Aggregate daily attribution at trade/signal/portfolio levels.
        5. Build normalized portfolio equity curve.
"""

from dataclasses import dataclass
from typing import Dict, List, Any

import pandas as pd

from trade.assets.calculate.xmultiply_attr_v2 import load_option_pnl_data
from trade.backtester_.option_vectorized_retrieval import OptionVectorizedRetrievalResult
from trade.helpers.helper import generate_option_tick_new as generate_opttick_new, to_datetime


@dataclass
class OptionVectorizedPnLRunResult:
    """Output container for option retrieval-to-PnL execution."""

    trade_daily: pd.DataFrame
    trade_summary: pd.DataFrame
    signal_daily: pd.DataFrame
    portfolio_daily: pd.DataFrame
    portfolio_equity_curve: pd.DataFrame
    failures: pd.DataFrame


class OptionVectorizedPnLRunner:
    """Runner that computes trade/signal/portfolio PnL from retrieval outputs."""

    def __init__(
        self,
        *,
        starting_nav: float = 1.0,
        fail_fast: bool = False,
        remove_trade_pnl_adjustment: bool = True,
    ) -> None:
        if starting_nav <= 0:
            raise ValueError("starting_nav must be positive.")
        self.starting_nav = float(starting_nav)
        self.fail_fast = fail_fast
        self.remove_trade_pnl_adjustment = remove_trade_pnl_adjustment

    def run(self, retrieval_result: OptionVectorizedRetrievalResult) -> OptionVectorizedPnLRunResult:
        """Execute decomposition and aggregations from retrieval result.

        Args:
            retrieval_result: Output from OptionVectorizedRetriever.run.

        Returns:
            OptionVectorizedPnLRunResult with trade/signal/portfolio tables.
        """
        selected = retrieval_result.selected_contracts.copy()
        if selected.empty:
            empty = pd.DataFrame()
            return OptionVectorizedPnLRunResult(
                trade_daily=empty,
                trade_summary=empty,
                signal_daily=empty,
                portfolio_daily=empty,
                portfolio_equity_curve=empty,
                failures=empty,
            )

        required_cols = {
            "signal_id",
            "ticker",
            "right",
            "strike",
            "expiration",
            "roll_index",
            "segment_start",
            "segment_end",
        }
        missing = required_cols - set(selected.columns)
        if missing:
            raise ValueError(f"selected_contracts missing required columns: {sorted(missing)}")

        selected = selected.sort_values(["signal_id", "roll_index"]).reset_index(drop=True)

        trade_daily_frames: List[pd.DataFrame] = []
        trade_summary_rows: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for row in selected.to_dict(orient="records"):
            signal_id = str(row["signal_id"])
            roll_index = int(row["roll_index"])
            trade_id = f"{signal_id}__roll_{roll_index}"

            try:
                self._validate_segment_presence(retrieval_result, signal_id=signal_id, roll_index=roll_index)
                start_dt = to_datetime(str(row["segment_start"]))
                end_dt = to_datetime(str(row["segment_end"]))
                if end_dt < start_dt:
                    raise ValueError("segment_end is earlier than segment_start")

                opttick = generate_opttick_new(
                    symbol=str(row["ticker"]),
                    right=str(row["right"]),
                    exp=to_datetime(str(row["expiration"])).strftime("%Y-%m-%d"),
                    strike=float(row["strike"]),
                )
                print(f"Processing trade_id={trade_id} with opttick={opttick} for segment {start_dt.date()} to {end_dt.date()}")
                payload = load_option_pnl_data(
                    yesterday=start_dt,
                    today=end_dt,
                    opttick=opttick,
                )
                print(f"  Loaded attribution data with {len(payload.attribution)} rows and columns: {payload.attribution.columns.tolist()}")
                attribution = pd.DataFrame(payload.attribution).copy()
                if attribution.empty:
                    raise ValueError("empty attribution from load_option_pnl_data")

                attribution.index = pd.to_datetime(attribution.index)
                attribution = attribution[(attribution.index >= start_dt) & (attribution.index <= end_dt)]
                if attribution.empty:
                    raise ValueError("attribution empty after segment clipping")

                attribution = self._normalize_trade_adjustment_columns(attribution)

                attribution["signal_id"] = signal_id
                attribution["trade_id"] = trade_id
                attribution["roll_index"] = roll_index
                attribution["ticker"] = str(row["ticker"])
                attribution["opttick"] = opttick
                attribution["segment_start"] = start_dt.strftime("%Y-%m-%d")
                attribution["segment_end"] = end_dt.strftime("%Y-%m-%d")
                attribution["date"] = attribution.index

                trade_daily_frames.append(attribution.reset_index(drop=True))
                trade_summary_rows.append(self._build_trade_summary_row(attribution=attribution))
            except Exception as exc:
                failures.append(
                    {
                        "signal_id": signal_id,
                        "roll_index": roll_index,
                        "trade_id": trade_id,
                        "reason": str(exc),
                    }
                )
                if self.fail_fast:
                    raise

        trade_daily = pd.concat(trade_daily_frames, ignore_index=True) if trade_daily_frames else pd.DataFrame()
        trade_summary = pd.DataFrame(trade_summary_rows) if trade_summary_rows else pd.DataFrame()
        failures_df = pd.DataFrame(failures) if failures else pd.DataFrame()

        signal_daily = self._aggregate_daily(trade_daily=trade_daily, by=["date", "signal_id"])
        portfolio_daily = self._aggregate_daily(trade_daily=trade_daily, by=["date"])
        equity_curve = self._build_normalized_equity_curve(portfolio_daily=portfolio_daily)

        return OptionVectorizedPnLRunResult(
            trade_daily=trade_daily,
            trade_summary=trade_summary,
            signal_daily=signal_daily,
            portfolio_daily=portfolio_daily,
            portfolio_equity_curve=equity_curve,
            failures=failures_df,
        )

    @staticmethod
    def _validate_segment_presence(
        retrieval_result: OptionVectorizedRetrievalResult,
        *,
        signal_id: str,
        roll_index: int,
    ) -> None:
        segments = retrieval_result.signal_ohlc.get(signal_id)
        if segments is None:
            raise ValueError(f"signal_ohlc missing signal_id '{signal_id}'")
        if roll_index < 0 or roll_index >= len(segments):
            raise ValueError(f"signal_ohlc missing segment for signal_id='{signal_id}', roll_index={roll_index}")

    def _normalize_trade_adjustment_columns(self, attribution: pd.DataFrame) -> pd.DataFrame:
        """Remove trade adjustment effects and enforce total_pnl semantics."""
        out = attribution.copy()

        if self.remove_trade_pnl_adjustment and "trade_pnl_adjustment" in out.columns:
            out = out.drop(columns=["trade_pnl_adjustment"])

        if "total_pnl_excl_trade_pnl" in out.columns:
            out["total_pnl"] = out["total_pnl_excl_trade_pnl"]
        elif "total_pnl" not in out.columns:
            numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
            out["total_pnl"] = out[numeric_cols].sum(axis=1) if numeric_cols else 0.0

        return out

    @staticmethod
    def _build_trade_summary_row(attribution: pd.DataFrame) -> Dict[str, Any]:
        row0 = attribution.iloc[0]
        total_col = "total_pnl" if "total_pnl" in attribution.columns else "total_pnl_excl_trade_pnl"
        out: Dict[str, Any] = {
            "signal_id": row0["signal_id"],
            "trade_id": row0["trade_id"],
            "roll_index": int(row0["roll_index"]),
            "ticker": row0["ticker"],
            "opttick": row0["opttick"],
            "segment_start": row0["segment_start"],
            "segment_end": row0["segment_end"],
            "n_days": int(attribution["date"].nunique()),
            "total_pnl": float(attribution[total_col].sum()),
            "explained_pnl": float(attribution["total_pnl_excl_trade_pnl"].sum())
            if "total_pnl_excl_trade_pnl" in attribution.columns
            else float(attribution[total_col].sum()),
            "unexplained_pnl": float(attribution["unexplained_pnl"].sum())
            if "unexplained_pnl" in attribution.columns
            else 0.0,
        }
        return out

    @staticmethod
    def _aggregate_daily(trade_daily: pd.DataFrame, by: List[str]) -> pd.DataFrame:
        if trade_daily.empty:
            return pd.DataFrame()

        ignore_cols = {
            "signal_id",
            "trade_id",
            "ticker",
            "opttick",
            "segment_start",
            "segment_end",
            "date",
        }
        numeric_cols = [
            c for c in trade_daily.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(trade_daily[c])
        ]
        if not numeric_cols:
            grouped = trade_daily[by].drop_duplicates().copy()
            return grouped.sort_values(by).reset_index(drop=True)

        grouped = trade_daily.groupby(by, as_index=False)[numeric_cols].sum()
        return grouped.sort_values(by).reset_index(drop=True)

    def _build_normalized_equity_curve(self, portfolio_daily: pd.DataFrame) -> pd.DataFrame:
        if portfolio_daily.empty:
            return pd.DataFrame()

        date_col = "date"
        if date_col not in portfolio_daily.columns:
            return pd.DataFrame()

        pnl_col = "total_pnl" if "total_pnl" in portfolio_daily.columns else "total_pnl_excl_trade_pnl"
        if pnl_col not in portfolio_daily.columns:
            return pd.DataFrame()

        curve = portfolio_daily[[date_col, pnl_col]].copy()
        curve = curve.sort_values(date_col).reset_index(drop=True)
        curve["cum_pnl"] = curve[pnl_col].cumsum()
        curve["equity"] = self.starting_nav + curve["cum_pnl"]
        curve["normalized_equity"] = curve["equity"] / self.starting_nav
        return curve[[date_col, pnl_col, "equity", "normalized_equity"]]
