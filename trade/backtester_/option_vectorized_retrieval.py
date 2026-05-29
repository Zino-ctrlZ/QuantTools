"""Vectorized option retrieval workflows for backtest signal windows.

Provides a retrieval-only API that maps directional signal windows to nearest
option contracts using configurable DTE/moneyness targets, then loads full
contract OHLC time series for each matched signal.

Core Dataclasses:
        OptionRetrievalSignal: Canonical input signal for option selection.
        SelectedOptionContract: Contract metadata selected for one signal.
        UnmatchedSignal: Structured reason when no contract is selected.
        OptionVectorizedRetrievalResult: End-to-end retrieval payload.

Core Functions:
        OptionVectorizedRetriever.run: Execute full retrieval pipeline.
        OptionVectorizedRetriever.from_multi_asset_strategy: Build signals from
            MultiAssetStrategy.simulate_all() output.

Processing Flow:
        1. Normalize source inputs into canonical signal rows.
        2. Query option chains with retrieve_chain_bulk on signal start dates.
        3. Rank candidates by weighted DTE/moneyness distance.
        4. Retrieve selected contract OHLC via retrieve_eod_ohlc.
        5. Return matched/unmatched diagnostics and per-signal series.

Risk/Assumptions:
        - This module does not compute PnL or execution slippage.
        - Explicit signal right takes precedence over side-derived right.

Usage:
        >>> retriever = OptionVectorizedRetriever(target_dte=30, target_moneyness=1.0, right="C")
        >>> result = retriever.run(signals=[
        ...     OptionRetrievalSignal(
        ...         ticker="AAPL",
        ...         start_date="2026-01-02",
        ...         end_date="2026-01-30",
        ...         side=1,
        ...         quantity=1.0,
        ...         signal_id="sig-1",
        ...     )
        ... ])
        >>> len(result.selected_contracts) >= 0
        True
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any, Tuple

import pandas as pd

from dbase.DataAPI.ThetaData import retrieve_chain_bulk, retrieve_eod_ohlc
from trade.backtester_._multi_asset_strategy import MultiAssetStrategy
from trade.datamanager.market_data import get_timeseries_obj
from trade.helpers.helper import to_datetime


DateLike = Union[datetime, str]


@dataclass
class OptionRetrievalSignal:
    """Canonical input signal for option retrieval.

    Args:
        ticker: Underlier symbol.
        start_date: Signal entry date (inclusive).
        end_date: Signal exit date (inclusive).
        side: Direction (+1 long, -1 short).
        quantity: Signed or absolute desired quantity.
        signal_id: Stable signal identifier.
        right: Optional explicit option right ("C"/"P"). If provided, wins.
        target_dte: Optional per-signal DTE override.
        target_moneyness: Optional per-signal moneyness override.
        metadata: Free-form per-signal metadata.
    """

    ticker: str
    start_date: DateLike
    end_date: DateLike
    side: int
    quantity: float
    signal_id: str
    right: Optional[str] = None
    target_dte: Optional[int] = None
    target_moneyness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedOptionContract:
    """Selected contract metadata for one signal."""

    signal_id: str
    ticker: str
    right: str
    strike: float
    expiration: str
    chain_date: str
    dte: int
    moneyness: Optional[float]
    score: float
    roll_index: int = 0
    segment_start: Optional[str] = None
    segment_end: Optional[str] = None


@dataclass
class UnmatchedSignal:
    """Structured unmatched signal record."""

    signal_id: str
    ticker: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionVectorizedRetrievalResult:
    """Output payload for vectorized option retrieval."""

    normalized_signals: pd.DataFrame
    selected_contracts: pd.DataFrame
    unmatched_signals: pd.DataFrame
    signal_ohlc: Dict[str, List[pd.DataFrame]]


class OptionVectorizedRetriever:
    """Vectorized option contract retrieval and OHLC assembly.

    Exactly one source must be supplied at run-time:
    - signals: Explicit list of OptionRetrievalSignal
    - strategy: MultiAssetStrategy, adapted via simulate_all()
    """

    def __init__(
        self,
        target_dte: int,
        target_moneyness: float,
        right: Optional[str] = None,
        *,
        dte_tolerance: int = 10,
        moneyness_tolerance: float = 0.2,
        dte_weight: float = 1.0,
        moneyness_weight: float = 1.0,
        roll_enabled: bool = False,
        roll_on_dte: int = 0,
        end_time: str = "16:00",
        print_url: bool = False,
    ) -> None:
        """Initialize retrieval defaults and selection controls.

        Args:
            target_dte: Default DTE target used when signal override is absent.
            target_moneyness: Default moneyness target used when override is absent.
            right: Optional default right ("C" or "P").
            dte_tolerance: Maximum allowed absolute DTE distance.
            moneyness_tolerance: Maximum allowed absolute moneyness distance.
            dte_weight: Score weight for DTE distance.
            moneyness_weight: Score weight for moneyness distance.
            roll_enabled: If True, iteratively rolls contracts until signal end.
            roll_on_dte: Roll trigger in calendar DTE days. Roll when exit is beyond
                (expiration - roll_on_dte).
            end_time: Chain retrieval clock time.
            print_url: Pass-through debug flag for ThetaData API.
        """
        if right is not None and right.upper() not in {"C", "P"}:
            raise ValueError("right must be one of {'C', 'P'} when provided.")
        self.target_dte = target_dte
        self.target_moneyness = target_moneyness
        self.right = right.upper() if right else None
        self.dte_tolerance = dte_tolerance
        self.moneyness_tolerance = moneyness_tolerance
        self.dte_weight = dte_weight
        self.moneyness_weight = moneyness_weight
        self.roll_enabled = roll_enabled
        self.roll_on_dte = int(roll_on_dte)
        self.end_time = end_time
        self.print_url = print_url
        self._chain_spot_cache: Dict[str, pd.Series] = {}

    def run(
        self,
        *,
        signals: Optional[List[OptionRetrievalSignal]] = None,
        strategy: Optional[MultiAssetStrategy] = None,
        include_open_positions: bool = False,
        strategy_end_date: Optional[DateLike] = None,
        n_size: Optional[int] = None,
    ) -> OptionVectorizedRetrievalResult:
        """Run the full retrieval workflow.

        Args:
            signals: Explicit signal rows to process.
            strategy: MultiAssetStrategy source; converted via simulate_all().
            include_open_positions: Include currently-open strategy positions.
            strategy_end_date: End date used when strategy positions are still open.
            n_size: Optional random sample size for the signal set. If None, all
                normalized signals are processed.
            Expected signal format:
                Each signal must be an OptionRetrievalSignal with the following
                fields: ticker, start_date, end_date, side, quantity, signal_id,
                and optional right, target_dte, target_moneyness, metadata. The
                start_date and end_date values may be YYYY-MM-DD strings or
                datetime objects.

        Returns:
            OptionVectorizedRetrievalResult with selected contracts and OHLC payloads.
        """
        normalized_signals = self._build_signals(
            signals=signals,
            strategy=strategy,
            include_open_positions=include_open_positions,
            strategy_end_date=strategy_end_date,
        )
        if n_size is not None:
            if n_size <= 0:
                raise ValueError("n_size must be a positive integer when provided.")
            if n_size < len(normalized_signals):
                normalized_signals = normalized_signals.sample(n=n_size).reset_index(drop=True)
        self._prime_chain_spot_cache(normalized_signals=normalized_signals)

        selected: List[SelectedOptionContract] = []
        unmatched: List[UnmatchedSignal] = []
        signal_ohlc: Dict[str, List[pd.DataFrame]] = {}

        for row in normalized_signals.to_dict(orient="records"):
            signal_row = OptionRetrievalSignal(
                ticker=row["ticker"],
                start_date=row["start_date"],
                end_date=row["end_date"],
                side=int(row["side"]),
                quantity=float(row["quantity"]),
                signal_id=str(row["signal_id"]),
                right=row.get("right"),
                target_dte=row.get("target_dte"),
                target_moneyness=row.get("target_moneyness"),
                metadata=row.get("metadata", {}),
            )
            roll_contracts, ohlc_segments, roll_unmatched = self._retrieve_signal_with_rolls(signal=signal_row)
            selected.extend(roll_contracts)
            unmatched.extend(roll_unmatched)
            if ohlc_segments:
                signal_ohlc[signal_row.signal_id] = ohlc_segments

        selected_df = pd.DataFrame([s.__dict__ for s in selected]) if selected else pd.DataFrame()
        unmatched_df = pd.DataFrame([u.__dict__ for u in unmatched]) if unmatched else pd.DataFrame()

        return OptionVectorizedRetrievalResult(
            normalized_signals=normalized_signals,
            selected_contracts=selected_df,
            unmatched_signals=unmatched_df,
            signal_ohlc=signal_ohlc,
        )

    def _build_signals(
        self,
        *,
        signals: Optional[List[OptionRetrievalSignal]],
        strategy: Optional[MultiAssetStrategy],
        include_open_positions: bool,
        strategy_end_date: Optional[DateLike],
    ) -> pd.DataFrame:
        """Build canonical signal DataFrame from exactly one input source."""
        if (signals is None and strategy is None) or (signals is not None and strategy is not None):
            raise ValueError("Provide exactly one source: signals or strategy.")

        if signals is not None:
            rows = [self._normalize_signal_dict(signal=s) for s in signals]
            return pd.DataFrame(rows)

        assert strategy is not None
        strategy_signals = self.from_multi_asset_strategy(
            strategy=strategy,
            include_open_positions=include_open_positions,
            strategy_end_date=strategy_end_date,
        )
        rows = [self._normalize_signal_dict(signal=s) for s in strategy_signals]
        return pd.DataFrame(rows)

    def _normalize_signal_dict(self, signal: OptionRetrievalSignal) -> Dict[str, Any]:
        """Normalize a single signal and enforce required fields."""
        start_dt = to_datetime(signal.start_date)
        end_dt = to_datetime(signal.end_date)

        if end_dt < start_dt:
            raise ValueError(f"Signal {signal.signal_id} has end_date earlier than start_date.")
        if signal.side not in (-1, 1):
            raise ValueError(f"Signal {signal.signal_id} side must be -1 or 1.")

        right = self._resolve_signal_right(signal=signal)

        return {
            "ticker": signal.ticker.upper(),
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "side": int(signal.side),
            "quantity": float(signal.quantity),
            "signal_id": str(signal.signal_id),
            "right": right,
            "target_dte": signal.target_dte,
            "target_moneyness": signal.target_moneyness,
            "metadata": signal.metadata or {},
        }

    def _resolve_signal_right(self, signal: OptionRetrievalSignal) -> str:
        """Resolve right using precedence: explicit signal > instance default > side."""
        if signal.right is not None:
            right = signal.right.upper()
        elif self.right is not None:
            right = self.right
        else:
            right = "C" if signal.side > 0 else "P"

        if right not in {"C", "P"}:
            raise ValueError(f"Signal {signal.signal_id} has invalid right '{right}'.")
        return right

    @staticmethod
    def from_multi_asset_strategy(
        strategy: MultiAssetStrategy,
        *,
        include_open_positions: bool = False,
        strategy_end_date: Optional[DateLike] = None,
    ) -> List[OptionRetrievalSignal]:
        """Create canonical signals from MultiAssetStrategy simulation trades.

        Args:
            strategy: QuantTools MultiAssetStrategy instance.
            include_open_positions: Include opens with missing closes.
            strategy_end_date: End date used for open positions when included.

        Returns:
            List of OptionRetrievalSignal records inferred from open/close trades.
        """
        results = strategy.simulate_all(finalize=not include_open_positions)
        signals: List[OptionRetrievalSignal] = []
        fallback_end_dt = to_datetime(strategy_end_date) if strategy_end_date is not None else None

        for ticker, trades in results.trades.items():
            pending_open: Dict[str, Dict[str, Any]] = {}
            open_counter = 0
            for trade in trades:
                action = str(trade.get("action", "")).lower()
                sig_id = str(trade.get("signal_id") or f"{ticker}-open-{open_counter}")
                if action == "open":
                    open_counter += 1
                    pending_open[sig_id] = trade
                    continue
                if action != "close":
                    continue

                open_trade = pending_open.pop(sig_id, None)
                if open_trade is None:
                    continue

                close_side = int(trade.get("side", 1))
                signals.append(
                    OptionRetrievalSignal(
                        ticker=ticker,
                        start_date=to_datetime(open_trade["date"]).strftime("%Y-%m-%d"),
                        end_date=to_datetime(trade["date"]).strftime("%Y-%m-%d"),
                        side=1 if close_side >= 0 else -1,
                        quantity=1.0,
                        signal_id=sig_id,
                    )
                )

            if include_open_positions and pending_open:
                if fallback_end_dt is None:
                    raise ValueError(
                        "strategy_end_date is required when include_open_positions=True and unmatched opens exist."
                    )
                for sig_id, open_trade in pending_open.items():
                    signals.append(
                        OptionRetrievalSignal(
                            ticker=ticker,
                            start_date=to_datetime(open_trade["date"]).strftime("%Y-%m-%d"),
                            end_date=fallback_end_dt.strftime("%Y-%m-%d"),
                            side=1,
                            quantity=1.0,
                            signal_id=sig_id,
                        )
                    )

        return signals

    def _select_contract_for_signal(
        self,
        signal: OptionRetrievalSignal,
        *,
        chain_date: Optional[str] = None,
        roll_index: int = 0,
    ) -> Tuple[Optional[SelectedOptionContract], Optional[UnmatchedSignal]]:
        """Select the nearest contract for one signal from chain data."""
        chain_date = chain_date or to_datetime(signal.start_date).strftime("%Y-%m-%d")
        target_right = self._resolve_signal_right(signal=signal)

        try:
            chain = retrieve_chain_bulk(
                symbol=signal.ticker,
                exp=0,
                start_date=chain_date,
                end_date=chain_date,
                end_time=self.end_time,
                option_type=target_right,
                print_url=self.print_url,
            )
        except Exception as exc:
            return None, UnmatchedSignal(
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                reason="chain_retrieval_error",
                details={"error": str(exc)},
            )

        if chain is None or len(chain) == 0:
            return None, UnmatchedSignal(
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                reason="empty_chain",
            )

        normalized_chain, err = self._normalize_chain(
            chain=chain,
            asof_date=chain_date,
            target_right=target_right,
            ticker=signal.ticker,
        )
        if err is not None:
            return None, UnmatchedSignal(
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                reason="invalid_chain_schema",
                details={"error": err},
            )

        target_dte = signal.target_dte if signal.target_dte is not None else self.target_dte
        target_moneyness = signal.target_moneyness if signal.target_moneyness is not None else self.target_moneyness

        ranked = normalized_chain.copy()
        ranked["dte_distance"] = (ranked["dte"] - float(target_dte)).abs()
        ranked["mny_distance"] = (ranked["moneyness"] - float(target_moneyness)).abs()
        ranked["score"] = self.dte_weight * ranked["dte_distance"] + self.moneyness_weight * ranked["mny_distance"]

        if "open_interest" not in ranked.columns:
            ranked["open_interest"] = 0.0
        ranked = ranked.sort_values(
            by=["score", "dte_distance", "mny_distance", "open_interest"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        best = ranked.iloc[0]
        if float(best["dte_distance"]) > float(self.dte_tolerance):
            return None, UnmatchedSignal(
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                reason="dte_out_of_tolerance",
                details={"distance": float(best["dte_distance"]), "tolerance": self.dte_tolerance},
            )
        if float(best["mny_distance"]) > float(self.moneyness_tolerance):
            return None, UnmatchedSignal(
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                reason="moneyness_out_of_tolerance",
                details={"distance": float(best["mny_distance"]), "tolerance": self.moneyness_tolerance},
            )

        contract = SelectedOptionContract(
            signal_id=signal.signal_id,
            ticker=signal.ticker,
            right=str(best["right"]),
            strike=float(best["strike"]),
            expiration=to_datetime(best["expiration"]).strftime("%Y-%m-%d"),
            chain_date=chain_date,
            dte=int(best["dte"]),
            moneyness=float(best["moneyness"]),
            score=float(best["score"]),
            roll_index=roll_index,
        )
        return contract, None

    def _retrieve_signal_with_rolls(
        self,
        *,
        signal: OptionRetrievalSignal,
    ) -> Tuple[List[SelectedOptionContract], List[pd.DataFrame], List[UnmatchedSignal]]:
        """Retrieve one or many contract segments for a signal.

        If roll is enabled and signal end extends past roll trigger, this method
        iteratively re-selects contracts on subsequent roll dates.
        """
        selected_contracts: List[SelectedOptionContract] = []
        unmatched: List[UnmatchedSignal] = []
        segments: List[pd.DataFrame] = []

        signal_end_dt = to_datetime(signal.end_date)
        segment_start_dt = to_datetime(signal.start_date)
        roll_index = 0

        while segment_start_dt <= signal_end_dt:
            chain_date = segment_start_dt.strftime("%Y-%m-%d")
            contract, miss = self._select_contract_for_signal(
                signal,
                chain_date=chain_date,
                roll_index=roll_index,
            )
            if miss is not None:
                unmatched.append(miss)
                break
            assert contract is not None

            expiration_dt = to_datetime(contract.expiration)
            if self.roll_enabled:
                trigger_dt = expiration_dt - timedelta(days=max(self.roll_on_dte, 0))
                segment_end_dt = min(signal_end_dt, trigger_dt)
                if segment_end_dt < segment_start_dt:
                    # Guard for near-expiry selections when roll_on_dte is large.
                    segment_end_dt = min(signal_end_dt, expiration_dt)
            else:
                segment_end_dt = signal_end_dt

            contract.segment_start = segment_start_dt.strftime("%Y-%m-%d")
            contract.segment_end = segment_end_dt.strftime("%Y-%m-%d")
            selected_contracts.append(contract)

            seg_signal = OptionRetrievalSignal(
                ticker=signal.ticker,
                start_date=contract.segment_start,
                end_date=contract.segment_end,
                side=signal.side,
                quantity=signal.quantity,
                signal_id=signal.signal_id,
                right=signal.right,
                target_dte=signal.target_dte,
                target_moneyness=signal.target_moneyness,
                metadata=signal.metadata,
            )
            ohlc_df, ohlc_err = self._retrieve_signal_ohlc(signal=seg_signal, contract=contract)
            if ohlc_err is not None:
                unmatched.append(ohlc_err)
                break
            ohlc_df["roll_index"] = roll_index
            ohlc_df["segment_start"] = contract.segment_start
            ohlc_df["segment_end"] = contract.segment_end
            segments.append(ohlc_df)

            if not self.roll_enabled:
                break
            if segment_end_dt >= signal_end_dt:
                break

            segment_start_dt = segment_end_dt + timedelta(days=1)
            roll_index += 1

        return selected_contracts, segments, unmatched

    def _retrieve_signal_ohlc(
        self,
        *,
        signal: OptionRetrievalSignal,
        contract: SelectedOptionContract,
    ) -> Tuple[pd.DataFrame, Optional[UnmatchedSignal]]:
        """Retrieve full OHLC time series for one selected contract."""
        start_str = to_datetime(signal.start_date).strftime("%Y-%m-%d")
        end_str = to_datetime(signal.end_date).strftime("%Y-%m-%d")

        try:
            ohlc = retrieve_eod_ohlc(
                symbol=contract.ticker,
                start_date=start_str,
                end_date=end_str,
                strike=contract.strike,
                exp=contract.expiration,
                right=contract.right,
                print_url=self.print_url,
            )
        except Exception as exc:
            return pd.DataFrame(), UnmatchedSignal(
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                reason="ohlc_retrieval_error",
                details={"error": str(exc)},
            )

        if ohlc is None or len(ohlc) == 0:
            return pd.DataFrame(), UnmatchedSignal(
                signal_id=signal.signal_id,
                ticker=signal.ticker,
                reason="empty_ohlc",
            )

        ohlc_df = pd.DataFrame(ohlc).copy()
        ohlc_df["signal_id"] = signal.signal_id
        ohlc_df["ticker"] = contract.ticker
        ohlc_df["strike"] = contract.strike
        ohlc_df["expiration"] = contract.expiration
        ohlc_df["right"] = contract.right
        return ohlc_df, None

    def _normalize_chain(
        self,
        chain: pd.DataFrame,
        asof_date: str,
        target_right: str,
        ticker: str,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Normalize chain DataFrame into ranking columns used by selector."""
        frame = pd.DataFrame(chain).copy()
        strike_col = self._find_col(frame.columns.tolist(), ["strike", "Strike"]) 
        right_col = self._find_col(frame.columns.tolist(), ["right", "Right", "put_call", "putcall", "option_type"])
        exp_col = self._find_col(frame.columns.tolist(), ["expiration", "Expiration", "exp", "expiry"])
        if strike_col is None or right_col is None or exp_col is None:
            return pd.DataFrame(), "missing required columns strike/right/expiration"

        oi_col = self._find_col(frame.columns.tolist(), ["open_interest", "Open_interest", "OpenInterest", "oi"])

        frame["strike"] = pd.to_numeric(frame[strike_col], errors="coerce")
        frame["right"] = frame[right_col].astype(str).str.upper()
        frame = frame[frame["right"] == target_right].copy()

        exp_raw = frame[exp_col].astype(str)
        exp_fmt = exp_raw.str.replace(r"[^0-9]", "", regex=True)
        exp_parsed = exp_raw.where(exp_fmt.str.len() != 8, other=exp_fmt)
        frame["expiration"] = to_datetime(exp_parsed.tolist())
        asof_dt = to_datetime(asof_date)
        frame["dte"] = (frame["expiration"] - asof_dt).dt.days

        frame = frame[(frame["dte"] >= 0) & frame["strike"].notna() & (frame["strike"] > 0)].copy()
        if frame.empty:
            return pd.DataFrame(), "no candidates after right/dte filtering"

        if oi_col is not None:
            frame["open_interest"] = pd.to_numeric(frame[oi_col], errors="coerce").fillna(0.0)
        else:
            frame["open_interest"] = 0.0

        chain_spot = self._get_chain_spot_on_date(ticker=ticker, date_str=asof_date)
        if chain_spot is None or chain_spot <= 0:
            return pd.DataFrame(), "missing chain_spot on asof date"
        frame["spot"] = float(chain_spot)

        # Keep moneyness convention identical to populate_cache_with_chain:
        # Calls: Strike / spot; Puts: spot / Strike
        frame["moneyness"] = 0.0
        frame.loc[frame["right"] == "C", "moneyness"] = (
            frame.loc[frame["right"] == "C", "strike"] / frame.loc[frame["right"] == "C", "spot"]
        )
        frame.loc[frame["right"] == "P", "moneyness"] = (
            frame.loc[frame["right"] == "P", "spot"] / frame.loc[frame["right"] == "P", "strike"]
        )

        return frame, None

    def _prime_chain_spot_cache(self, normalized_signals: pd.DataFrame) -> None:
        """Preload chain-spot timeseries per ticker for signal date range.

        Uses MarketTimeseries chain_spot endpoint to ensure moneyness is always
        computed from chain spot rather than equity spot.
        """
        self._chain_spot_cache = {}
        if normalized_signals.empty:
            return

        mt = get_timeseries_obj(live=False)
        grouped = normalized_signals.groupby("ticker", as_index=False).agg(
            start_date=("start_date", "min"),
            end_date=("start_date", "max"),
        )
        for row in grouped.to_dict(orient="records"):
            ticker = str(row["ticker"]).upper()
            start_str = to_datetime(row["start_date"]).strftime("%Y-%m-%d")
            end_str = to_datetime(row["end_date"]).strftime("%Y-%m-%d")
            try:
                ts = mt.get_timeseries(
                    sym=ticker,
                    factor="chain_spot",
                    start_date=start_str,
                    end_date=end_str,
                ).chain_spot
            except Exception:
                continue

            if ts is None or len(ts) == 0:
                continue

            ts_df = pd.DataFrame(ts).copy()
            ts_df.index = pd.to_datetime(ts_df.index)
            spot_col = self._find_col(
                ts_df.columns.tolist(),
                ["close", "Close", "spot", "Spot", "chain_spot", "Chain_spot"],
            )
            if spot_col is None:
                continue

            series = pd.to_numeric(ts_df[spot_col], errors="coerce").dropna()
            if series.empty:
                continue

            series.index = pd.to_datetime(series.index)
            self._chain_spot_cache[ticker] = series.sort_index()

    def _get_chain_spot_on_date(self, ticker: str, date_str: str) -> Optional[float]:
        """Get chain spot for ticker on date, using forward-fill within cached history."""
        series = self._chain_spot_cache.get(ticker.upper())
        if series is None or series.empty:
            return None

        target = to_datetime(date_str)
        if target in series.index:
            val = series.loc[target]
            return float(val) if pd.notna(val) else None

        prior = series[series.index <= target]
        if prior.empty:
            return None
        val = prior.iloc[-1]
        return float(val) if pd.notna(val) else None

    @staticmethod
    def _find_col(columns: List[str], aliases: List[str]) -> Optional[str]:
        """Find the first matching column name by case-insensitive alias."""
        lookup = {col.lower(): col for col in columns}
        for alias in aliases:
            if alias.lower() in lookup:
                return lookup[alias.lower()]
        return None
