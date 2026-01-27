import os
import signal
from pathlib import Path
import pandas as pd
from trade.helpers.Logging import setup_logger
from trade import register_signal
logger = setup_logger("trade.optionlib.vol.implied_vol")

TIME_BUCKET = []


def _record_time(start_time: float, end_time: float, action: str, info: dict) -> None:
    elapsed = end_time - start_time
    meta = {
        "action": action,
        "elapsed_time": elapsed,
    }
    meta.update(info)
    TIME_BUCKET.append(meta)


def _offload_time_bucket():
    """Offload the time bucket to a CSV for analysis."""
    if not TIME_BUCKET:
        logger.info("No timing data to offload.")
        return

    ## Loc
    loc = Path(os.environ.get("GEN_CACHE_PATH", ".")) / "timing_analysis"
    file_name = loc / "time_analysis.csv"
    loc.mkdir(parents=True, exist_ok=True)

    ## Load old data if exists and append
    if file_name.exists():
        old_data = pd.read_csv(file_name)
        old_records = old_data.to_dict(orient="records")
        TIME_BUCKET.extend(old_records)

    df = pd.DataFrame(TIME_BUCKET)
    df.to_csv(file_name, index=False)
    TIME_BUCKET.clear()


register_signal(signal.SIGTERM, _offload_time_bucket)
register_signal(signal.SIGINT, _offload_time_bucket)
register_signal("exit", _offload_time_bucket)
