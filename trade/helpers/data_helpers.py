from trade.backtester_.backtester_ import PTDataset
from datetime import datetime
from trade.helpers.Logging import setup_logger
logger = setup_logger("trade.helpers.data_helpers")

## Ensure all datasets have the current time
def log_missing_data(dataset: PTDataset, T_0: str | datetime) -> None:
    """
    Log missing data for each dataset in timeseries.
    """
    tick = dataset.name
    if T_0.date() not in dataset.data.index.date and T_0.hour < 10:
        logger.error(f"Dataset for {tick} does not contain the current time {T_0}")
    else:
        logger.info(f"Dataset for {tick} contains the current time {T_0}")


## Format Columns in timeseries to Capitalize
def format_columns(dataset: PTDataset) -> None:
    """
    Format columns in each dataset of timeseries to capitalize.
    """
    tick = dataset.name
    dataset.data.columns = [col.capitalize() for col in dataset.data.columns]
    logger.info(f"Formatted columns for {tick}: {dataset.data.columns.tolist()}")
