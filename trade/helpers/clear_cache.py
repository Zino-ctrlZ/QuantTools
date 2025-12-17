import json
import shutil
from datetime import date, datetime
from pathlib import Path
import os
from trade.helpers.Logging import setup_logger
from dotenv import load_dotenv
logger = setup_logger('trade.helpers.clear_cache', stream_log_level="DEBUG")
load_dotenv()
REGISTRY = Path(os.environ["WORK_DIR"]) / "trade" / "helpers" / "clear_dirs.json"

def cleanup_expired_caches():
    files_deleted = 0
    if not REGISTRY.exists():
        logger.info(f"No cache registry found at {REGISTRY}, skipping cleanup.")
        return

    # 1) load the registry
    with REGISTRY.open() as f:
        data = json.load(f)

    today = date.today()
    changed = False

    # 2) iterate over a static list of items to allow mutation
    for cache_dir, expiry_str in list(data.items()):
        try:
            exp = datetime.fromisoformat(expiry_str).date()
        except ValueError:
            # bad format – skip or log
            continue

        if today >= exp:
            # 3) delete the cache folder
            logger.info(f"Deleting expired cache directory: {cache_dir}, expired on {exp}")
            shutil.rmtree(cache_dir, ignore_errors=True)
            # 4) remove from registry
            data.pop(cache_dir, None)
            changed = True
            files_deleted += 1
    
    # 5) write back updated registry if anything changed
    if changed:
        tmp = REGISTRY.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(REGISTRY)
        logger.info(f"Updated cache registry after cleanup. Deleted {files_deleted} cache directories.")
    else:
        logger.info(f"No expired caches to delete on {today}.")

if __name__ == "__main__":
    cleanup_expired_caches()
