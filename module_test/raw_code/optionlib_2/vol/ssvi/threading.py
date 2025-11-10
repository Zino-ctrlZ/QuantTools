from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Semaphore
from trade.helpers.Logging import setup_logger
logger = setup_logger('optionlib.ssvi.threading')

class BackgroundFits:
    """
    One shared, bounded thread pool for all background fits.
    - max_workers: threads running background fits concurrently
    - max_queue: limit queued-but-not-started fits (backpressure)
    Policy: if the queue is full, we SKIP the submission (non-blocking).
    """
    def __init__(self, max_workers: int = 3, max_queue: int = 25):
        self._exec = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ssvi-bg")
        # Semaphore counts 'in-flight' = running + queued tasks
        self._slots = Semaphore(max_queue + max_workers)
        self._futs: Dict[str, Future] = {}

    def submit(self, key: str, fn, *args, **kwargs) -> bool:
        """
        Return True if scheduled, False if skipped due to full queue.
        """

        ## _slots.acquire() is non-blocking; if it fails, we skip the job
        ## Essentially, we are counting. if it's full, we skip.
        ## Slots start with given max_queue + max_workers, acquire() reduces count by 1, release() increases count by 1
        acquired = self._slots.acquire(blocking=False)
        if not acquired:

            ## If acquire() fails, we skip the job
            logger.info("BG queue full; skipping background job %s", key)
            return False

        def _done_cb(f: Future):
            ## Release the slot after the job is done (whether success or failure)
            self._slots.release()
            exc = f.exception()
            if exc:
                logger.warning("BG job %s failed: %s", key, exc, exc_info=True)

            ## Remove from tracking dict
            self._futs.pop(key, None)

        fut = self._exec.submit(fn, *args, **kwargs)
        fut.add_done_callback(_done_cb)
        self._futs[key] = fut
        return True

    def status(self, key: Optional[str] = None) -> Dict[str, str]:
        """
        Get the status of all background jobs.
        Returns a dict mapping job keys to their status: "running", "done", or "queued".
        """
        st = {}
        for k, f in (self._futs.items()):
            if f.running():
                status = "running"
            elif f.done():
                status = "done"
            else:
                status = "queued"
            st[k] = status
        return st if key is None else {key: st.get(key, "not found")}
    

    def shutdown(self) -> None:
        """
        Shutdown the background executor.
        """
        self._exec.shutdown(wait=False, cancel_futures=True)
        self._futs.clear()

    def restart(self) -> None:
        """
        Restart the background executor.
        """
        self.shutdown()
        self._exec = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ssvi-bg")
        self._futs.clear()