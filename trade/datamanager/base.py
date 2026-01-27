from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Optional, Type, TypeVar
from trade.datamanager.config import OptionDataConfig
from trade.helpers.helper import CustomCache
from trade.helpers.Logging import setup_logger
from pathlib import Path
from .vars import DM_GEN_PATH
from ._enums import Interval, ArtifactType, SeriesId
from .utils.enums_utils import construct_cache_key
logger = setup_logger("trade.datamanager.base")

# Assumes you already have these (from your cache_key module)
# from cache_key import construct_cache_key, Interval, ArtifactType, SeriesId

T = TypeVar("T")


# REMEBER: Take out the commented out parts
@dataclass(frozen=True, slots=True)
class CacheSpec:
    """
    Optional: a small config object you can pass around, so all managers
    initialize their caches in a consistent way.

    If you already have a cache registry/factory, you may not need this.

    args:
        base_dir (Optional[Path]): Directory for cache storage.
        default_expire_days (Optional[int]): Default expiration time in days. This is how many days till the entire cache entry expires.
        default_expire_seconds (Optional[int]): Default expiration time in seconds. This is how many seconds till a single cache entry expires.
        cache_fname (Optional[str]): Foldername for the cache storage.
        clear_on_exit (bool): If True, clears the cache on exit.
    """

    base_dir: Optional[Path] = DM_GEN_PATH.as_posix()
    default_expire_days: Optional[int] = 500
    default_expire_seconds: Optional[int] = None
    cache_fname: Optional[str] = None
    clear_on_exit: bool = False


class BaseDataManager(ABC):
    """
    Foundation class for all DataManagers.

    Goals:
    - Every inheritor gets a cache.
    - Every inheritor MUST define CACHE_NAME.
    - Provide consistent key creation (namespaced).
    - Provide thin get/set/get_or_compute wrappers.
    - Keep business logic out of the base.
    """

    CACHE_NAME: ClassVar[str] = ""
    DEFAULT_INTERVAL: ClassVar[Optional["Interval"]] = None
    DEFAULT_SERIES_ID: ClassVar["SeriesId"]  # prefer explicit in subclasses
    _CACHE_NAME_REGISTRY: ClassVar[Dict[str, Type["BaseDataManager"]]] = {}
    CONFIG: OptionDataConfig = OptionDataConfig()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforces that all subclasses define CACHE_NAME and DEFAULT_SERIES_ID."""
        super().__init_subclass__(**kwargs)

        if cls is BaseDataManager:
            return

        cache_name = getattr(cls, "CACHE_NAME", None)

        if not isinstance(cache_name, str) or not cache_name.strip():
            raise TypeError(f"{cls.__name__} must define a non-empty class variable CACHE_NAME: str")

        cache_name = cache_name.strip()

        # Enforce uniqueness to avoid collisions
        existing = cls._CACHE_NAME_REGISTRY.get(cache_name) # noqa
        # if existing is not None and existing is not cls:
        #     raise TypeError(
        #         f"Duplicate CACHE_NAME='{cache_name}'. "
        #         f"Already used by {existing.__name__}. "
        #         f"Pick a unique CACHE_NAME for {cls.__name__}."
        #     )

        cls._CACHE_NAME_REGISTRY[cache_name] = cls

        # Optional: enforce that DEFAULT_SERIES_ID exists (if you want)
        if not hasattr(cls, "DEFAULT_SERIES_ID"):
            raise TypeError(f"{cls.__name__} must define DEFAULT_SERIES_ID (e.g., SeriesId.HIST).")

    def __init__(
        self,
        *,
        cache_spec: Optional[CacheSpec] = None,
        enable_namespacing: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        cache:
            Your existing CustomCache instance (diskcache-backed).
        cache_spec:
            Optional shared configuration (base_dir, TTL defaults, etc.).
        enable_namespacing:
            If True, keys are prefixed with CACHE_NAME to avoid collisions.
        """
        self.cache_spec = cache_spec or CacheSpec(cache_fname=self.CACHE_NAME)
        self.cache = CustomCache(
            location=self.cache_spec.base_dir,
            fname=self.cache_spec.cache_fname,
            expire_days=self.cache_spec.default_expire_days,
            clear_on_exit=self.cache_spec.clear_on_exit,
        )
        self.enable_namespacing = enable_namespacing
        out = self.cache.expire()
        if out > 0:
            logger.info(f"{self.CACHE_NAME} has expired {out} entries")

    # Key construction
    def make_key(
        self,
        *,
        symbol: str,
        interval: Optional[Interval] = None,
        artifact_type: ArtifactType,
        series_id: Optional[SeriesId] = None,
        **extra_parts: Any,
    ) -> str:
        """
        Namespaced key builder that wraps your construct_cache_key.

        You decided:
        - no caching SNAPSHOT series_id (but you might still request it)
        - time is explicit if you do AT_TIME
        """
        interval = interval if interval is not None else self.DEFAULT_INTERVAL
        series_id = series_id if series_id is not None else self.DEFAULT_SERIES_ID

        raw = construct_cache_key(
            symbol=symbol,
            interval=interval,
            artifact_type=artifact_type,
            series_id=series_id,
            **extra_parts,
        )

        if not self.enable_namespacing:
            return raw

        return f"{self.CACHE_NAME}|{raw}"

    # Cache IO
    def get(self, key: str, default: Any = None) -> Any:
        return self.cache.get(key, default=default)

    def set(self, key: str, value: Any, *, expire: Optional[int] = None) -> None:
        if expire is None:
            expire = self.cache_spec.default_expire_seconds
        self.cache.set(key, value, expire=expire)

    def delete(self, key: str) -> None:
        self.cache.delete(key)

    def contains(self, key: str) -> bool:
        return key in self.cache

    def cache_it(self, key: str, value: Any, *, expire: Optional[int] = None) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.cache() not implemented.")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        *,
        expire: Optional[int] = None,
        force: bool = False,
    ) -> T:
        """
        Read-through caching helper.

        force=True bypasses cache read, recomputes and overwrites cache.
        """
        if not force:
            hit = self.cache.get(key, default=None)
            if hit is not None:
                return hit  # type: ignore[return-value]

        value = compute_fn()
        self.set(key, value, expire=expire)
        return value

    # Offload hook (cron calls this)
    def offload(self, *args: Any, **kwargs: Any) -> None:
        """
        Optional standard hook.

        You can override in subclasses or implement a shared offloader that
        knows how to iterate keys / export values. Keeping it as a stub here
        avoids forcing a storage design too early.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.offload() not implemented.")
