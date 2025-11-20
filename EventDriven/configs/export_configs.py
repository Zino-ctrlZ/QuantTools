
from EventDriven.configs.base import BaseConfigs
from collections.abc import Mapping
from typing import Any, Iterable, Tuple, Dict
from datetime import datetime
from dataclasses import dataclass
import yaml
import logging


@dataclass
class RunConfigBundle:
    run_name: str
    created_at: datetime
    configs: dict[str, dict]

    
    def save_exported_configs(self, filename: str):
        """
        Save the exported configs to a YAML file.
        
        Args:
            filename (str): The path to the file where configs will be saved.
        """
        with open(filename, "w") as f:
            yaml.dump(self.configs, f)

    def __repr__(self):
        return f"RunConfigBundle(run_name={self.run_name!r}, created_at={self.created_at!r}, configs={list(self.configs.keys())})"


def collect_run_configs(root: Any, debug: bool = False) -> Dict[str, BaseConfigs]:
    """
    Return a dict of {label: config_instance} for all *unique*
    BaseConfigs instances under `root`.

    - Dedupes by object id globally in this call
    - Keeps *all* instances for a given class
    - Label format: "<ClassName>_<i>" where i is per-class index
    """
    configs: Dict[str, BaseConfigs] = {}
    counts: Dict[str, int] = {}
    seen_ids: set[int] = set()

    for cfg, path in walk_configs(root):
        cfg_id = id(cfg)
        if cfg_id in seen_ids:
            continue
        seen_ids.add(cfg_id)

        cls_name = cfg.__class__.__name__
        counts[cls_name] = counts.get(cls_name, 0) + 1
        label = f"{cls_name}_{counts[cls_name]}"

        # If you want to debug where it came from:
        if debug:
            print(f"[CONFIG FOUND] {label} at {path}")

        configs[label] = cfg

    return configs


def export_run_configs(root: Any, debug: bool = False) -> RunConfigBundle:
    """
    Export all configs into plain Python dicts (e.g. to save to DB, JSON, etc.).
    """
    configs = collect_run_configs(root, debug=debug)
    exported = {}
    run_name = None

    for label, cfg in configs.items():
        # pydantic dataclasses usually have .__dict__ already good enough
        exported[label] = dict(cfg.__dict__)
        if run_name is None and "run_name" in exported[label]:
            run_name = exported[label]["run_name"]
    return RunConfigBundle(run_name=run_name, created_at=datetime.now(), configs=exported)


# def walk_configs(root: Any, _seen=None):
#     """
#     Recursively walk an object graph and yield all BaseConfigs instances.
#     This works as long as configs are reachable via attributes / lists / dicts.
#     """

#     if _seen is None:
#         _seen = set()

#     obj_id = id(root)
#     if obj_id in _seen:
#         return
#     _seen.add(obj_id)

#     # If the object itself is a config, yield it
#     if isinstance(root, BaseConfigs):
#         yield root

#     # Handle containers first
#     if isinstance(root, dict):
#         for v in root.values():
#             yield from walk_configs(v, _seen)
#         return

#     if isinstance(root, (list, tuple, set)):
#         for v in root:
#             yield from walk_configs(v, _seen)
#         return

#     # For "normal" objects, walk their attributes
#     try:
#         attrs = vars(root)
#         if not isinstance(attrs, dict):
#             return
#     except TypeError:
#         # e.g. builtins, C-extensions, etc.
#         return

#     for v in attrs.values():
#         yield from walk_configs(v, _seen)

def walk_configs(root: Any, _seen=None, _path: str = "root") -> Iterable[Tuple[BaseConfigs, str]]:
    """
    Recursively walk an object graph starting at `root` and yield
    (config_instance, path) for all *unique* BaseConfigs instances.

    - Dedupes by object id per call
    - Skips logger / handler / logrecord internals
    - Skips IPython / Jupyter / traitlets internals
    """

    if _seen is None:
        _seen = set()

    obj_id = id(root)
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    # 1) Direct hit: it's a config
    if isinstance(root, BaseConfigs):
        # Debug print if you want to see them:
        # print(f"[FOUND CONFIG] {_path} → {root.__class__.__name__} (id={obj_id})")
        yield root, _path
        # Don't return; configs may contain other configs inside them.

    # 2) Cut off obvious gateways into notebook internals

    # Don't recurse into logging internals
    if isinstance(root, (logging.Logger, logging.Handler, logging.LogRecord)):
        return

    # Don't recurse into IPython / Jupyter / traitlets / jupyter_client internals
    mod = getattr(root, "__module__", "") or ""
    if mod.startswith(("ipykernel", "IPython", "traitlets", "jupyter_client")):
        return

    # 3) Containers

    # Mappings (dict-like)
    if isinstance(root, Mapping):
        try:
            items = root.items()
        except NotImplementedError:
            # e.g. urllib3's RecentlyUsedContainer
            return

        for k, v in items:
            child_path = f"{_path}[{repr(k)}]"
            yield from walk_configs(v, _seen, child_path)
        return

    # Sequences
    if isinstance(root, (list, tuple, set)):
        for idx, v in enumerate(root):
            child_path = f"{_path}[{idx}]"
            yield from walk_configs(v, _seen, child_path)
        return

    # 4) Primitive leaves we don't introspect
    if isinstance(root, (str, bytes, int, float, bool, type(None))):
        return

    # 5) Generic objects: walk their __dict__ if it's a real dict
    try:
        attrs = vars(root)
        if not isinstance(attrs, dict):
            return
    except TypeError:
        # builtins, C-extensions, etc.
        return

    for name, v in attrs.items():
        # Also skip the 'logger' attribute explicitly
        if name == "logger":
            continue
        child_path = f"{_path}.{name}"
        yield from walk_configs(v, _seen, child_path)


def tag_run(root: Any, run_name: str):
    """
    Set run_name on every config reachable from `root`.
    """
    for cfg, _ in walk_configs(root):
        cfg.set(run_name=run_name)

