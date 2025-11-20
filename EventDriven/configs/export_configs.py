from typing import Any
from EventDriven.configs.base import BaseConfigs
from datetime import datetime
from dataclasses import dataclass
import yaml


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


def collect_run_configs(root: Any):
    """
    Return a dict of {label: config_instance} for all configs under `root`.
    """
    configs = {}
    counts = {}

    for cfg in walk_configs(root):
        cls_name = cfg.__class__.__name__
        counts.setdefault(cls_name, 0)
        counts[cls_name] += 1

        # avoid overwriting when multiple instances of same class exist
        label = f"{cls_name}_{counts[cls_name]}"
        configs[label] = cfg

    return configs


def export_run_configs(root: Any):
    """
    Export all configs into plain Python dicts (e.g. to save to DB, JSON, etc.).
    """
    configs = collect_run_configs(root)
    exported = {}
    run_name = None

    for label, cfg in configs.items():
        # pydantic dataclasses usually have .__dict__ already good enough
        exported[label] = dict(cfg.__dict__)
        if run_name is None and "run_name" in exported[label]:
            run_name = exported[label]["run_name"]
    return RunConfigBundle(run_name=run_name, created_at=datetime.now(), configs=exported)


def walk_configs(root: Any, _seen=None):
    """
    Recursively walk an object graph and yield all BaseConfigs instances.
    This works as long as configs are reachable via attributes / lists / dicts.
    """

    if _seen is None:
        _seen = set()

    obj_id = id(root)
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    # If the object itself is a config, yield it
    if isinstance(root, BaseConfigs):
        yield root

    # Handle containers first
    if isinstance(root, dict):
        for v in root.values():
            yield from walk_configs(v, _seen)
        return

    if isinstance(root, (list, tuple, set)):
        for v in root:
            yield from walk_configs(v, _seen)
        return

    # For "normal" objects, walk their attributes
    try:
        attrs = vars(root)
    except TypeError:
        # e.g. builtins, C-extensions, etc.
        return

    for v in attrs.values():
        yield from walk_configs(v, _seen)


def tag_run(root: Any, run_name: str):
    """
    Set run_name on every config reachable from `root`.
    """
    for cfg in walk_configs(root):
        cfg.set(run_name=run_name)

