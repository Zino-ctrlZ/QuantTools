
from EventDriven.configs.base import BaseConfigs
from collections.abc import Mapping
from typing import Any, Iterable, Tuple, Dict
from datetime import datetime
from dataclasses import dataclass, field
import yaml
import logging
import importlib



logger = logging.getLogger(__name__)


@dataclass
class ConfigLocation:
    """Metadata about where a config lives in the object graph."""
    label: str
    config_class: str
    path: str
    parent_path: str
    attribute_name: str
    object_id: int


@dataclass
class RunConfigBundle:
    run_name: str
    created_at: datetime
    configs: dict[str, dict]
    metadata: dict[str, dict] = field(default_factory=dict)  # Stores path info per config

    
    def save_to_yaml(self, filename: str):
        """
        Save the configs and metadata to a YAML file.
        
        Args:
            filename (str): The path to the file where configs will be saved.
        """
        data = {
            'run_name': self.run_name,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            'configs': self.configs,
            'metadata': self.metadata
        }
        # Use safe_dump to avoid Python object tags
        with open(filename, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load_from_yaml(cls, filename: str) -> 'RunConfigBundle':
        """
        Load a config bundle from a YAML file.
        
        Args:
            filename (str): The path to the YAML file.
            
        Returns:
            RunConfigBundle: The loaded config bundle.
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        confs = data.get('configs', {})
        conf_bund_cls = {}
        for label in confs.keys():
            conf_name = label.split("_")[0]
            cls_module = importlib.import_module("EventDriven.configs.core")
            conf_cls = getattr(cls_module, conf_name)
            conf_bund_cls[conf_name] = conf_cls(**confs[label])
        
        ret =  cls(
            run_name=data.get('run_name', ''),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else data.get('created_at', datetime.now()),
            configs=conf_bund_cls,
            metadata=data.get('metadata', {})
        )
        return ret
    
    def apply_to(self, root: Any, strict: bool = True, verify_paths: bool = True) -> Dict[str, str]:
        """
        Apply this bundle's configs to an object.
        
        Args:
            root: The root object to apply configs to
            strict: If True, raise errors on mismatches
            verify_paths: If True, verify config paths match metadata
            
        Returns:
            Dict mapping labels to their applied paths
        """
        return apply_run_configs(root, self.configs, self.metadata, strict=strict, verify_paths=verify_paths)

    def __repr__(self):
        return f"RunConfigBundle(run_name={self.run_name!r}, created_at={self.created_at!r}, configs={list(self.configs.keys())})"


def collect_run_configs(root: Any, debug: bool = False, include_metadata: bool = False) -> Dict[str, BaseConfigs] | Tuple[Dict[str, BaseConfigs], Dict[str, ConfigLocation]]:
    """
    Return a dict of {label: config_instance} for all *unique*
    BaseConfigs instances under `root`.

    - Dedupes by object id globally in this call
    - Keeps *all* instances for a given class
    - Label format: "<ClassName>_<i>" where i is per-class index
    
    Args:
        root: The root object to collect configs from
        debug: If True, print debug info
        include_metadata: If True, return (configs, metadata) tuple
        
    Returns:
        Dict of {label: config} or tuple of (configs, metadata)
    """
    configs: Dict[str, BaseConfigs] = {}
    metadata: Dict[str, ConfigLocation] = {}
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

        # Parse path to extract parent and attribute
        parts = path.rsplit('.', 1)
        parent_path = parts[0] if len(parts) > 1 else "root"
        attr_name = parts[1] if len(parts) > 1 else path

        # Store metadata
        location = ConfigLocation(
            label=label,
            config_class=cls_name,
            path=path,
            parent_path=parent_path,
            attribute_name=attr_name,
            object_id=cfg_id
        )
        metadata[label] = location

        # If you want to debug where it came from:
        if debug:
            print(f"[CONFIG FOUND] {label} at {path}")

        configs[label] = cfg

    if include_metadata:
        return configs, metadata
    return configs


def _sanitize_for_yaml(obj: Any) -> Any:
    """
    Convert objects to YAML-safe basic types.
    Handles nested dicts, lists, tuples, and custom objects.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, (datetime, )):
        return obj.isoformat()
    
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(item) for item in obj]
    
    if isinstance(obj, BaseConfigs):
        # Recursively sanitize config objects
        return {k: _sanitize_for_yaml(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    
    # For other objects, try to convert to dict or return string representation
    if hasattr(obj, '__dict__'):
        try:
            return {k: _sanitize_for_yaml(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        except Exception:
            return str(obj)
    
    return str(obj)


def export_run_configs(root: Any, debug: bool = False) -> RunConfigBundle:
    """
    Export all configs into plain Python dicts (e.g. to save to DB, JSON, etc.).
    Includes metadata for path verification during application.
    """
    configs, metadata_objs = collect_run_configs(root, debug=debug, include_metadata=True)
    exported = {}
    metadata_dict = {}
    run_name = None

    for label, cfg in configs.items():
        # Sanitize config dict for YAML serialization
        exported[label] = _sanitize_for_yaml(dict(cfg.__dict__))
        if run_name is None and "run_name" in exported[label]:
            run_name = exported[label]["run_name"]
        
        # Store metadata as dict (for YAML serialization)
        loc = metadata_objs[label]
        metadata_dict[label] = {
            'config_class': loc.config_class,
            'path': loc.path,
            'parent_path': loc.parent_path,
            'attribute_name': loc.attribute_name
        }
    
    return RunConfigBundle(
        run_name=run_name, 
        created_at=datetime.now(), 
        configs=exported,
        metadata=metadata_dict
    )


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


def apply_run_configs(
    root: Any,
    configs: Dict[str, dict],
    metadata: Dict[str, dict] = None,
    strict: bool = True,
    verify_paths: bool = True
) -> Dict[str, str]:
    """
    Apply configs (as dicts from YAML) back to the object graph with validation.
    
    This is Strategy #1: Path-based verification.
    
    Args:
        root: The root object (e.g., OptionSignalBacktest instance)
        configs: Dict of {label: config_dict} from YAML/export
        metadata: Dict of {label: metadata_dict} with path info
        strict: If True, raise error if config not found
        verify_paths: If True, verify paths match metadata
        
    Returns:
        Dict mapping labels to their paths in the object graph
        
    Raises:
        ValueError: If config not found or type mismatch
        TypeError: If config type doesn't match expected type
        
    Example:
        >>> # Export configs
        >>> bundle = export_run_configs(backtest)
        >>> bundle.save_to_yaml('my_configs.yaml')
        >>> 
        >>> # Later, load and apply
        >>> bundle = RunConfigBundle.load_from_yaml('my_configs.yaml')
        >>> bundle.apply_to(new_backtest)
    """
    # Collect current config locations
    current_configs, current_metadata = collect_run_configs(root, debug=False, include_metadata=True)
    
    applied = {}
    errors = []
    
    for label, config_dict in configs.items():
        # Check if this config exists in current object
        if label not in current_configs:
            msg = f"Config {label} not found in object graph. Available: {list(current_configs.keys())}"
            if strict:
                raise ValueError(msg)
            errors.append(msg)
            continue
        
        current_config = current_configs[label]
        current_loc = current_metadata[label]
        
        # Verify config class matches (using metadata if available)
        if metadata and label in metadata:
            expected_class = metadata[label].get('config_class')
            actual_class = current_loc.config_class
            
            if expected_class != actual_class:
                msg = (
                    f"Config class mismatch for {label}:\n"
                    f"  Expected: {expected_class}\n"
                    f"  Got: {actual_class}\n"
                    f"  At path: {current_loc.path}"
                )
                if strict:
                    raise TypeError(msg)
                errors.append(msg)
                continue
            
            # Verify path matches (if requested)
            if verify_paths:
                expected_path = metadata[label].get('path')
                actual_path = current_loc.path
                
                if expected_path != actual_path:
                    msg = (
                        f"Config path mismatch for {label}:\n"
                        f"  Expected: {expected_path}\n"
                        f"  Got: {actual_path}\n"
                        f"  This suggests the object structure changed."
                    )
                    logger.warning(msg)
                    if strict:
                        raise ValueError(msg)
                    errors.append(msg)
        
        # Apply config by updating attributes
        for field_name, field_value in config_dict.items():
            if not field_name.startswith('_'):  # Skip private fields
                try:
                    setattr(current_config, field_name, field_value)
                except Exception as e:
                    msg = f"Failed to set {label}.{field_name} = {field_value}: {e}"
                    if strict:
                        raise ValueError(msg) from e
                    errors.append(msg)
        
        applied[label] = current_loc.path
    
    if errors and not strict:
        logger.warning(f"Applied configs with {len(errors)} warnings:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return applied


def validate_config_placement(root: Any, raise_on_error: bool = False) -> list[str]:
    """
    Validate that all configs are correctly placed and typed.
    
    This is Strategy #4: Validation function.
    
    Args:
        root: The root object to validate
        raise_on_error: If True, raise ValueError on first error
        
    Returns:
        List of validation errors (empty if all valid)
        
    Example:
        >>> errors = validate_config_placement(backtest)
        >>> if errors:
        ...     print("Validation errors:", errors)
    """
    errors = []
    
    # Define expected config types for known attributes
    EXPECTED_CONFIGS = {
        'OptionSignalBacktest': {
            'config': 'BacktesterConfig',
        },
        'OptionSignalPortfolio': {
            'config': 'PortfolioManagerConfig',
            'cash_allocator_config': 'CashAllocatorConfig',
        },
        'RiskManager': {
            'config': 'RiskManagerConfig',
        },
        'OrderPicker': {
            'config': 'OrderPickerConfig',
            'order_schema_configs': 'OrderSchemaConfigs',
            'chain_config': 'ChainConfig',
        },
        'PositionAnalyzer': {
            'config': 'PositionAnalyzerConfig',
        },
        'LimitsAndSizingCog': {
            'config': 'LimitsAndSizingConfig',
            'sizer_configs': ('DefaultSizerConfigs', 'ZscoreSizerConfigs'),  # Can be either
            'limits_enabled_config': 'LimitsEnabledConfig',
        },
    }
    
    def check_object(obj, path="root", _seen=None):
        if _seen is None:
            _seen = set()
        
        obj_id = id(obj)
        if obj_id in _seen:
            return
        _seen.add(obj_id)
        
        obj_type = type(obj).__name__
        
        # Check if this object type has expected configs
        if obj_type in EXPECTED_CONFIGS:
            expected = EXPECTED_CONFIGS[obj_type]
            
            for attr_name, expected_type in expected.items():
                if not hasattr(obj, attr_name):
                    error_msg = f"{path}: Missing required config attribute '{attr_name}'"
                    errors.append(error_msg)
                    if raise_on_error:
                        raise ValueError(error_msg)
                    continue
                
                attr_value = getattr(obj, attr_name)
                if attr_value is None:
                    continue  # Allow None values
                
                actual_type_name = type(attr_value).__name__
                
                # Handle multiple allowed types
                if isinstance(expected_type, tuple):
                    if actual_type_name not in expected_type:
                        error_msg = (
                            f"{path}.{attr_name}: Expected one of {expected_type}, "
                            f"got {actual_type_name}"
                        )
                        errors.append(error_msg)
                        if raise_on_error:
                            raise TypeError(error_msg)
                else:
                    if actual_type_name != expected_type:
                        error_msg = (
                            f"{path}.{attr_name}: Expected {expected_type}, "
                            f"got {actual_type_name}"
                        )
                        errors.append(error_msg)
                        if raise_on_error:
                            raise TypeError(error_msg)
        
        # Recurse into nested objects (avoid infinite loops)
        try:
            attrs = vars(obj)
        except TypeError:
            return
        
        for attr_name, attr_value in attrs.items():
            if attr_name.startswith('_') or attr_name == 'logger':
                continue
            
            child_path = f"{path}.{attr_name}"
            
            if isinstance(attr_value, BaseConfigs):
                # Configs themselves might contain configs
                check_object(attr_value, child_path, _seen)
            elif hasattr(attr_value, '__dict__') and not isinstance(attr_value, (str, int, float, list, dict, tuple)):
                check_object(attr_value, child_path, _seen)
    
    check_object(root)
    return errors


def apply_and_validate_configs(
    root: Any,
    configs: Dict[str, dict],
    metadata: Dict[str, dict] = None,
    strict: bool = True,
    verify_paths: bool = True,
    validate_after: bool = True
) -> Dict[str, str]:
    """
    Apply configs with full safety checks (Strategies 1 + 4 combined).
    
    Args:
        root: The root object to apply configs to
        configs: Dict of {label: config_dict}
        metadata: Dict of {label: metadata_dict}
        strict: If True, raise errors on mismatches
        verify_paths: If True, verify paths match metadata
        validate_after: If True, run validation after applying
        
    Returns:
        Dict mapping labels to their paths
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> # Load from YAML
        >>> bundle = RunConfigBundle.load_from_yaml('configs.yaml')
        >>> 
        >>> # Apply with full validation
        >>> results = apply_and_validate_configs(
        ...     backtest, 
        ...     bundle.configs, 
        ...     bundle.metadata
        ... )
        >>> print(f"Applied {len(results)} configs successfully")
    """
    # Apply configs with path verification
    results = apply_run_configs(
        root, 
        configs, 
        metadata=metadata,
        strict=strict, 
        verify_paths=verify_paths
    )
    
    # Validate placement
    if validate_after:
        errors = validate_config_placement(root, raise_on_error=strict)
        if errors:
            error_msg = "Config validation failed after application:\n" + "\n".join(f"  - {err}" for err in errors)
            if strict:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
    
    return results

