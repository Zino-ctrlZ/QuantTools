# Config Management System

Complete guide for exporting, saving, loading, and applying configurations with full validation.

## Overview

The config management system provides:
1. **Export**: Collect all configs from an object graph
2. **Save**: Persist configs to YAML with metadata
3. **Load**: Read configs from YAML
4. **Apply**: Insert configs back with path verification (Strategy #1)
5. **Validate**: Ensure correct placement (Strategy #4)

## Quick Start

### 1. Export and Save Configs

```python
from EventDriven.configs.export_configs import export_run_configs

# Export all configs from backtest object
bundle = export_run_configs(evb_backtest, debug=False)

# Save to YAML file (includes metadata for validation)
bundle.save_to_yaml('my_backtest_config.yaml')

print(f"Saved {len(bundle.configs)} configs")
print(f"Run name: {bundle.run_name}")
```

### 2. Load and Apply Configs

```python
from EventDriven.configs.export_configs import RunConfigBundle

# Create a new backtest object
new_backtest = OptionSignalBacktest(trades_, initial_capital=cash)

# Load configs from YAML
bundle = RunConfigBundle.load_from_yaml('my_backtest_config.yaml')

# Apply configs with validation
results = bundle.apply_to(new_backtest, strict=True, verify_paths=True)

print(f"Applied {len(results)} configs successfully")
for label, path in results.items():
    print(f"  {label} -> {path}")
```

## Detailed Usage

### Collecting Configs

```python
from EventDriven.configs.export_configs import collect_run_configs

# Basic collection
configs = collect_run_configs(evb_backtest)
print(f"Found configs: {list(configs.keys())}")

# With metadata (includes path information)
configs, metadata = collect_run_configs(evb_backtest, include_metadata=True)

for label, location in metadata.items():
    print(f"{label}:")
    print(f"  Class: {location.config_class}")
    print(f"  Path: {location.path}")
    print(f"  Attribute: {location.attribute_name}")
```

### Modifying Configs

```python
# Collect configs
configs = collect_run_configs(evb_backtest)

# Modify as needed
configs["OrderSchemaConfigs_1"].target_dte = 365
configs["OrderSchemaConfigs_1"].min_moneyness = 0.70

# Changes are reflected immediately in the original object
# (configs are references to the actual config objects)
```

### Export with Metadata

```python
from EventDriven.configs.export_configs import export_run_configs

# Export creates a bundle with:
# - configs: dict of config data
# - metadata: dict of path/type information
bundle = export_run_configs(evb_backtest)

# Inspect metadata
for label, meta in bundle.metadata.items():
    print(f"{label}: {meta['path']}")

# Save to YAML (metadata included automatically)
bundle.save_to_yaml('configs_with_metadata.yaml')
```

### YAML File Format

The saved YAML file looks like:

```yaml
run_name: bkt_test_11
created_at: '2025-11-28T23:15:30.123456'
configs:
  BacktesterConfig_1:
    run_name: bkt_test_11
    raise_errors: true
    t_plus_n: 1
    max_slippage_pct: 0.125
    min_slippage_pct: 0.05
  OrderSchemaConfigs_1:
    run_name: bkt_test_11
    target_dte: 270
    strategy: vertical
    structure_direction: long
    spread_ticks: 1
    # ... more fields
metadata:
  BacktesterConfig_1:
    config_class: BacktesterConfig
    path: root.config
    parent_path: root
    attribute_name: config
  OrderSchemaConfigs_1:
    config_class: OrderSchemaConfigs
    path: root.risk_manager.order_picker.order_schema_configs
    parent_path: root.risk_manager.order_picker
    attribute_name: order_schema_configs
  # ... more metadata
```

### Apply with Different Strictness Levels

```python
# Strict mode: Raises errors on any mismatch
try:
    bundle.apply_to(new_backtest, strict=True, verify_paths=True)
except ValueError as e:
    print(f"Validation failed: {e}")

# Lenient mode: Logs warnings but continues
results = bundle.apply_to(new_backtest, strict=False, verify_paths=False)
print(f"Applied with warnings: {results}")
```

### Manual Application with Full Control

```python
from EventDriven.configs.export_configs import (
    apply_run_configs,
    apply_and_validate_configs
)

# Load bundle
bundle = RunConfigBundle.load_from_yaml('configs.yaml')

# Option 1: Apply with path verification
results = apply_run_configs(
    new_backtest,
    bundle.configs,
    bundle.metadata,
    strict=True,
    verify_paths=True
)

# Option 2: Apply with validation afterward
results = apply_and_validate_configs(
    new_backtest,
    bundle.configs,
    bundle.metadata,
    strict=True,
    verify_paths=True,
    validate_after=True  # Runs validation checks
)
```

### Validation Only

```python
from EventDriven.configs.export_configs import validate_config_placement

# Check if configs are correctly placed
errors = validate_config_placement(evb_backtest, raise_on_error=False)

if errors:
    print("Validation errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("All configs are correctly placed!")

# Raise on first error
validate_config_placement(evb_backtest, raise_on_error=True)
```

## Safety Mechanisms

### 1. Path Verification (Strategy #1)

Ensures configs go to the exact location they were exported from:

```python
# When you export configs, metadata stores the path
# e.g., "root.risk_manager.order_picker.order_schema_configs"

# When you apply, the system verifies:
# - Config label exists in new object
# - Config class matches (OrderSchemaConfigs)
# - Path matches expected location
# - No structural changes occurred
```

### 2. Type Checking

```python
# System checks:
# - Config class name matches expected type
# - Attribute types are correct
# - No accidental type swaps

# Example error:
# "Config class mismatch for OrderSchemaConfigs_1:
#   Expected: OrderSchemaConfigs
#   Got: ChainConfig
#   At path: root.risk_manager.order_picker.order_schema_configs"
```

### 3. Structural Validation (Strategy #4)

```python
# Validates known object patterns:
EXPECTED_CONFIGS = {
    'OptionSignalBacktest': {
        'config': 'BacktesterConfig',
    },
    'RiskManager': {
        'config': 'RiskManagerConfig',
    },
    # etc.
}

# Catches issues like:
# - Missing required config attributes
# - Wrong config type assigned
# - Structural changes in object graph
```

### 4. Path Mismatch Detection

```python
# Detects when object structure changed:
# "Config path mismatch for OrderSchemaConfigs_1:
#   Expected: root.risk_manager.order_picker.order_schema_configs
#   Got: root.risk_manager.new_picker.schema_config
#   This suggests the object structure changed."
```

## Common Workflows

### Workflow 1: Save Current Setup

```python
# After configuring backtest in notebook
confs = collect_run_configs(evb_backtest)
confs["OrderSchemaConfigs_1"].target_dte = 365
# ... more modifications

# Export and save
bundle = export_run_configs(evb_backtest)
bundle.save_to_yaml('production_config_v1.yaml')
```

### Workflow 2: Load Previous Setup

```python
# Create new backtest
backtest = OptionSignalBacktest(trades_, initial_capital=20000)

# Load previous config
bundle = RunConfigBundle.load_from_yaml('production_config_v1.yaml')
bundle.apply_to(backtest)

# Run with loaded config
backtest.run()
```

### Workflow 3: Compare Configs

```python
# Export from two different backtests
bundle1 = export_run_configs(backtest1)
bundle2 = export_run_configs(backtest2)

# Compare specific configs
for label in bundle1.configs:
    if label in bundle2.configs:
        cfg1 = bundle1.configs[label]
        cfg2 = bundle2.configs[label]
        
        for key in cfg1:
            if cfg1[key] != cfg2[key]:
                print(f"{label}.{key}: {cfg1[key]} != {cfg2[key]}")
```

### Workflow 4: Config Versioning

```python
# Save with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

bundle = export_run_configs(evb_backtest)
bundle.save_to_yaml(f'configs_{bundle.run_name}_{timestamp}.yaml')
```

## Error Handling

### Common Errors and Solutions

#### 1. Config Not Found

```python
# Error: "Config OrderSchemaConfigs_2 not found in object graph"
# Solution: Object has fewer instances of this config type
# - Check if object structure changed
# - Use strict=False to skip missing configs
```

#### 2. Type Mismatch

```python
# Error: "Config class mismatch for OrderSchemaConfigs_1:
#         Expected: OrderSchemaConfigs, Got: ChainConfig"
# Solution: Config was replaced with wrong type
# - Verify object initialization
# - Check for accidental overwrites
```

#### 3. Path Mismatch

```python
# Error: "Config path mismatch..."
# Solution: Object structure changed
# - Set verify_paths=False to ignore
# - Update configs to new structure
# - Re-export from new structure
```

#### 4. Missing Attributes

```python
# Error: "root.risk_manager: Missing required config attribute 'config'"
# Solution: Expected attribute not present
# - Check object initialization
# - Ensure all required configs are set
```

## Best Practices

1. **Always include metadata** when saving to YAML (done automatically)
2. **Use strict=True** during development to catch issues early
3. **Use strict=False** in production to handle graceful degradation
4. **Tag runs** before exporting for traceability
5. **Version your config files** with timestamps or git
6. **Validate after loading** to catch structural issues
7. **Keep configs in git** for reproducibility

## Integration with Existing Code

Your existing code pattern works perfectly:

```python
# Collect and modify (existing pattern)
confs = collect_run_configs(evb_backtest, debug=False)
confs["OrderSchemaConfigs_1"].target_dte = 270
# ... more modifications

# NEW: Save for later use
bundle = export_run_configs(evb_backtest)
bundle.save_to_yaml('my_setup.yaml')

# NEW: Load in another session
bundle = RunConfigBundle.load_from_yaml('my_setup.yaml')
bundle.apply_to(new_backtest)
```

## API Reference

### Functions

- `collect_run_configs(root, debug=False, include_metadata=False)` - Collect configs from object
- `export_run_configs(root, debug=False)` - Export configs with metadata to bundle
- `apply_run_configs(root, configs, metadata, strict=True, verify_paths=True)` - Apply configs with validation
- `validate_config_placement(root, raise_on_error=False)` - Validate config placement
- `apply_and_validate_configs(root, configs, metadata, ...)` - Apply + validate in one call
- `tag_run(root, run_name)` - Set run_name on all configs

### Classes

- `RunConfigBundle` - Container for configs + metadata
  - `save_to_yaml(filename)` - Save to YAML file
  - `load_from_yaml(filename)` - Load from YAML file (classmethod)
  - `apply_to(root, strict=True, verify_paths=True)` - Apply to object

- `ConfigLocation` - Metadata about config location
  - `label` - Config label (e.g., "OrderSchemaConfigs_1")
  - `config_class` - Class name
  - `path` - Full path in object graph
  - `parent_path` - Parent object path
  - `attribute_name` - Attribute name on parent
