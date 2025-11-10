import os
import yaml


def load_config():
    """
    Load the configuration from the YAML file.
    
    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """
    with open(f"{os.environ['WORK_DIR']}/module_test/raw_code/optionlib/core/config.yaml", mode='r', encoding='utf-8') as file:
        _config = yaml.safe_load(file)
    return _config

config = load_config()
