import yaml
import os

config = None

def load_config():
    """
    Load the configuration from the YAML file.
    
    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """
    global config
    with open(f"{os.environ['WORK_DIR']}/module_test/raw_code/optionlib/core/config.yaml", "r") as file:
        config = yaml.safe_load(file)

load_config()