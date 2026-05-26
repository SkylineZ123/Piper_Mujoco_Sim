"""Load YAML configs from the configs/ directory."""
import os
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(name: str) -> dict:
    path = os.path.join(_ROOT, 'configs', name)
    with open(path) as f:
        return yaml.safe_load(f)
