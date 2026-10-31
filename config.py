"""Load data paths from config."""

import yaml
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.yaml"

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Defaults
    cfg.setdefault('data_root', '~/AstroData/Lights')
    cfg.setdefault('results_dir', '~/AstroData/results')
    cfg.setdefault('asiair_root', None)

    # Expand ~ and resolve
    for key in ('data_root', 'results_dir', 'asiair_root'):
        if cfg[key]:
            cfg[key] = Path(cfg[key]).expanduser().resolve()

    return cfg


def get_data_root() -> Path:
    cfg = load_config()
    root = cfg['data_root']
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_results_dir() -> Path:
    cfg = load_config()
    d = cfg['results_dir']
    d.mkdir(parents=True, exist_ok=True)
    return d