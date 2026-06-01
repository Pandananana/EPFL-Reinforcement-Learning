import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import PROJECT_ENVS, PPOConfig, get_config
from src.ppo import train_ppo
from .core import (
    JOINT_SPACE,
    JOINT_SPACE_HARD,
    OFAT_GRID,
    PAPER_DEFAULT,
    grid_search,
    load_summary,
    optuna_search,
    random_search,
    sweep_one_hparam,
    sweep_sensitivity_report,
    train_point,
)
from . import plots

__all__ = [
    "PROJECT_ENVS", "PPOConfig", "get_config", "train_ppo",
    "PAPER_DEFAULT", "OFAT_GRID", "JOINT_SPACE", "JOINT_SPACE_HARD",
    "sweep_one_hparam", "sweep_sensitivity_report",
    "random_search", "grid_search", "optuna_search",
    "train_point", "load_summary", "plots",
]
