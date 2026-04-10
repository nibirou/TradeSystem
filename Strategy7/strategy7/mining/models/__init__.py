"""Factor-mining model modules.

Each mining framework is implemented in a dedicated file so future frameworks
can be plugged in with one-file-per-model style.
"""

from .custom_model import run_custom_model
from .fundamental_multiobj_model import run_fundamental_multiobj_model
from .minute_parametric_model import run_minute_parametric_model
from .minute_parametric_plus_model import run_minute_parametric_plus_model
from .ml_ensemble_alpha_model import run_ml_ensemble_alpha_model
from .gplearn_symbolic_alpha_model import run_gplearn_symbolic_alpha_model

__all__ = [
    "run_custom_model",
    "run_fundamental_multiobj_model",
    "run_minute_parametric_model",
    "run_minute_parametric_plus_model",
    "run_ml_ensemble_alpha_model",
    "run_gplearn_symbolic_alpha_model",
]
