"""
strategies/registry.py
-----------------------
Auto-discovery registry for all trading strategies.

DESIGN:
  Instead of hardcoding strategy names in app.py (which breaks every time
  a new strategy is added), this module:

  1. Scans every .py file under the strategies/ directory (recursively).
  2. Imports each module and finds every class that:
       - Is a subclass of BaseStrategy
       - Is NOT BaseStrategy itself (not abstract)
       - Has a public __init__ with inspectable parameters
  3. Builds a schema dict for each strategy:
       {
         "class_name": "EMACrossover",
         "display_name": "EMA Crossover (9/21)",
         "module_path": "strategies.base",
         "description": "...",
         "category": "...",       ← optional, from class attribute
         "params": [
           {"name": "fast_period", "type": "int", "default": 9, "min": None, "max": None},
           ...
         ],
       }
  4. Returns a dict keyed by class_name → schema.

  Result: dashboard automatically shows ALL strategies the moment a new
  strategy file is dropped into any strategies/ subdirectory and server
  restarts. Zero manual registration required.

USAGE:
  from strategies.registry import get_strategy_registry, load_strategy

  registry = get_strategy_registry()   # {name: schema_dict}
  strategy = load_strategy("EMACrossover", {"fast_period": 9, "slow_period": 21})

PARAMETER TYPE DETECTION:
  We inspect default values and type annotations via Python's inspect module.
  Rules:
    int default   → type "int"
    float default → type "float"
    str default   → type "select" (options from __init__ docstring or [default])
    bool default  → type "bool"
    None default  → type "any"
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

_HERE            = Path(__file__).resolve().parent
_PROJECT_ROOT    = _HERE.parent
_REGISTRY_CACHE: Optional[Dict[str, dict]] = None   # cached after first scan


def get_strategy_registry(force_refresh: bool = False) -> Dict[str, dict]:
    """
    Return the complete strategy registry. Results are cached after first call.

    Args:
        force_refresh: If True, re-scan all strategy files (use after adding
                       a new strategy at runtime).

    Returns:
        Dict keyed by class_name → schema dict.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None or force_refresh:
        _REGISTRY_CACHE = _build_registry()
    return _REGISTRY_CACHE


def load_strategy(class_name: str, params: Dict[str, Any]):
    """
    Instantiate a strategy by class name with the given parameters.

    Args:
        class_name: e.g. "EMACrossover"
        params:     e.g. {"fast_period": 9, "slow_period": 21}

    Returns:
        Strategy instance

    Raises:
        KeyError:   if class_name is not in registry
        TypeError:  if params are invalid for the strategy
    """
    registry = get_strategy_registry()
    if class_name not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"Strategy '{class_name}' not found. "
            f"Available strategies: {available}"
        )

    schema      = registry[class_name]
    module_path = schema["module_path"]

    # Import the module and get the class
    module = importlib.import_module(module_path)
    cls    = getattr(module, class_name)

    # Coerce param types to match what the constructor expects
    coerced = _coerce_params(params, schema["params"])

    try:
        return cls(**coerced)
    except TypeError as e:
        raise TypeError(
            f"Invalid parameters for {class_name}: {e}. "
            f"Expected parameters: {[p['name'] for p in schema['params']]}"
        ) from e


# =============================================================================
# Internal: build the full registry by scanning strategy files
# =============================================================================

def _build_registry() -> Dict[str, dict]:
    """
    Scan all Python files under strategies/ and discover BaseStrategy subclasses.
    """
    # Import BaseStrategy for isinstance checks
    try:
        from strategies.base import BaseStrategy
    except ImportError:
        logger.error("Cannot import BaseStrategy — registry will be empty.")
        return {}

    registry: Dict[str, dict] = {}

    # Find all .py files under strategies/ (recursive, excluding __pycache__)
    py_files = [
        p for p in _HERE.rglob("*.py")
        if "__pycache__" not in p.parts
        and p.name != "__init__.py"
        and p.name != "registry.py"
        and p.name != "base_strategy.py"   # exclude abstract base
    ]

    for py_file in sorted(py_files):
        module_path = _file_to_module_path(py_file)
        if module_path is None:
            continue

        try:
            module = _safe_import(module_path)
            if module is None:
                continue

            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Must be a concrete BaseStrategy subclass defined in THIS module
                if (obj is BaseStrategy
                        or not issubclass(obj, BaseStrategy)
                        or obj.__module__ != module_path):
                    continue

                # Skip abstract classes (those that still have abstractmethods)
                if inspect.isabstract(obj):
                    continue

                schema = _extract_schema(obj, module_path)
                if schema:
                    registry[name] = schema
                    logger.debug(f"  Discovered: {name} from {module_path}")

        except Exception as e:
            logger.warning(f"Error scanning {module_path}: {e}")

    logger.info(f"Strategy registry: {len(registry)} strategies discovered")
    return registry


def _file_to_module_path(py_file: Path) -> Optional[str]:
    """Convert a file path to a dotted module path relative to the project root."""
    try:
        rel = py_file.relative_to(_PROJECT_ROOT)
        # e.g. strategies/momentum/ema_crossover.py → strategies.momentum.ema_crossover
        parts = list(rel.with_suffix("").parts)
        return ".".join(parts)
    except ValueError:
        return None


def _safe_import(module_path: str):
    """Import a module by dotted path; return None on any failure."""
    # Ensure project root is in sys.path
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        logger.debug(f"  Cannot import {module_path}: {e}")
        return None


def _extract_schema(cls: Type, module_path: str) -> Optional[dict]:
    """
    Extract a parameter schema from a strategy class.

    Returns None if the class cannot be introspected.
    """
    try:
        sig  = inspect.signature(cls.__init__)
        params: List[dict] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue

            default = (
                None if param.default is inspect.Parameter.empty
                else param.default
            )
            annotation = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else type(default) if default is not None else None
            )

            param_schema = _build_param_schema(
                param_name, default, annotation, cls
            )
            params.append(param_schema)

        # Extract category from class attribute if present
        category = getattr(cls, "CATEGORY", None)
        if category is None:
            # Infer from module path
            if "momentum" in module_path or "trend" in module_path.lower():
                category = "Trend / Momentum"
            elif "reversion" in module_path or "mean" in module_path.lower():
                category = "Mean Reversion"
            elif "arbitrage" in module_path or "pairs" in module_path.lower():
                category = "Arbitrage / Pairs"
            else:
                category = "Other"

        # Try to get a clean display name by instantiating with defaults
        display_name = cls.__name__
        try:
            default_kwargs = {
                p["name"]: p["default"]
                for p in params
                if p["default"] is not None
            }
            temp_instance = cls(**default_kwargs)
            display_name  = temp_instance.name
        except Exception:
            pass

        return {
            "class_name":   cls.__name__,
            "display_name": display_name,
            "module_path":  module_path,
            "description":  inspect.cleandoc(cls.__doc__ or ""),
            "category":     category,
            "params":       params,
        }

    except Exception as e:
        logger.debug(f"  Cannot extract schema from {cls.__name__}: {e}")
        return None


def _build_param_schema(
    name: str,
    default: Any,
    annotation,
    cls: Type,
) -> dict:
    """Build a single parameter schema dict."""

    # Determine type string
    if annotation in (int,) or isinstance(default, int) and not isinstance(default, bool):
        type_str = "int"
    elif annotation in (float,) or isinstance(default, float):
        type_str = "float"
    elif annotation in (bool,) or isinstance(default, bool):
        type_str = "bool"
    elif annotation in (str,) or isinstance(default, str):
        type_str = "str"
    else:
        type_str = "any"

    # Infer sensible min/max from common parameter names
    min_val, max_val, step = _infer_bounds(name, default, type_str)

    # For string params, try to infer options from __init__ docstring or source
    options = None
    if type_str == "str":
        options = _infer_str_options(name, default, cls)
        if options:
            type_str = "select"

    schema = {
        "name":    name,
        "type":    type_str,
        "default": default,
        "min":     min_val,
        "max":     max_val,
        "step":    step,
    }
    if options:
        schema["options"] = options

    return schema


def _infer_bounds(name: str, default: Any, type_str: str):
    """
    Infer min/max/step bounds from parameter name heuristics.
    These are soft hints for the UI — they don't enforce correctness.
    """
    n = name.lower()
    if type_str not in ("int", "float"):
        return None, None, None

    # Period / lookback parameters
    if any(k in n for k in ("period", "lookback", "window", "bars")):
        return 2, 500, 1

    # Level / threshold parameters
    if any(k in n for k in ("level", "threshold", "oversold", "overbought")):
        return 1, 99, 1

    # Multiplier / std_dev
    if any(k in n for k in ("multiplier", "mult", "std_dev", "sigma")):
        return 0.5, 10.0, 0.5

    # Percentage
    if any(k in n for k in ("pct", "percent", "rate")):
        return 0.0, 100.0, 0.5

    # Fast/slow period naming convention
    if "fast" in n:
        return 2, 50, 1
    if "slow" in n:
        return 5, 500, 1

    # Generic fallback
    if isinstance(default, int):
        return 1, 500, 1
    if isinstance(default, float):
        return 0.0, 100.0, 0.5

    return None, None, None


def _infer_str_options(name: str, default: str, cls: Type) -> Optional[list]:
    """
    Try to find valid string options for a string parameter by inspecting
    __init__ source code for common patterns like:
      if mode not in ("reversion", "breakout")
    """
    try:
        source = inspect.getsource(cls.__init__)
        # Look for patterns like: ("opt1", "opt2", ...) or ("opt1", "opt2")
        import re
        # Match things like: ("reversion", "breakout") or ['reversion', 'breakout']
        pattern = r'[(\[]\s*"([^"]+)"(?:\s*,\s*"([^"]+)")+\s*[)\]]'
        for match in re.finditer(pattern, source):
            # Get all quoted strings near the param name
            full = match.group(0)
            opts = re.findall(r'"([^"]+)"', full)
            if default in opts:
                return opts
    except Exception:
        pass

    return [default] if default else None


def _coerce_params(params: Dict[str, Any], schema: List[dict]) -> Dict[str, Any]:
    """
    Coerce incoming parameter values to their correct Python types
    based on the schema. Handles the case where JSON sends everything
    as float (e.g. fast_period: 9.0 must become int 9).
    """
    coerced = {}
    schema_map = {p["name"]: p for p in schema}

    for k, v in params.items():
        if k in schema_map:
            ptype = schema_map[k]["type"]
            try:
                if ptype == "int":
                    coerced[k] = int(v)
                elif ptype == "float":
                    coerced[k] = float(v)
                elif ptype == "bool":
                    coerced[k] = bool(v)
                elif ptype in ("str", "select"):
                    coerced[k] = str(v)
                else:
                    coerced[k] = v
            except (ValueError, TypeError):
                coerced[k] = v   # pass through on coercion failure
        else:
            coerced[k] = v   # unknown param: pass through, let constructor handle

    return coerced
