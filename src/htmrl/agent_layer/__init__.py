"""Public exports for the agent layer.

This package-level module exists to reduce import boilerplate in callers.
Consumers can import the layer once, for example import htmrl.agent_layer as ag,
and then reference ag.Brain or ag.Trainer instead of importing each symbol from
its concrete module.

The exports are loaded lazily to avoid import-time side effects. In particular,
Trainer pulls in plotting helpers and other heavier dependencies, so eager imports
here would make lightweight package imports unexpectedly expensive and brittle.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

HTM_MODULE = "htmrl.agent_layer.pullin.pullin_htm"

if TYPE_CHECKING:
    from htmrl.agent_layer.agent import Agent
    from htmrl.agent_layer.agent_interface import AgentInterface
    from htmrl.agent_layer.agent_runtime import AgentRuntimeConfig
    from htmrl.agent_layer.agent_server import AgentWebSocketServer
    from htmrl.agent_layer.pullin.pullin_brain import Brain
    from htmrl.agent_layer.pullin.pullin_htm import ColumnField, Field, InputField, OutputField
    from htmrl.agent_layer.train import Trainer

__all__ = [
    "Agent",
    "AgentRuntimeConfig",
    "AgentInterface",
    "AgentWebSocketServer",
    "Brain",
    "ColumnField",
    "Field",
    "InputField",
    "ValueField",
    "OutputField",
    "Trainer",
]


_EXPORTS = {
    "Agent": ("htmrl.agent_layer.agent", "Agent"),
    "AgentRuntimeConfig": ("htmrl.agent_layer.agent_runtime", "AgentRuntimeConfig"),
    "AgentInterface": ("htmrl.agent_layer.agent_interface", "AgentInterface"),
    "Brain": ("htmrl.agent_layer.pullin.pullin_brain", "Brain"),
    "ColumnField": (HTM_MODULE, "ColumnField"),
    "Field": (HTM_MODULE, "Field"),
    "Cell": (HTM_MODULE, "Cell"),
    "InputField": (HTM_MODULE, "InputField"),
    "OutputField": (HTM_MODULE, "OutputField"),
    "AgentWebSocketServer": ("htmrl.agent_layer.agent_server", "AgentWebSocketServer"),
    "Trainer": ("htmrl.agent_layer.train", "Trainer"),
}


def __getattr__(name: str) -> Any:
    """Resolve layer exports lazily to keep package imports lightweight.

    This preserves the convenient ag.Symbol API introduced by the import-cleanup
    refactor while avoiding eager imports of modules that have deeper dependency trees.
    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
