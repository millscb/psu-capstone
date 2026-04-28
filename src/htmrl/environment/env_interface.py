"""Exposed interfaces for environment layer components."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvInterface(Protocol):
    """Defines the interface for environment layer components."""

    def step(self, action: Any) -> None:
        """Takes a step in the environment using the provided action.

        Args:
            action: The action to be taken in the environment.
        """
        ...

    def reset(self) -> None:
        """Reset the env to base state"""
        ...

    def render(self) -> None:
        """Renders the current state of the environment."""
        ...

    def close(self) -> None:
        """Cleans up any resources used by the environment."""
        ...
