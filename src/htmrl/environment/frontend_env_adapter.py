"""Adapter for client-driven frontend environments.

This adapter matches the observation/action schema used by the browser-side
environments. Unlike ``EnvAdapter``, it does not own a Python Gym environment.
Instead, it provides the small subset of adapter behavior the Agent and
WebSocket server need when the environment loop runs in the frontend.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class FrontendEnvAdapter:
    """Bridge browser-defined environments into the Agent/Brain stack.

    Args:
        env_name: Logical environment name for logging/debugging.
        observation_labels: Ordered labels expected from the frontend.
        action_count: Number of discrete actions the client environment accepts.
        initial_observation: Optional initial observation payload for reset.
    """

    def __init__(
        self,
        env_name: str,
        observation_labels: list[str],
        action_count: int,
        initial_observation: dict[str, float] | None = None,
    ) -> None:
        self._env_name = env_name
        self._observation_labels = list(observation_labels)
        self._initial_observation = initial_observation or {
            label: 0.0 for label in self._observation_labels
        }
        self._obs = dict(self._initial_observation)

        # Match the pieces of the Gym adapter API used by Agent.
        self.action_space = gym.spaces.Discrete(action_count)
        self._action_space = self.action_space
        self.observation_space = gym.spaces.Dict(
            {
                label: gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(),
                    dtype=float,
                )
                for label in self._observation_labels
            }
        )

    def get_observation_spec(self) -> dict[str, Any]:
        """Return a schema describing the frontend observation payload."""

        return {
            "space_type": "Dict",
            "keys": list(self._observation_labels),
        }

    def get_action_spec(self) -> dict[str, Any]:
        """Return a schema describing the discrete frontend action space."""

        return {
            "space_type": "Discrete",
            "n": self.action_space.n,
            "values": list(range(self.action_space.n)),
        }

    def observation_to_inputs(self, observation: Any) -> dict[str, Any]:
        """Normalize raw frontend observations into Brain input fields."""

        if isinstance(observation, dict):
            return {
                label: observation.get(label, self._initial_observation.get(label, 0.0))
                for label in self._observation_labels
            }

        if isinstance(observation, (list, tuple)):
            return {
                label: observation[index] if index < len(observation) else 0.0
                for index, label in enumerate(self._observation_labels)
            }

        if len(self._observation_labels) == 1:
            return {self._observation_labels[0]: observation}

        raise ValueError(
            "Frontend observation must be a dict keyed by labels or a sequence matching label order."
        )

    def action_to_inputs(self, action: Any) -> dict[str, Any]:
        """Convert one action value into flat fields for Brain-side comparison."""

        return {"action": action}

    def reset_bridge(self) -> dict[str, Any]:
        """Return an initial handoff packet for a client-driven episode."""

        self._obs = dict(self._initial_observation)
        return {
            "obs": dict(self._obs),
            "inputs": self.observation_to_inputs(self._obs),
            "info": {"env_name": self._env_name, "source": "frontend"},
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the frontend adapter state.

        Args:
            seed: Unused; present for API compatibility.
            options: Unused; present for API compatibility.

        Returns:
            The initial observation and info payload.
        """

        del seed, options
        bridge = self.reset_bridge()
        return bridge["obs"], bridge["info"]

    def step_bridge(self, action: Any) -> dict[str, Any]:
        """Reject server-driven stepping for frontend-owned environments."""

        raise RuntimeError(
            f"Environment '{self._env_name}' is frontend-driven; use websocket observation messages instead of server-side step(). Action was: {action!r}"
        )
