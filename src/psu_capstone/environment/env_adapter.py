"""Adapter between Gym environments and the Brain/Agent loop.

Think of this class as a translator sitting between two sides:

1. Gym side (normal RL calls):
    - reset() -> (obs, info)
    - step(action) -> (obs, reward, terminated, truncated, info)
2. Brain/Agent side (translated bridge dicts):
    - reset_bridge() -> {obs, inputs, info}
    - step_bridge() -> {obs, inputs, reward, terminated, truncated, info}

A bridge is just one dictionary that bundles:

- what Gym returned
- flattened inputs that are easy for the Brain to consume

Typical runtime flow:

1. caller creates EnvAdapter around a Gym env id or env instance
2. caller asks reset_bridge() for the first observation and HTM inputs
3. Brain consumes bridge["inputs"]
4. caller selects an action
5. caller asks step_bridge(action) for the next observation and HTM inputs
6. repeat until terminated or truncated is true

This keeps the adapter compatible with Gym while still giving the Brain the
simple key/value input map it expects.
"""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from psu_capstone.log import get_logger


class EnvAdapter(gym.Wrapper):
    """Wrap any Gymnasium environment and translate data for HTM usage.

    This class does not change the environment rules. It only:

    - forwards normal Gym calls
    - exposes space metadata
    - flattens observations/actions into Brain-friendly fields

    Args:
        gym_env: Gym environment id (for gym.make) or any pre-built
            gym.Env instance (for example FinGym(...)).
        reward_shaper: Optional callback used to transform the raw reward
            from ``step`` using ``(reward, terminated, truncated, obs)``.
        **gym_kwargs: Optional kwargs used only when gym_env is a
            string id. These are forwarded to gym.make(...).Example:
            EnvAdapter("CartPole-v1", render_mode="human").
            Do not pass these when gym_env is already a gym.Env object.

    Raises:
        ValueError: If ``gym_kwargs`` are provided when ``gym_env`` is already
            a ``gym.Env`` instance instead of a string id.
    """

    def __init__(
        self,
        gym_env: str | gym.Env = "CartPole-v1",
        reward_shaper: Callable[[float, bool, bool, Any], float] | None = None,
        **gym_kwargs: Any,
    ) -> None:
        # Constructor role in the adapter: ensure we always wrap one concrete
        # Gym env object, regardless of whether caller passed an id or instance.
        if isinstance(gym_env, str):
            # Allowed path: gym_env is an id, so create the env here and pass
            # constructor kwargs through to gym.make(...).
            # Example: EnvAdapter("CartPole-v1", render_mode="human").
            self._wrapped_env = gym.make(gym_env, **gym_kwargs)
        else:
            if gym_kwargs:
                # Invalid path: gym_env is already an env instance.
                # At this point, creation-time kwargs are too late to apply
                # because the env was already created before this adapter.
                # Correct usage is either:
                # 1) EnvAdapter("CartPole-v1", render_mode="human")
                # 2) EnvAdapter(existing_env)  # no gym_kwargs
                raise ValueError("gym_kwargs can only be used when gym_env is a string id.")
            # If the caller already built an env, just wrap it.
            self._wrapped_env = gym_env

        # Maintain compatibility with older runtime code that expects
        # adapters to expose the underlying Gym env as ``_env``.
        self._env = self._wrapped_env

        super().__init__(self._wrapped_env)

        # Internal observation cache for episode state.
        self._obs: Any | None = None
        self._logger = get_logger(self)

        # Optional reward shaper: (reward, terminated, truncated, obs) -> float
        self._reward_shaper = reward_shaper

    def _to_serializable(self, value: Any) -> Any:
        """Convert numpy values to JSON-friendly Python types.

        Call chain for runtime observation/action values:
            reset_bridge/step_bridge -> observation_to_inputs ->
            _space_value_to_inputs -> _flatten_value -> _to_serializable

        Call chain for space metadata values:
            get_observation_spec/get_action_spec -> _space_to_schema ->
            _to_serializable
        """

        # Local role: normalize data types so downstream code can treat values
        # as plain Python for flattening, logging, and metadata output.

        if isinstance(value, np.ndarray):
            # Arrays are not JSON-native; convert to nested Python lists.
            return value.tolist()
        if isinstance(value, np.generic):
            # NumPy scalar wrappers (np.float32, np.int64, etc.) -> Python scalar.
            return value.item()
        if isinstance(value, tuple):
            # Normalize tuples recursively so mixed nested outputs stay consistent.
            return [self._to_serializable(item) for item in value]
        if isinstance(value, list):
            # Normalize each list element recursively.
            return [self._to_serializable(item) for item in value]
        if isinstance(value, dict):
            # Normalize nested mappings used by Dict observation/action spaces.
            return {key: self._to_serializable(item) for key, item in value.items()}

        # Primitive Python values are already serializable.
        return value

    def _space_to_schema(self, space: gym.Space[Any]) -> dict[str, Any]:
        """Take a Gym space object and convert it into a plain, serializable description of that space.

        This does not return runtime values. It only describes shape/type
        details for debugging, introspection, or setup checks.
        """

        # Semantic role: describe the structure/constraints of a space as a schema.
        # Adapter-facing role: expose Gym space details in one predictable,
        # serializable dictionary structure for outside inspection.

        if isinstance(space, gym.spaces.Discrete):
            return {
                "space_type": "Discrete",
                "n": space.n,
                "values": list(range(space.n)),
            }

        if isinstance(space, gym.spaces.MultiDiscrete):
            nvec = space.nvec.tolist()
            return {
                "space_type": "MultiDiscrete",
                "nvec": nvec,
                "values": [list(range(int(n))) for n in nvec],
            }

        if isinstance(space, gym.spaces.MultiBinary):
            return {
                "space_type": "MultiBinary",
                "n": self._to_serializable(space.n),
                "values": [0, 1],
            }

        if isinstance(space, gym.spaces.Box):
            return {
                "space_type": "Box",
                "shape": list(space.shape),
                "dtype": str(space.dtype),
                "low": self._to_serializable(space.low),
                "high": self._to_serializable(space.high),
            }

        if isinstance(space, gym.spaces.Tuple):
            return {
                "space_type": "Tuple",
                "spaces": [self._space_to_schema(subspace) for subspace in space.spaces],
            }

        if isinstance(space, gym.spaces.Dict):
            return {
                "space_type": "Dict",
                "spaces": {
                    key: self._space_to_schema(subspace) for key, subspace in space.spaces.items()
                },
            }

        return {
            "space_type": type(space).__name__,
            "sample": self._to_serializable(space.sample()),
        }

    def _flatten_value(self, value: Any, prefix: str) -> dict[str, Any]:
        """Flatten nested values into a simple key/value map.

        Args:
            value: The value to flatten (dict, list, or scalar).
            prefix: Key prefix used to build namespaced output keys.

        Returns:
            A flat dictionary mapping prefixed string keys to scalar values.

        Example:
            a nested value can turn into keys like observation_0 or
            action_price_open.
        """

        # Normalize first so flattening only handles plain Python containers.
        serializable = self._to_serializable(value)

        # Local role: recursively turn nested values into flat key/value pairs
        # the Brain can consume as a simple input mapped dictionary.
        if isinstance(serializable, dict):
            flattened: dict[str, Any] = {}
            for key, item in serializable.items():
                flattened.update(self._flatten_value(item, f"{prefix}_{key}"))
            return flattened

        if isinstance(serializable, list):
            flattened = {}
            for index, item in enumerate(serializable):
                flattened.update(self._flatten_value(item, f"{prefix}_{index}"))
            return flattened

        return {prefix: serializable}

    def _space_value_to_inputs(
        self,
        space: gym.Space[Any],
        value: Any,
        prefix: str,
    ) -> dict[str, Any]:
        """Convert a value into flat Brain input fields.

        We walk Dict/Tuple spaces using the space definition so key names stay
        stable and predictable over time.
        """

        # Local role: walk the declared Gym space so output keys are stable.
        # Outside role: guarantees consistent field names across episodes.
        if isinstance(space, gym.spaces.Dict):
            flattened: dict[str, Any] = {}
            for key, subspace in space.spaces.items():
                flattened.update(self._space_value_to_inputs(subspace, value[key], str(key)))
            return flattened

        if isinstance(space, gym.spaces.Tuple):
            flattened = {}
            for index, subspace in enumerate(space.spaces):
                flattened.update(
                    self._space_value_to_inputs(subspace, value[index], f"{prefix}_{index}")
                )
            return flattened

        return self._flatten_value(value, prefix)

    def get_observation_spec(self) -> dict[str, Any]:
        """Get a generic description of the environment observation space."""

        # Outside role: lets Brain-side setup code to inspect expected input shape.
        return self._space_to_schema(self.observation_space)

    def get_action_spec(self) -> dict[str, Any]:
        """Get a generic description of the environment action space."""

        # Outside role: exposes action schema for policy/debug tooling.
        return self._space_to_schema(self.action_space)

    def observation_to_inputs(self, observation: Any) -> dict[str, Any]:
        """Convert one observation into the Brain input map.

        Most callers use this through reset_bridge and step_bridge.
        """

        # Outside role: converts one raw Gym observation into Brain fields.
        return self._space_value_to_inputs(self.observation_space, observation, "observation")

    def action_to_inputs(self, action: Any) -> dict[str, Any]:
        """Convert one action value into flat fields.

        Helpful when matching candidate actions against Brain predictions.
        """

        # Outside role: converts one action value into comparable flat fields.
        return self._space_value_to_inputs(self.action_space, action, "action")

    def get_observation(self) -> dict[str, Any]:
        """Get a generic description of the environment observation space."""

        # Compatibility role: preserve legacy wrapper shape for older callers.
        return {"observations": self.get_observation_spec()}

    def get_action(self) -> dict[str, Any]:
        """Get a generic description of the environment action space."""

        # Compatibility role: preserve legacy wrapper shape for older callers.
        return {"actions": self.get_action_spec()}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Run one normal Gym step.

        Use step_bridge if you also need Brain-ready flattened inputs.
        """

        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous-action envs expect ndarray-shaped actions.
            if np.isscalar(action):
                action = np.full(
                    self.action_space.shape, float(action), dtype=self.action_space.dtype
                )
            else:
                action = np.asarray(action, dtype=self.action_space.dtype)
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # Local role: run one Gym transition and normalize result types.
        self._obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs, float(reward), bool(terminated), bool(truncated), info

    def step_bridge(self, action: Any) -> dict[str, Any]:
        """Run one step and return the translated bridge dict.

        In plain terms: this is the handoff packet that bridges env-side data
        to Brain/Agent-side data.

        Normal Brain/Agent call order after choosing an action is:

        1. call step_bridge(action)
        2. read bridge["obs"] as the raw next observation
        3. read bridge["inputs"] as the Brain-ready input map
        4. inspect reward/done flags to decide whether the episode continues
        """

        # Outside role: this is the main per-step handoff from Gym to Brain.
        obs, reward, terminated, truncated, info = self.step(action)

        if self._reward_shaper is not None:
            reward = self._reward_shaper(reward, terminated, truncated, obs)

        action_inputs = self.action_to_inputs(action)

        self._logger.info(
            "[Layer:EnvAdapter] action=%s action_inputs=%s reward=%.6f terminated=%s truncated=%s",
            action,
            action_inputs,
            float(reward),
            bool(terminated),
            bool(truncated),
        )

        return {
            "obs": obs,
            "inputs": self.observation_to_inputs(obs),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Run one normal Gym reset.

        Use reset_bridge if you immediately need Brain-ready inputs.
        """

        # Local role: start a new Gym episode and cache the latest observation.
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def reset_bridge(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset and return the translated bridge dict.

        Typical episode-start flow:

        1. call reset_bridge()
        2. feed bridge["inputs"] into the Brain
        3. choose an action
        4. continue with step_bridge(action)
        """

        # Outside role: this is the episode-start handoff from Gym to Brain.
        obs, info = self.reset(seed=seed, options=options)

        return {"obs": obs, "inputs": self.observation_to_inputs(obs), "info": info}

    def render(self) -> Any:
        """Render using the wrapped environment's render behavior."""

        # Pass-through role: keep Gym rendering behavior unchanged.
        return self.env.render()

    def close(self) -> None:
        """Close the wrapped environment and release resources."""

        # Pass-through role: ensure env resources are released by Gym.
        self.env.close()
