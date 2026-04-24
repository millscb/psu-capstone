"""Agent orchestration for HTM-driven reinforcement learning.

This module defines the top-level Agent that owns the environment loop.
The Agent is responsible for:

- resetting the environment at episode boundaries
- converting each environment timestep into HTM Brain input
- asking the active policy to choose the next environment action
- stepping the environment forward
- passing transition data to future learning updates

The Brain processes observations, while the Agent decides how to turn the
current state and Brain outputs into a concrete environment action.
"""

import os
import random
from collections import defaultdict
from typing import Any, Literal

import numpy as np

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.pullin.sungur import ValueField
from psu_capstone.environment.env_adapter import EnvAdapter
from psu_capstone.log import get_logger


def _import_ppo() -> Any:
    """Import PPO lazily to avoid torch/cuda import side effects in non-PPO paths."""
    from stable_baselines3 import PPO

    return PPO


class Agent:
    """Coordinate Brain inference, policy selection, and environment interaction.

    The Agent is the runtime owner of the RL loop. On each timestep it:

    1. reads the current observation cached from the environment
    2. feeds the flattened observation inputs into the Brain
    3. selects an action using the active policy mode
    4. executes that action through the environment adapter
    5. stores the resulting next observation for the following timestep

    The Agent supports multiple action-selection strategies through
    ``policy_mode``. At the moment those are:

    - ``q_table``: epsilon-greedy action selection over a tabular value store
    - ``ppo``: action selection delegated to an injected PPO-style model
    - ``brain``: prefer Brain predictions when usable, otherwise fall back to
    the Q-table selector. RL/TD error propagation is handled via the Brain's
    ValueField(s) using the sungur_agent implementation.

    Args:
        brain: HTM Brain instance that consumes flattened observation inputs
            and produces predictions. RL policy updates use ValueField for TD error.
        adapter: Environment adapter that bridges Gym-style spaces and raw
            values into forms usable by the Agent and Brain.
        episodes: Number of training episodes to run when ``train`` is called.
        policy_mode: Active action-selection strategy. ``q_table`` uses
            epsilon-greedy lookup over cached values. ``ppo`` delegates action
            selection to a PPO-like model. ``brain`` attempts to rank actions
            using Brain predictions before falling back, and updates ValueField(s).
        ppo_policy: Stable-Baselines3 policy class name used when creating
            a PPO model internally.
        ppo_kwargs: Optional keyword arguments forwarded to
            ``stable_baselines3.PPO`` during model construction.
        ppo_deterministic: Passed to ``ppo_model.predict`` when policy mode is
            ``ppo``.

    Raises:
        TypeError: If ``policy_mode`` is ``'q_table'`` but the environment
            does not have a Discrete action space."""

    def __init__(
        self,
        brain: Brain | None = None,
        adapter: EnvAdapter | None = None,
        episodes: int = 500,
        policy_mode: Literal["q_table", "brain", "ppo"] = "ppo",
        ppo_policy: str = "MlpPolicy",
        ppo_kwargs: dict[str, Any] | None = None,
        ppo_deterministic: bool = True,
        force_brain_mode: bool = False,
        trainer: Any = None,
        config: Any = None,
    ) -> None:
        # --- Shared attributes ---
        self._adapter = adapter
        self._logger = get_logger(self)
        self._policy_mode = policy_mode
        self._brain_action_confidence_threshold = 0.1
        self._force_brain_mode = force_brain_mode
        self._q_state_decimals = 2
        self._training_episodes = episodes
        self._training_error = []
        self._obs = None
        self._inputs = {}
        # Use config.seed if available for reproducibility
        seed = getattr(config, "seed", None) if config is not None else None
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()
        self._last_action_source = "unknown"
        self._last_reward = 0.0
        # Keep q-table storage available across modes for compatibility checks.
        self._action_count: int | None = None
        self._q_values = defaultdict(lambda: np.array([], dtype=float))

        # --- Brain setup ---
        if brain is None and trainer is not None and adapter is not None and config is not None:
            self._brain = trainer.build_brain_for_env(adapter, config)
        elif brain is not None:
            self._brain = brain
        else:
            raise ValueError(
                "Agent requires either a brain or a trainer+adapter+config to build one."
            )

        # --- Policy-specific setup ---
        if self._policy_mode == "q_table":
            self._learning_rate = 0.1
            self._discount_factor = 0.99
            self._epsilon = 1.0
            self._epsilon_decay = 0.001
            # q_table policy only makes sense when actions can be indexed.
            action_spec = self._adapter.get_action_spec() if self._adapter else {}
            action_count = action_spec.get("n")
            is_integer_like = isinstance(action_count, (int, np.integer))
            if action_spec.get("space_type") != "Discrete" or not is_integer_like:
                raise TypeError("q_table policy requires a Discrete action space.")
            self._action_count = int(action_count) if is_integer_like else None
            self._q_values = defaultdict(self._new_q_row)
        elif self._policy_mode == "ppo":
            self._ppo_policy = ppo_policy
            self._ppo_kwargs = ppo_kwargs or {}
            self._ppo_deterministic = ppo_deterministic
            self._ppo_model = self._build_ppo_model()
        elif self._policy_mode == "brain":
            # Brain mode uses fallback logic, no extra setup needed here
            pass
        else:
            raise ValueError(f"Unknown policy_mode: {self._policy_mode}")

    def save(self, path=None):
        """Save the PPO model to disk using Stable-Baselines3's .save().

        Args:
            path: Optional path to save the PPO model. If not provided, uses the default PPO model path.
        """
        if self._policy_mode != "ppo":
            raise RuntimeError("save() is only supported for PPO policy mode.")
        if self._ppo_model is None:
            raise RuntimeError("No PPO model to save.")
        model_path = path or self._get_ppo_model_path()
        self._ppo_model.save(model_path)
        self._logger.info("PPO model saved to %s.", model_path)

    def load(self, path=None):
        """Load the PPO model from disk using Stable-Baselines3's .load().

        Args:
            path: Optional path to load the PPO model from. If not provided, uses the default PPO model path.
        """
        if self._policy_mode != "ppo":
            raise RuntimeError("load() is only supported for PPO policy mode.")
        model_path = path or self._get_ppo_model_path()
        env = self._adapter._env
        kwargs = {"device": "cpu"}
        if hasattr(self, "_ppo_kwargs"):
            kwargs.update(self._ppo_kwargs)
        ppo = _import_ppo()
        self._ppo_model = ppo.load(model_path, env=env, **kwargs)
        self._logger.info("PPO model loaded from %s.", model_path)

    def _get_ppo_model_path(self) -> str:
        """Return a unique PPO model path based on the environment id."""
        env_id = getattr(self._adapter, "env_id", None)
        if (
            env_id is None
            and hasattr(self._adapter, "_env")
            and hasattr(self._adapter._env, "spec")
        ):
            env_id = getattr(self._adapter._env.spec, "id", None)
        if env_id is None:
            env_id = "unknown_env"
        return f"ppo_model_{env_id}.zip"

    def _build_ppo_model(self) -> Any:
        """Create or load a Stable-Baselines3 PPO model bound to the adapter environment."""

        if not hasattr(self._adapter, "_env"):
            raise ValueError("ppo policy mode requires an adapter with a Gym environment (_env).")

        env = self._adapter._env
        kwargs = {"device": "cpu", **self._ppo_kwargs}
        ppo = _import_ppo()
        model_path = self._get_ppo_model_path()
        if os.path.exists(model_path):
            self._logger.info(f"Loading pre-trained PPO model from {model_path}")
            return ppo.load(model_path, env=env, **kwargs)
        self._logger.info(
            f"No pre-trained PPO model found for env {model_path}; creating new PPO model."
        )
        return ppo(self._ppo_policy, env, verbose=0, **kwargs)

    def _select_ppo_action(self, obs: Any) -> Any:
        """Select an action from the configured PPO-style policy model."""

        if self._ppo_model is None:
            self._ppo_model = self._build_ppo_model()

        action, _state = self._ppo_model.predict(obs, deterministic=self._ppo_deterministic)
        self._logger.info("PPO observation: %s | PPO selected action: %s", obs, action)
        self._last_action_source = "ppo"
        if isinstance(action, np.ndarray) and action.shape == ():
            return action.item()
        return action

    def train_ppo(self, total_timesteps: int = 50_000) -> None:
        """Train the PPO model for the given number of environment timesteps and save it per environment.

        Args:
            total_timesteps: Total environment steps to train for.
                50 000 is enough for CartPole-v1 to converge reliably.
        """

        if self._policy_mode != "ppo":
            raise RuntimeError("train_ppo() called but policy_mode is not 'ppo'.")
        if self._ppo_model is None:
            self._ppo_model = self._build_ppo_model()
        self._logger.info("Training PPO for %d timesteps...", total_timesteps)

        # Use self._logger to log PPO training progress directly
        self._ppo_model.learn(total_timesteps=total_timesteps)
        self._logger.info("PPO training complete. Saving model...")
        model_path = self._get_ppo_model_path()
        self._ppo_model.save(model_path)
        self._logger.info(f"PPO model saved to {model_path}.")

    def switch_policy(self, new_mode: Literal["q_table", "brain", "ppo"]) -> None:
        """Switch the active policy mode at runtime.

        Args:
            new_mode: The policy mode to switch to.
        """
        self._logger.info("Switching policy from '%s' to '%s'.", self._policy_mode, new_mode)
        self._policy_mode = new_mode

    def _new_q_row(self) -> np.ndarray:
        """Create a zero-initialized action-value row for one state."""

        if self._action_count is None:
            return np.array([], dtype=float)

        return np.zeros(self._action_count, dtype=float)

    def _state_key(self, obs: Any) -> tuple[tuple[str, Any], ...]:
        """Convert raw observations into a hashable key for tabular lookup."""

        # Use adapter-normalized inputs so state identity matches Brain-facing features.
        obs_inputs = self._adapter.observation_to_inputs(obs)
        quantized_inputs = {
            key: (
                round(float(value), self._q_state_decimals)
                if isinstance(value, (int, float, np.integer, np.floating))
                else value
            )
            for key, value in obs_inputs.items()
        }
        return tuple(sorted(quantized_inputs.items()))

    def reset_episode(self) -> dict[str, Any]:
        """Reset the environment, then cache the initial observation.

        Returns:
            Adapter reset bridge containing at least the raw observation,
            flattened HTM inputs, and environment info.
        """

        result = self._adapter.reset_bridge()
        self._obs = result["obs"]
        self._inputs = result["inputs"]
        self._last_reward = 0.0
        return result

    def _initialize_q_values(self, obs: Any) -> tuple[tuple[str, Any], ...]:
        """Ensure an observation has an entry in the tabular value store."""

        state_key = self._state_key(obs)

        # Touching the defaultdict lazily allocates a zero row for unseen states.
        _ = self._q_values[state_key]
        return state_key

    def _action_index(self, action: Any) -> int | None:
        """Resolve an action value to its tabular Q-row index when possible."""

        if isinstance(action, (int, np.integer)) and self._action_count is not None:
            index = int(action)
            if 0 <= index < self._action_count:
                return index

        candidate_actions = self._candidate_actions()
        if not candidate_actions:
            return None

        for index, candidate in enumerate(candidate_actions):
            if candidate == action:
                return index

        return None

    def _candidate_actions(self) -> list[Any]:
        """Enumerate actions when the adapter exposes a finite action list.

        Returns:
            A flat list of candidate environment actions for action spaces that
            can be enumerated directly. Returns an empty list for spaces that
            are continuous or otherwise not represented as a single flat list of
            values.
        """

        action_spec = self._adapter.get_action_spec()
        values = action_spec.get("values")

        if isinstance(values, list) and all(not isinstance(item, list) for item in values):
            return values

        return []

    def _get_action_space(self) -> Any | None:
        """Return adapter action space across old/new adapter attribute names."""

        action_space = getattr(self._adapter, "_action_space", None)
        if action_space is not None:
            return action_space
        return getattr(self._adapter, "action_space", None)

    def _sample_action(self) -> Any:
        """Sample a valid action from the adapter action space."""

        action_space = self._get_action_space()
        if action_space is None or not hasattr(action_space, "sample"):
            raise ValueError("Adapter does not expose a sample-able action space.")
        return action_space.sample()

    def _select_q_action(self, obs: Any) -> Any:
        """Select an action using epsilon-greedy Q-table behavior.

        This strategy prefers exploration while ``epsilon`` is high, then
        gradually shifts toward exploitation of the best known action value for
        the current observation.

        Args:
            obs: Current environment observation.

        Returns:
            A concrete action compatible with the environment action space.
        """

        state_key = self._initialize_q_values(obs)

        if self._epsilon > 0.1:
            self._epsilon -= self._epsilon_decay

        candidate_actions = self._candidate_actions()

        if not candidate_actions:
            self._last_action_source = "q_table"
            return self._sample_action()

        # Epsilon-greedy action selection
        if self._rng.random() < self._epsilon:
            self._last_action_source = "q_table"
            return self._sample_action()

        q_row = self._q_values[state_key]

        # If spec/value mismatch occurs, degrade safely to random exploration.
        if len(q_row) != len(candidate_actions):
            self._last_action_source = "q_table"
            return self._sample_action()

        max_q = float(np.max(q_row))
        best_indices = [index for index, value in enumerate(q_row) if float(value) == max_q]
        best_index = self._rng.choice(best_indices)
        self._last_action_source = "q_table"
        return candidate_actions[best_index]

    def select_action(self, obs: Any, brain_outputs: dict[str, Any] | None = None) -> Any:
        """Select an action using the current policy mode only (no fallback chains, all logic in this method)."""
        if self._policy_mode == "brain":
            # --- Brain policy logic ---
            action = None
            if isinstance(brain_outputs, dict):
                # Try direct action hint from outputs
                for _name, payload in brain_outputs.items():
                    if isinstance(payload, dict) and "action" in payload:
                        confidence = payload.get("confidence", 1.0)
                        confidence_value = (
                            float(confidence) if isinstance(confidence, (int, float)) else 1.0
                        )
                        if (
                            confidence_value >= self._brain_action_confidence_threshold
                            or self._force_brain_mode
                        ):
                            self._last_action_source = "brain"
                            if (
                                self._force_brain_mode
                                and confidence_value < self._brain_action_confidence_threshold
                            ):
                                self._logger.info(
                                    "force_brain_mode: using brain action with low confidence %.3f.",
                                    confidence_value,
                                )
                            action = payload["action"]
                            break
            if action is None:
                candidate_actions = self._candidate_actions()
                action_space = self._get_action_space()
                # If no candidate actions, handle continuous (Box) action space
                if (
                    not candidate_actions
                    and action_space is not None
                    and hasattr(action_space, "sample")
                ):
                    # Try to use Brain output if available
                    if isinstance(brain_outputs, dict):
                        # Look for a continuous action output
                        for _name, payload in brain_outputs.items():
                            if isinstance(payload, dict) and "action" in payload:
                                val = payload["action"]
                                # If action space is Box, clip to valid range
                                if hasattr(action_space, "low") and hasattr(action_space, "high"):
                                    val = np.clip(val, action_space.low, action_space.high)
                                action = val
                                self._last_action_source = "brain"
                                break
                    if action is None:
                        # Fallback: sample a valid action from the space
                        action = action_space.sample()
                        self._last_action_source = "brain"
                    # Ensure action is correct shape for Box spaces
                    if (
                        hasattr(action_space, "shape")
                        and hasattr(action_space, "dtype")
                        and hasattr(action_space, "low")
                        and hasattr(action_space, "high")
                    ):
                        # Handle all possible scalar/int/float/array cases
                        if isinstance(action, (int, float, np.integer, np.floating)):
                            action = np.full(action_space.shape, action, dtype=action_space.dtype)
                        else:
                            action = np.asarray(action, dtype=action_space.dtype)
                            if action.shape == ():
                                action = np.full(
                                    action_space.shape, action.item(), dtype=action_space.dtype
                                )
                            elif action.shape != action_space.shape:
                                try:
                                    action = action.reshape(action_space.shape)
                                except Exception:
                                    action = np.broadcast_to(action, action_space.shape).astype(
                                        action_space.dtype
                                    )
                        action = np.clip(action, action_space.low, action_space.high)
                elif candidate_actions:
                    predictions = self._brain.prediction()
                    if not isinstance(predictions, dict):
                        self._logger.info("Brain predictions unavailable; choosing random action.")
                        action = self._sample_action()
                    else:
                        action_predictions = {
                            key: value
                            for key, value in predictions.items()
                            if key.startswith("action")
                        }
                        if not action_predictions:
                            self._logger.info(
                                "No action-like brain predictions; choosing random action."
                            )
                            action = self._sample_action()
                        else:

                            def score_predicted_action(action):
                                action_inputs = self._adapter.action_to_inputs(action)
                                score = 0.0
                                for key, value in action_inputs.items():
                                    if (
                                        key not in action_predictions
                                        or action_predictions[key] is None
                                    ):
                                        continue
                                    prediction = action_predictions[key]
                                    confidence = action_predictions.get(f"{key}.conf", 1.0)
                                    weight = (
                                        float(confidence)
                                        if isinstance(confidence, (int, float))
                                        else 1.0
                                    )
                                    if isinstance(prediction, (int, float)) and isinstance(
                                        value, (int, float)
                                    ):
                                        score -= abs(float(prediction) - float(value)) * weight
                                    else:
                                        score += weight if prediction == value else 0.0
                                return score

                            action = max(candidate_actions, key=score_predicted_action)
                            self._last_action_source = "brain"
        elif self._policy_mode == "ppo":
            # --- PPO policy logic ---
            if self._ppo_model is None:
                self._ppo_model = self._build_ppo_model()
            action, _state = self._ppo_model.predict(obs, deterministic=self._ppo_deterministic)
            self._logger.info("PPO observation: %s | PPO selected action: %s", obs, action)
            self._last_action_source = "ppo"
            if isinstance(action, np.ndarray) and action.shape == ():
                action = action.item()
        elif self._policy_mode == "q_table":
            # --- Q-table policy logic ---
            state_key = self._initialize_q_values(obs)
            if self._epsilon > 0.1:
                self._epsilon -= self._epsilon_decay
            candidate_actions = self._candidate_actions()
            if not candidate_actions:
                self._last_action_source = "q_table"
                action = self._sample_action()
            elif self._rng.random() < self._epsilon:
                self._last_action_source = "q_table"
                action = self._sample_action()
            else:
                q_row = self._q_values[state_key]
                if len(q_row) != len(candidate_actions):
                    self._last_action_source = "q_table"
                    action = self._sample_action()
                else:
                    max_q = float(np.max(q_row))
                    best_indices = [
                        index for index, value in enumerate(q_row) if float(value) == max_q
                    ]
                    best_index = self._rng.choice(best_indices)
                    self._last_action_source = "q_table"
                    action = candidate_actions[best_index]
        else:
            self._logger.warning(
                f"Unknown policy_mode: {self._policy_mode}, choosing random action."
            )
            action = self._sample_action()
        self._logger.info(
            "Policy step: mode=%s source=%s action=%s",
            self._policy_mode,
            self._last_action_source,
            action,
        )

        # --- FINAL Box action enforcement for all selection paths ---
        action_space = self._get_action_space()
        if (
            action_space is not None
            and hasattr(action_space, "shape")
            and hasattr(action_space, "dtype")
            and hasattr(action_space, "low")
            and hasattr(action_space, "high")
        ):
            if isinstance(action, (int, float, np.integer, np.floating)):
                action = np.full(action_space.shape, action, dtype=action_space.dtype)
            else:
                action = np.asarray(action, dtype=action_space.dtype)
                if action.shape == ():
                    action = np.full(action_space.shape, action.item(), dtype=action_space.dtype)
                elif action.shape != action_space.shape:
                    try:
                        action = action.reshape(action_space.shape)
                    except Exception:
                        action = np.broadcast_to(action, action_space.shape).astype(
                            action_space.dtype
                        )
            action = np.clip(action, action_space.low, action_space.high)

        return action

    def step(self, learn: bool = True) -> dict[str, Any]:
        """Run one complete agent-controlled environment timestep.

        The current cached observation is first converted into Brain inputs and
        processed by ``Brain.step``. The Agent then selects an environment
        action, applies it through the adapter, receives the resulting
        transition, and caches the next observation for the following call.

        Args:
            learn: Passed through to ``Brain.step`` to enable or disable Brain
                learning for this timestep.

        Returns:
            A transition dictionary containing the current observation and
            inputs, chosen action, next observation and inputs, reward, done
            flags, and environment info.

        Raises:
            RuntimeError: If no observation is available after attempting to
                initialize the episode state.
        """

        if self._obs is None:
            # First step in an episode initializes cached observation/input state.
            self.reset_episode()

        current_obs = self._obs
        current_inputs = self._inputs

        if current_obs is None:
            raise RuntimeError("Agent observation state was not initialized.")

        step_inputs = dict(current_inputs)
        step_inputs.setdefault("reward", float(self._last_reward))

        brain_outputs = self._brain.step(step_inputs, learn=learn)
        action = self.select_action(current_obs, brain_outputs=brain_outputs)
        result = self._adapter.step_bridge(action)

        next_obs = result["obs"]
        reward = result["reward"]
        if self._policy_mode == "ppo":
            self._logger.info(
                "PPO step: obs=%s, action=%s, reward=%s, next_obs=%s",
                current_obs,
                action,
                reward,
                next_obs,
            )
        elif self._policy_mode == "brain":
            self._logger.info(
                "Brain step: obs=%s, action=%s, reward=%s, next_obs=%s",
                current_obs,
                action,
                reward,
                next_obs,
            )
        elif self._policy_mode == "q_table":
            self._logger.info(
                "Q-table step: obs=%s, action=%s, reward=%s, next_obs=%s",
                current_obs,
                action,
                reward,
                next_obs,
            )
        self._last_reward = float(reward)

        self.update(
            current_obs,
            action,
            reward,
            next_obs,
            done=result["terminated"] or result["truncated"],
        )

        # Cache the next timestep state so the next Agent.step() can continue the loop.
        self._obs = next_obs
        self._inputs = result["inputs"]

        return {
            "obs": current_obs,
            "inputs": current_inputs,
            "action": action,
            "next_obs": next_obs,
            "next_inputs": result["inputs"],
            "reward": reward,
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "info": result["info"],
        }

    def update(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool = False,
    ) -> None:
        """Update policy state from a transition.

        - ``q_table``: apply one-step TD update.
        - ``ppo``: no update here; PPO training is managed separately.
        - ``brain``: keep the current RL update stub in place. If this step
            actually used q-table fallback, run the q-table TD update so those
            fallback actions can still learn tabular values.

        Args:
            obs: Observation before the action was taken.
            action: Environment action selected for the timestep.
            reward: Reward returned by the environment.
            next_obs: Observation returned after the transition.
            done: Whether the transition ended the episode.
        """

        if self._policy_mode == "ppo":
            # PPO model training lifecycle is managed externally.
            return

        if self._policy_mode == "brain":
            if self._last_action_source == "q_table":
                # In brain mode, allow tabular updates when action selection
                # explicitly fell back to q_table behavior.
                pass
            else:
                has_value_field = any(
                    isinstance(field, ValueField)
                    for field in getattr(self._brain, "fields", {}).values()
                )
                if has_value_field:
                    self._brain.rl_policy_update(reward)
                return

        if self._action_count is None:
            return

        action_index = self._action_index(action)
        if action_index is None:
            return

        state_key = self._initialize_q_values(obs)
        next_state_key = self._initialize_q_values(next_obs)

        current_q = float(self._q_values[state_key][action_index])
        next_best_q = 0.0 if done else float(np.max(self._q_values[next_state_key]))

        td_target = float(reward) + (self._discount_factor * next_best_q)
        td_error = td_target - current_q

        self._q_values[state_key][action_index] = current_q + (self._learning_rate * td_error)
        self._training_error.append(abs(td_error))

    def train(self) -> None:
        """Run the configured number of episodes using agent-owned stepping.

        Each episode is reset through ``reset_episode`` and advanced by calling
        ``step`` until the environment reports termination or truncation.
        """

        for _episode in range(self._training_episodes):
            self.reset_episode()
            done = False

            while not done:
                transition = self.step(learn=True)
                done = transition["terminated"] or transition["truncated"]
