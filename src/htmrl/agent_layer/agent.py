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

import random
from collections import defaultdict
from typing import Any, Literal

import numpy as np

from htmrl.agent_layer.pullin.pullin_brain import Brain
from htmrl.environment.env_adapter import EnvAdapter
from htmrl.log import get_logger


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
      the Q-table selector

    Args:
        brain: HTM Brain instance that consumes flattened observation inputs
            and produces predictions.
        adapter: Environment adapter that bridges Gym-style spaces and raw
            values into forms usable by the Agent and Brain.
        episodes: Number of training episodes to run when ``train`` is called.
        policy_mode: Active action-selection strategy. ``q_table`` uses
            epsilon-greedy lookup over cached values. ``ppo`` delegates action
            selection to a PPO-like model. ``brain`` attempts to rank actions
            using Brain predictions before falling back.
        ppo_policy: Stable-Baselines3 policy class name used when creating
            a PPO model internally.
        ppo_kwargs: Optional keyword arguments forwarded to
            ``stable_baselines3.PPO`` during model construction.
        ppo_deterministic: Passed to ``ppo_model.predict`` when policy mode is
            ``ppo``.
        q_learning_rate: Learning rate used by tabular Q updates.
        q_discount_factor: Discount factor used by TD targets.
        epsilon_start: Initial epsilon value for epsilon-greedy exploration.
        epsilon_decay: Epsilon decay applied after each Q-table action selection.
        reward_scale: Multiplier applied to rewards before update logic runs.

    Raises:
        TypeError: If ``policy_mode`` is ``'q_table'`` but the environment
            does not have a Discrete action space.
    """

    def __init__(
        self,
        brain: Brain,
        adapter: EnvAdapter,
        episodes: int = 1000,
        policy_mode: Literal["q_table", "brain", "ppo"] = "q_table",
        ppo_policy: str = "MlpPolicy",
        ppo_kwargs: dict[str, Any] | None = None,
        ppo_deterministic: bool = True,
        q_learning_rate: float = 0.1,
        q_discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.01,
        reward_scale: float = 1.0,
    ) -> None:
        self._learning_rate: float = q_learning_rate
        self._discount_factor: float = q_discount_factor
        self._epsilon: float = epsilon_start
        self._epsilon_decay: float = epsilon_decay
        self._reward_scale: float = reward_scale

        self._brain = brain
        self._adapter = adapter
        self._policy_mode: Literal["q_table", "brain", "ppo"] = policy_mode
        self._ppo_policy = ppo_policy
        self._ppo_kwargs = ppo_kwargs or {}
        self._ppo_model: Any | None = None
        self._ppo_deterministic = ppo_deterministic
        self._logger = get_logger(self)
        self._logger.info(
            "[Layer:Agent] update params lr=%.4f gamma=%.4f epsilon_start=%.4f epsilon_decay=%.4f reward_scale=%.4f",
            self._learning_rate,
            self._discount_factor,
            self._epsilon,
            self._epsilon_decay,
            self._reward_scale,
        )

        # q_table policy only makes sense when actions can be indexed.
        action_spec = self._adapter.get_action_spec()
        action_count = action_spec.get("n")
        is_integer_like = isinstance(action_count, (int, np.integer))
        if self._policy_mode == "q_table" and (
            action_spec.get("space_type") != "Discrete" or not is_integer_like
        ):
            raise TypeError("q_table policy requires a Discrete action space.")

        self._action_count: int | None = int(action_count) if is_integer_like else None
        self._q_values: defaultdict[Any, np.ndarray] = defaultdict(self._new_q_row)

        self._training_episodes: int = episodes
        self._training_error: list[float] = []
        self._obs: Any | None = None
        self._inputs: dict[str, Any] = {}
        self._rng = random.Random()

        if self._policy_mode == "ppo":
            self._ppo_model = self._build_ppo_model()

    def _build_ppo_model(self) -> Any:
        """Create a Stable-Baselines3 PPO model bound to the adapter environment."""

        if not hasattr(self._adapter, "_env"):
            raise ValueError("ppo policy mode requires an adapter with a Gym environment (_env).")

        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for policy_mode='ppo'. "
                "Install with: uv sync --extra rl"
            ) from e

        env = self._adapter._env
        return PPO(self._ppo_policy, env, verbose=0, **self._ppo_kwargs)

    def _select_ppo_action(self, obs: Any) -> Any:
        """Select an action from the configured PPO-style policy model."""

        if self._ppo_model is None:
            self._ppo_model = self._build_ppo_model()

        action, _state = self._ppo_model.predict(obs, deterministic=self._ppo_deterministic)
        if isinstance(action, np.ndarray) and action.shape == ():
            return action.item()
        return action

    def _new_q_row(self) -> np.ndarray:
        """Create a zero-initialized action-value row for one state."""

        if self._action_count is None:
            return np.array([], dtype=float)

        return np.zeros(self._action_count, dtype=float)

    def _state_key(self, obs: Any) -> tuple[tuple[str, Any], ...]:
        """Convert raw observations into a hashable key for tabular lookup."""

        # Use adapter-normalized inputs so state identity matches Brain-facing features.
        obs_inputs = self._adapter.observation_to_inputs(obs)
        return tuple(sorted(obs_inputs.items()))

    def reset_episode(self) -> dict[str, Any]:
        """Reset the environment, then cache the initial observation.

        Returns:
            Adapter reset bridge containing at least the raw observation,
            flattened HTM inputs, and environment info.
        """

        result = self._adapter.reset_bridge()
        self._obs = result["obs"]
        self._inputs = result["inputs"]
        self._logger.info("[Layer:Agent] episode reset complete")
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
            return self._adapter.action_space.sample()

        # Epsilon-greedy action selection
        if self._rng.random() < self._epsilon:
            return self._adapter.action_space.sample()

        q_row = self._q_values[state_key]

        # If spec/value mismatch occurs, degrade safely to random exploration.
        if len(q_row) != len(candidate_actions):
            return self._adapter.action_space.sample()

        best_index = int(np.argmax(q_row))
        return candidate_actions[best_index]

    def _score_predicted_action(
        self,
        action_inputs: dict[str, Any],
        predictions: dict[str, Any],
    ) -> float:
        """Score how well a candidate action aligns with Brain predictions.

        Numeric predictions are scored by distance, optionally weighted by a
        ``<field>.conf`` confidence value. Non-numeric predictions receive a
        positive score when they match exactly.

        Args:
            action_inputs: Flattened representation of one candidate action.
            predictions: Brain prediction mapping, including optional
                confidence entries.

        Returns:
            A larger score for better alignment with the predicted action-like
            outputs.
        """

        score = 0.0

        for key, value in action_inputs.items():
            if key not in predictions or predictions[key] is None:
                continue

            prediction = predictions[key]
            confidence = predictions.get(f"{key}.conf", 1.0)
            weight = float(confidence) if isinstance(confidence, (int, float)) else 1.0

            if isinstance(prediction, (int, float)) and isinstance(value, (int, float)):
                score -= abs(float(prediction) - float(value)) * weight
            else:
                score += weight if prediction == value else 0.0

        return score

    def _select_brain_action(self, obs: Any) -> Any:
        """Select an action using Brain predictions when available.

        This strategy asks the Brain for predictions, filters for fields that
        look like action outputs, and ranks candidate actions against those
        predictions. If the Brain does not expose usable action predictions,
        selection falls back to the Q-table policy.

        Args:
            obs: Current environment observation.

        Returns:
            A concrete environment action.
        """

        return self._select_brain_action_from_outputs(obs, brain_outputs=None)

    def _extract_action_from_brain_outputs(self, brain_outputs: dict[str, Any]) -> Any | None:
        """Extract a direct action hint from Brain.step output payloads."""

        for _name, payload in brain_outputs.items():
            if isinstance(payload, dict) and "action" in payload:
                return payload["action"]

        return None

    def _select_brain_action_from_outputs(
        self,
        obs: Any,
        brain_outputs: dict[str, Any] | None,
    ) -> Any:
        """Select action from output-field hints, then prediction scoring, then q-table fallback."""

        if isinstance(brain_outputs, dict):
            action_hint = self._extract_action_from_brain_outputs(brain_outputs)
            if action_hint is not None:
                return action_hint

        candidate_actions = self._candidate_actions()

        if not candidate_actions:
            return self._adapter._action_space.sample()

        predictions = self._brain.prediction()
        if not isinstance(predictions, dict):
            # Current Brain API is still evolving; use q_table as a robust fallback.
            return self._select_q_action(obs)

        action_predictions = {
            key: value for key, value in predictions.items() if key.startswith("action")
        }

        if not action_predictions:
            # No action-like prediction keys yet, so use the baseline selector.
            return self._select_q_action(obs)

        return max(
            candidate_actions,
            key=lambda action: self._score_predicted_action(
                self._adapter.action_to_inputs(action),
                action_predictions,
            ),
        )

    def select_action(self, obs: Any, brain_outputs: dict[str, Any] | None = None) -> Any:
        """Dispatch action selection to the active policy implementation.

        Args:
            obs: Current environment observation.
            brain_outputs: Optional dict of Brain prediction outputs used when
                ``policy_mode`` is ``'brain'``.

        Returns:
            The environment action chosen by the active policy mode.
        """

        if self._policy_mode == "brain":
            return self._select_brain_action_from_outputs(obs, brain_outputs)

        if self._policy_mode == "ppo":
            return self._select_ppo_action(obs)

        else:
            return self._select_q_action(obs)

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
        current_inputs = dict(self._inputs)

        brain_input_fields = getattr(self._brain, "_input_fields", None)
        if (
            isinstance(brain_input_fields, dict)
            and "reward" in brain_input_fields
            and "reward" not in current_inputs
        ):
            # Some Brain configurations include an explicit reward input field.
            # Before the environment step, use a neutral default reward.
            current_inputs["reward"] = 0.0

        if current_obs is None:
            raise RuntimeError("Agent observation state was not initialized.")

        self._logger.info("[Layer:Agent] step start policy=%s learn=%s", self._policy_mode, learn)
        brain_outputs = self._brain.step(current_inputs, learn=learn)
        action = self.select_action(current_obs, brain_outputs=brain_outputs)
        result = self._adapter.step_bridge(action)
        action_inputs = self._adapter.action_to_inputs(action)

        next_obs = result["obs"]
        reward = result["reward"]

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

        self._logger.info(
            "[Layer:Agent] step end action=%s action_inputs=%s reward=%.6f terminated=%s truncated=%s",
            action,
            action_inputs,
            float(reward),
            bool(result["terminated"]),
            bool(result["truncated"]),
        )

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

        This method applies a one-step tabular Q-learning update when the
        active policy is backed by a discrete action index. If an action cannot
        be resolved to a Q-row index, the update is skipped. In ``brain``
        policy mode this method is currently a no-op.

        Args:
            obs: Observation before the action was taken.
            action: Environment action selected for the timestep.
            reward: Reward returned by the environment.
            next_obs: Observation returned after the transition.
            done: Whether the transition ended the episode.
        """

        scaled_reward = float(reward) * self._reward_scale

        if self._policy_mode == "brain":
            self._brain.rl_policy_update(reward=scaled_reward)
            return

        if self._policy_mode == "ppo":
            # PPO model training lifecycle is managed externally.
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

        td_target = scaled_reward + (self._discount_factor * next_best_q)
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
