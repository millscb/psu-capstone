"""Unit tests for Agent orchestration and policy behavior."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

from htmrl.agent_layer.agent import Agent
from htmrl.agent_layer.brain import Brain
from htmrl.agent_layer.HTM import ColumnField, InputField, OutputField
from htmrl.encoder_layer.rdse import RDSEParameters
from htmrl.environment.env_adapter import EnvAdapter


class _AdapterStub(EnvAdapter):
    """Environment adapter stub that mimics a one-step episodic transition.

    Args:
        n: Number of discrete actions exposed in the action spec.
        space_type: Space type label used to test q_table validation behavior.
    """

    def __init__(self, n: int = 2, space_type: str = "Discrete") -> None:
        self._spec = {
            "space_type": space_type,
            "n": n,
            "values": list(range(n)) if space_type == "Discrete" else [],
        }
        self.action_space = gym.spaces.Discrete(n)
        self.observation_space = None  # type: ignore[assignment]
        self._action_space = gym.spaces.Discrete(n)
        self.last_step_action: Any | None = None

    def get_action_spec(self) -> dict[str, Any]:
        """Return the configured action-space description."""
        return self._spec

    def observation_to_inputs(self, observation: Any) -> dict[str, Any]:
        """Convert observations to the flat input format expected by Agent."""
        if isinstance(observation, dict):
            return observation
        return {"observation": observation}

    def action_to_inputs(self, action: Any) -> dict[str, Any]:
        """Convert an action value into a minimal keyed payload."""
        return {"action": action}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a deterministic initial state for episode reset tests."""

        return {"state": 0}, {}

    def reset_bridge(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return deterministic adapter bridge for reset tests."""

        return {
            "obs": {"state": 0},
            "inputs": {"state": 0},
            "info": {},
        }

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Record the chosen action and return a deterministic transition."""

        self.last_step_action = action
        return {"state": 1}, 1.0, True, False, {"source": "stub"}

    def step_bridge(self, action: Any) -> dict[str, Any]:
        """Return deterministic adapter bridge for step tests."""

        self.last_step_action = action
        return {
            "obs": {"state": 1},
            "inputs": {"state": 1},
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
            "info": {"source": "stub"},
        }


class _PPOMock:
    """Minimal PPO-like model stub for policy dispatch tests."""

    def __init__(self, action: int = 0) -> None:
        self.action = action
        self.calls: list[tuple[Any, bool]] = []

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
        self.calls.append((observation, deterministic))
        return self.action, None


def test_q_table_policy_requires_discrete_action_space() -> None:
    """q_table mode should reject non-discrete action spaces at construction."""

    adapter = EnvAdapter("MountainCarContinuous-v0")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        with pytest.raises(TypeError, match="q_table policy requires a Discrete action space"):
            Agent(brain=brain, adapter=adapter, policy_mode="q_table")
    finally:
        adapter.env.close()


def test_q_values_row_initialized_by_action_count() -> None:
    """Newly seen states should get zero Q rows sized to action-space cardinality."""

    adapter = _AdapterStub(n=4)
    brain = _build_real_brain_for_adapter_inputs(adapter)
    agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")

    state_key = agent._initialize_q_values({"state": 5})
    q_row = agent._q_values[state_key]

    assert isinstance(q_row, np.ndarray)
    assert q_row.shape == (4,)
    assert np.allclose(q_row, np.zeros(4))


def test_select_q_action_uses_argmax_when_not_exploring() -> None:
    """With epsilon disabled, q_table policy should pick the maximum-valued action."""

    adapter = _AdapterStub(n=3)
    brain = _build_real_brain_for_adapter_inputs(adapter)
    agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")

    state_key = agent._state_key({"state": 2})
    agent._q_values[state_key] = np.array([0.1, 0.8, 0.3])
    agent._epsilon = 0.0

    action = agent.select_action({"state": 2})

    assert action == 1


def test_update_applies_q_learning_bootstrap_target() -> None:
    """update should apply one-step Q-learning with next-state bootstrap value."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")

        state_key = agent._state_key({"state": 0})
        next_state_key = agent._state_key({"state": 1})
        agent._q_values[state_key] = np.array([0.0, 0.0])
        agent._q_values[next_state_key] = np.array([0.5, 0.2])

        agent.update({"state": 0}, action=0, reward=1.0, next_obs={"state": 1})

        # lr=0.1, gamma=0.99 -> target=1.0 + 0.99*0.5 = 1.495, new_q=0.1495
        assert np.isclose(agent._q_values[state_key][0], 0.1495, rtol=1e-09, atol=1e-09)
    finally:
        adapter.env.close()


def test_update_ignores_bootstrap_on_terminal_transition() -> None:
    """update should not bootstrap next-state value when transition is terminal."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")

        state_key = agent._state_key({"state": 0})
        next_state_key = agent._state_key({"state": 1})
        agent._q_values[state_key] = np.array([0.0, 0.0])
        agent._q_values[next_state_key] = np.array([10.0, 10.0])

        agent.update({"state": 0}, action=0, reward=1.0, next_obs={"state": 1}, done=True)

        # Terminal target is reward only: target=1.0, new_q=0.1
        assert np.isclose(agent._q_values[state_key][0], 0.1, rtol=1e-09, atol=1e-09)
    finally:
        adapter.env.close()


def test_update_is_noop_for_brain_policy_mode() -> None:
    """update should not mutate q-table values when running in brain policy mode."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="brain")

        state_key = agent._state_key({"state": 0})
        next_state_key = agent._state_key({"state": 1})
        agent._q_values[state_key] = np.array([0.3, 0.7])
        agent._q_values[next_state_key] = np.array([0.9, 0.1])

        before = agent._q_values[state_key].copy()
        agent.update({"state": 0}, action=0, reward=1.0, next_obs={"state": 1}, done=False)

        assert np.allclose(agent._q_values[state_key], before)
        assert agent._training_error == []
    finally:
        adapter.env.close()


def test_step_runs_brain_then_env_and_returns_transition() -> None:
    """Agent.step should process Brain input, act, and return the transition payload."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")
        agent._epsilon = 0.0

        with patch.object(brain, "step", wraps=brain.step) as brain_step_spy:
            transition = agent.step(learn=False)

        assert brain_step_spy.call_count == 1
        call_inputs = brain_step_spy.call_args[0][0]
        assert isinstance(call_inputs, dict)
        assert isinstance(transition["obs"], np.ndarray)
        assert isinstance(transition["next_obs"], np.ndarray)
        assert transition["action"] in (0, 1)
        assert isinstance(transition["reward"], float)
        assert isinstance(transition["terminated"], bool)
        assert isinstance(transition["truncated"], bool)
    finally:
        adapter.env.close()


def test_brain_policy_uses_action_from_brain_step_output() -> None:
    """brain mode should prefer direct action hints from Brain.step outputs."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_with_output_field(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="brain")
        agent._epsilon = 0.0

        # Activate all output cells so decode selects the last motor action (1)
        output_field = brain.fields["action_output"]
        assert isinstance(output_field, OutputField)
        for cell in output_field.cells:
            cell.set_active()  # type: ignore

        transition = agent.step(learn=False)

        assert transition["action"] == 1
    finally:
        adapter.env.close()


def test_ppo_policy_selects_action_from_injected_model() -> None:
    """ppo mode should delegate action selection to internally built PPO model."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        ppo: Any = _PPOMock(action=1)

        with patch.object(Agent, "_build_ppo_model", return_value=ppo):
            agent = Agent(brain=brain, adapter=adapter, policy_mode="ppo")
            transition = agent.step(learn=False)

        assert transition["action"] == 1
        assert len(ppo.calls) == 1
        _, deterministic = ppo.calls[0]
        assert deterministic is True
    finally:
        adapter.env.close()


def test_ppo_policy_without_model_raises_value_error() -> None:
    """ppo mode should fail fast when adapter does not expose a Gym env."""

    # _AdapterStub intentionally skips EnvAdapter.__init__ so has no _env attribute,
    # which triggers the ValueError in _build_ppo_model.
    adapter = _AdapterStub(n=2)
    brain = _build_real_brain_for_adapter_inputs(adapter)

    with pytest.raises(ValueError, match="requires an adapter with a Gym environment"):
        Agent(brain=brain, adapter=adapter, policy_mode="ppo")


def _build_real_brain_for_adapter_inputs(adapter: EnvAdapter) -> Brain:
    """Create a real Brain whose input fields match adapter input keys."""

    reset_bridge = adapter.reset_bridge()
    input_names = list(reset_bridge["inputs"].keys())
    input_fields: dict[str, InputField] = {}

    for index, name in enumerate(input_names):
        rdse_params = RDSEParameters(
            size=64,
            active_bits=0,
            sparsity=0.02,
            resolution=0.001,
            category=False,
            seed=100 + index,
        )
        input_fields[name] = InputField(size=64, encoder_params=rdse_params)

    # Add reward input field to match agent step inputs
    reward_params = RDSEParameters(
        size=64,
        active_bits=0,
        sparsity=0.02,
        resolution=0.01,
        category=False,
        seed=999,
    )
    input_fields["reward"] = InputField(size=64, encoder_params=reward_params)

    column_field = ColumnField(
        input_fields=list(input_fields.values()),
        non_spatial=True,
        num_columns=64,
        cells_per_column=8,
    )
    fields = {**input_fields, "column": column_field}
    return Brain(fields)


def _build_real_brain_with_output_field(adapter: EnvAdapter) -> Brain:
    """Create a real Brain with matching inputs plus a real OutputField."""

    brain = _build_real_brain_for_adapter_inputs(adapter)
    output_field = OutputField(size=4, motor_action=(0, 1))
    brain.fields["action_output"] = output_field
    return Brain(brain.fields)


def test_real_brain_agent_adapter_gym_single_step_q_table() -> None:
    """Real Brain + Adapter should complete one CartPole step through Agent in q_table mode."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")
        agent._epsilon = 0.0

        transition = agent.step(learn=False)

        assert isinstance(transition["obs"], np.ndarray)
        assert isinstance(transition["next_obs"], np.ndarray)
        assert transition["action"] in (0, 1)
        assert isinstance(transition["inputs"], dict)
        assert isinstance(transition["next_inputs"], dict)
        assert "reward" in transition
    finally:
        adapter.env.close()


def test_real_brain_policy_mode_fallback_to_q_table() -> None:
    """brain policy mode should still step CartPole by falling back when no action predictions exist."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="brain")
        agent._epsilon = 0.0

        transition = agent.step(learn=False)

        assert transition["action"] in (0, 1)
        assert isinstance(transition["terminated"], bool)
        assert isinstance(transition["truncated"], bool)
    finally:
        adapter.env.close()


def test_real_brain_reads_and_encodes_adapter_inputs() -> None:
    """Agent.step should pass adapter inputs into real InputField encoders."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")
        agent._epsilon = 0.0

        encode_spies: dict[str, Any] = {}
        with ExitStack() as stack:
            for name, input_field in brain._input_fields.items():
                encode_spies[name] = stack.enter_context(
                    patch.object(input_field, "encode", wraps=input_field.encode)
                )

            transition = agent.step(learn=False)

        for name, encode_spy in encode_spies.items():
            encode_spy.assert_called_once_with(transition["inputs"][name])
    finally:
        adapter.env.close()


def test_real_input_fields_encode_values_into_sdr_vectors() -> None:
    """Real InputField instances should encode adapter values into binary SDR vectors."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        reset_bridge = adapter.reset_bridge()

        for name, value in reset_bridge["inputs"].items():
            input_field = brain._input_fields[name]
            encoded = input_field.encode(value)

            assert isinstance(encoded, list)
            assert len(encoded) == len(input_field.cells)
            assert set(encoded).issubset({0, 1})
            assert sum(encoded) > 0
    finally:
        adapter.env.close()


def test_real_output_field_decode_drives_brain_policy_action() -> None:
    """Real OutputField decode payload should directly determine brain-policy action."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_with_output_field(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="brain")

        # Activate all output cells so decode selects the last index deterministically.
        output_field = brain.fields["action_output"]
        assert isinstance(output_field, OutputField)
        for cell in output_field.cells:
            cell.set_active()  # type: ignore

        transition = agent.step(learn=False)

        assert transition["action"] == 1
        assert transition["action"] in (0, 1)
    finally:
        adapter.env.close()
