"""Tests for AgentWebSocketServer visualization payload helpers."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from htmrl.agent_layer.agent_server import AgentWebSocketServer
from htmrl.agent_layer.pullin.pullin_brain import Brain
from htmrl.agent_layer.train import Trainer
from htmrl.environment.frontend_env_adapter import FrontendEnvAdapter

pytest.importorskip("websockets")


@pytest.fixture()
def real_server() -> AgentWebSocketServer:
    """Build a server with the real pullin Brain/HTM stack for TradingEnv schema."""
    adapter = FrontendEnvAdapter(
        env_name="TradingEnv",
        observation_labels=["open", "high", "low", "close", "volume"],
        action_count=3,
        initial_observation={
            "open": 1769.5,
            "high": 1773.5,
            "low": 1739.75,
            "close": 1749.25,
            "volume": 1623508.0,
        },
    )
    config = SimpleNamespace(
        input_size=64,
        resolution=0.01,
        seed=5,
        cells_per_column=4,
    )
    trainer = Trainer(Brain({}))
    brain = trainer.build_brain_for_env(adapter, config)
    agent = SimpleNamespace(
        _training_episodes=10,
        _adapter=adapter,
        _brain=brain,
    )
    return AgentWebSocketServer(agent=cast(Any, agent))


pytest.importorskip("websockets")


# Test Type: system test
def test_build_trading_visualization_returns_none_for_non_trading_obs(
    # TS-29 TC-261
    real_server: AgentWebSocketServer,
) -> None:
    server = real_server

    payload = server._build_trading_visualization(
        obs={"x": 1.0},
        action=1,
        reward=0.1,
        total_reward=0.3,
        brain_outputs={},
        step=2,
    )

    assert payload is None


# Test Type: system test
def test_build_trading_visualization_computes_good_buy_alignment(
    # TS-29 TC-261
    real_server: AgentWebSocketServer,
) -> None:
    server = real_server
    obs = {
        "open": 100.0,
        "high": 104.0,
        "low": 99.5,
        "close": 103.0,
        "volume": 250000.0,
    }
    brain_outputs = {"action_output": {"action": 1, "confidence": 0.8}}

    payload = server._build_trading_visualization(
        obs=obs,
        action=1,
        reward=1.5,
        total_reward=3.5,
        brain_outputs=brain_outputs,
        step=5,
    )

    assert payload is not None
    assert payload["action"]["label"] == "buy"
    assert payload["alignment_score"] == pytest.approx(1.0)
    assert payload["quality"] == "good"
    assert payload["brain_confidence"] == pytest.approx(0.8)
    assert payload["ohlcv"]["close"] == pytest.approx(103.0)
