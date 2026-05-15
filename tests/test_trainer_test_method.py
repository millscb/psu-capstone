"""Focused regression tests for Trainer.test edge cases."""

from types import SimpleNamespace

import pytest

from htmrl.agent_layer.pullin.pullin_brain import Brain
from htmrl.agent_layer.train import Trainer


class _FakeBrain:
    def __init__(self) -> None:
        self._column_fields = {"c": SimpleNamespace(bursting_columns=[])}
        self._prediction_calls = 0
        self.step_calls = 0

    def prediction(self) -> dict[str, float]:
        self._prediction_calls += 1
        return {"value_input": 1.0}

    def step(self, inputs, learn=False):  # noqa: ANN001
        self.step_calls += 1
        return {"inputs": inputs, "learn": learn}


@pytest.fixture
def trainer() -> Trainer:
    return Trainer(Brain())


# Test Type: unit test
def test_test_method_rejects_empty_series(trainer: Trainer) -> None:
    fake_brain = _FakeBrain()
    with pytest.raises(ValueError, match="empty series"):
        trainer.test(fake_brain, {"value_input": []}, steps=None)


# Test Type: unit test
def test_test_method_honors_zero_steps(trainer: Trainer) -> None:
    fake_brain = _FakeBrain()
    result = trainer.test(fake_brain, {"value_input": [1.0, 2.0]}, steps=0)
    assert fake_brain.step_calls == 0
    assert result["errors"]["value_input"] == []
    assert result["evaluation_bursts"] == []
