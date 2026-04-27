"""HTM Brain consumer for Gymnasium FrozenLake-v1 environment.

Wires a Brain with:
  - CategoryEncoder InputField for discrete state observations (16 tiles)
  - ColumnField for temporal memory / sequence learning
  - ValueFieldMixin go/nogo fields for TD-based reward signaling
  - OutputField for stochastic action selection (4 actions)

Usage:
    python src/frozen_lake.py
"""

import sys
from pathlib import Path
from typing import Any

# Allow running this file directly via an absolute path without installing the package.
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import gymnasium as gym  # noqa: E402

from psu_capstone.agent_layer.pullin.pullin_brain import Brain  # noqa: E402
from psu_capstone.agent_layer.pullin.pullin_htm import (  # noqa: E402
    ColumnField,
    InputField,
    OutputField,
)
from psu_capstone.agent_layer.pullin.sungur import ValueField  # noqa: E402
from psu_capstone.encoder_layer.category_encoder import CategoryParameters  # noqa: E402
from psu_capstone.encoder_layer.rdse import RDSEParameters  # noqa: E402
from psu_capstone.environment.pullin.pullin_gym_adapter import GymBrain  # noqa: E402

# -- Environment ----------------------------------------------------------------

ENV_NAME = "FrozenLake-v1"
NUM_STATES = 16
NUM_ACTIONS = 4
ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

# -- Encoder parameters ---------------------------------------------------------

STATE_ENCODER_SIZE = 256
STATE_ACTIVE_BITS = 10

ACTION_ENCODER_SIZE = 128
ACTION_ACTIVE_BITS = 10

# -- HTM parameters -------------------------------------------------------------

CELLS_PER_COLUMN = 8
NUM_EPISODES = 5_000
MAX_STEPS_PER_EPISODE = 100
PRINT_EVERY = 5


def build_brain() -> Brain:
    """Construct the HTM brain for FrozenLake."""

    # State encoder: one unique SDR per tile (0-15)
    state_encoder_params = CategoryParameters(
        w=STATE_ACTIVE_BITS,
        category_list=[str(state) for state in range(NUM_STATES)],
    )
    state_field = InputField(encoder_params=state_encoder_params)

    # Pre-register all 16 states so the encoder cache is warm
    for s in range(NUM_STATES):
        state_field.encoder.encode(str(s))  # type: ignore[arg-type]

    # Temporal memory layer
    column_field = ColumnField(
        input_fields=[state_field],
        non_spatial=True,
        cells_per_column=CELLS_PER_COLUMN,
    )

    # Go / NoGo value fields for TD reward signaling
    go_field = ValueField(
        input_fields=[column_field], non_spatial=True, cells_per_column=CELLS_PER_COLUMN
    )
    nogo_field = ValueField(
        input_fields=[column_field], non_spatial=True, cells_per_column=CELLS_PER_COLUMN
    )

    column_field.go_field = go_field
    column_field.nogo_field = nogo_field

    # Action output field driven by column layer activity
    action_encoder_params = RDSEParameters(
        size=ACTION_ENCODER_SIZE,
        active_bits=ACTION_ACTIVE_BITS,
        sparsity=0.0,
        radius=0.0,
        resolution=1.0,
        category=False,
        seed=1,
    )
    action_field = OutputField(
        input_field=column_field,
        encoder_params=action_encoder_params,
        size=ACTION_ENCODER_SIZE,
    )

    # Pre-register action encodings (0-3) so decode can map back
    for a in range(NUM_ACTIONS):
        action_field.encoder.encode(a)

    return Brain(
        {  # type: ignore[arg-type]  # ValueFieldMixin isn't a Field subclass
            "state": state_field,
            "columns": column_field,
            "go": go_field,
            "nogo": nogo_field,
            "action": action_field,
        }
    )


def pick_action(brain: Brain) -> int:
    """Decode the OutputField activation into a discrete action (legacy helper)."""
    action_field: OutputField = brain.fields["action"]  # type: ignore[assignment]
    result = action_field.decode()
    value = result["value"]
    if value is not None and 0 <= int(value) < NUM_ACTIONS:
        return int(value)
    import random

    return random.randint(0, NUM_ACTIONS - 1)


# -- FrozenLake adapters -------------------------------------------------------


def obs_to_inputs(obs: Any) -> dict[str, Any]:
    """Convert FrozenLake observation (int 0-15) to Brain input dict."""
    return {"state": str(obs)}


def behavior_to_action(behavior: dict[str, Any]) -> int:
    """Convert Brain behavior dict to a discrete FrozenLake action."""
    value = behavior.get("action")
    if isinstance(value, dict):
        value = value.get("action")
    if value is not None and 0 <= int(value) < NUM_ACTIONS:
        return int(value)
    import random

    return random.randint(0, NUM_ACTIONS - 1)


def build_agent() -> GymBrain:
    """Construct the HTM brain wrapped for Gymnasium use."""
    brain = build_brain()
    return GymBrain(brain, obs_to_inputs, behavior_to_action)


def run_episode(agent: GymBrain, env: gym.Env, learn: bool = True) -> tuple[float, int]:
    """Run a single FrozenLake episode."""
    obs, _ = env.reset()
    agent.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.step(obs, reward=total_reward, learn=learn)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            agent.step(obs, reward=total_reward, learn=learn)
            return total_reward, step + 1

    return total_reward, MAX_STEPS_PER_EPISODE


def step_fn(env: gym.Env, action: int, _timestep: int) -> dict[str, Any]:
    """Step function for ClusterTracker input snapshots (legacy helper)."""
    obs, reward, _, _, _ = env.step(action)
    stimulus = obs_to_inputs(obs)
    stimulus["reward"] = reward
    return stimulus


def main() -> None:
    env = gym.make(ENV_NAME, is_slippery=False, render_mode="human", reward_schedule=(20, -2, -1))
    agent = build_agent()

    wins = 0
    recent_wins = 0
    recent_window = PRINT_EVERY

    for episode in range(1, NUM_EPISODES + 1):
        reward, _ = run_episode(agent, env, learn=True)
        if reward > 0:
            wins += 1
            recent_wins += 1

        if episode % recent_window == 0:
            win_rate = recent_wins / recent_window
            print(
                f"Episode {episode:>5d} | "
                f"Recent win rate: {win_rate:.2%} | "
                f"Total wins: {wins}/{episode}"
            )
            recent_wins = 0

    # Evaluation (no learning)
    eval_episodes = 100
    eval_wins = 0
    for _ in range(eval_episodes):
        reward, _ = run_episode(agent, env, learn=False)
        if reward > 0:
            eval_wins += 1

    print(
        f"\nEvaluation ({eval_episodes} episodes, no learning): "
        f"Win rate = {eval_wins / eval_episodes:.2%}"
    )

    env.close()


if __name__ == "__main__":
    main()
