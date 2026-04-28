"""Protocol definitions for agent layer components.

This module defines the interface contract that all agents must satisfy
for decision-making and learning in the HTM reinforcement learning system.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AgentInterface(Protocol):
    """Protocol defining the interface requirements for HTM-RL agents.

    Agents implementing this protocol can select actions based on state
    observations and update their internal policy based on experience
    in reinforcement learning environments.
    """

    def select_action(self, state: tuple) -> Any:
        """Select an action based on the current state observation.

        Processes the current state through the agent's decision-making
        mechanism (typically HTM + policy network) to determine the best action.

        Args:
            state: Current environment state as a tuple of observations.

        Returns:
            The action to take in the environment.
        """
        ...

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute the given action and return the resulting experience tuple.

        This method should interact with the environment to perform the action,
        observe the next state, receive the reward, and determine if the episode
        has terminated.

        Args:
            action: The action to execute in the environment.

        Returns:
            A tuple of (next_state, reward, done, info) where:
                - next_state: The new state observation after taking the action.
                - reward: The reward received from the environment.
                - done: A boolean indicating if the episode has ended.
                - info: A dictionary with any additional information from the environment.
        """

        ...

    def update(self, state: tuple, action: Any, reward: float, next_state: tuple) -> None:
        """Update the agent's internal model and policy based on experience.

        This method should use the experience tuple (state, action, reward, next_state)
        to update the agent's learning mechanism, such as updating the HTM model
        or training a policy network.

        Args:
            state: The state observed before taking the action.
            action: The action taken by the agent.
            reward: The reward received from the environment.
            next_state: The state observed after taking the action.
        """

        ...
