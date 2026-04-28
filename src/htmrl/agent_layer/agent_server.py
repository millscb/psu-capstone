"""WebSocket server for real-time agent visualization and control.

This module provides a WebSocket server that the web visualization client
connects to in order to run episodes, send observations, and receive actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.server import ServerConnection

from htmrl.agent_layer.agent import Agent
from htmrl.log import get_logger


class AgentWebSocketServer:
    """WebSocket server for real-time agent control and visualization.

    Bridges a web visualization client to a running Python agent. The server
    manages the agent's episode loop and translates between JSON messages
    and agent API calls.

    Args:
        agent: The Agent instance to run episodes.
        host: WebSocket server bind address (default: localhost).
        port: WebSocket server bind port (default: 8765).
        log_level: Logging verbosity (default: logging.INFO).
    """

    def __init__(
        self,
        agent: Agent,
        host: str = "localhost",
        port: int = 8765,
        log_level: int = logging.INFO,
    ) -> None:
        self._agent = agent
        self._host = host
        self._port = port
        self._logger = get_logger("AgentWebSocketServer")
        raw_max_episodes = getattr(agent, "_training_episodes", None)
        self._max_episodes: int | None = (
            int(raw_max_episodes) if isinstance(raw_max_episodes, (int, float)) else None
        )
        self._max_steps_per_episode: int | None = getattr(
            agent._adapter, "max_steps_per_episode", None
        )
        self._connections: set[ServerConnection] = set()
        self._episode_state: dict[ServerConnection, dict[str, Any]] = {}

    async def handle_client(self, websocket: ServerConnection) -> None:
        """Handle a single client connection lifecycle.

        Args:
            websocket: The ServerConnection from websockets library.
        """
        self._connections.add(websocket)
        client_id = id(websocket)
        self._logger.info(f"Client {client_id} connected from {websocket.remote_address}")

        self._episode_state[client_id] = {
            "episode_active": False,
            "episode_num": 0,
            "step_num": 0,
            "total_reward": 0.0,
            "prev_obs": None,
            "prev_action": None,
        }

        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "config",
                        "max_episodes": self._max_episodes,
                        "max_steps_per_episode": self._max_steps_per_episode,
                    }
                )
            )

            await websocket.send(
                json.dumps(
                    {
                        "type": "ready",
                        "message": "Server ready",
                        "max_episodes": self._max_episodes,
                        "max_steps_per_episode": self._max_steps_per_episode,
                    }
                )
            )

            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self._process_message(websocket, client_id, data)
                    if response:
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError as e:
                    self._logger.error(f"Invalid JSON from {client_id}: {e}")
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"Invalid JSON: {str(e)}"})
                    )
                except Exception as e:
                    self._logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"Server error: {str(e)}"})
                    )

        except websockets.exceptions.ConnectionClosed:
            self._logger.info(f"Client {client_id} disconnected normally")
        except Exception as e:
            self._logger.error(f"Unexpected error with client {client_id}: {e}")
        finally:
            self._connections.discard(websocket)
            self._episode_state.pop(client_id, None)
            self._logger.info(f"Client {client_id} cleanup complete")

    async def _process_message(
        self, websocket: ServerConnection, client_id: int, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process an incoming message and return a response.

        Supported message types:
        - "start_episode": Initialize a new episode
        - "reset": Reset and start a fresh episode
        - "step": Run one agent timestep with the provided observation
        - "stop": End the current episode
        - "status": Return current episode state

        Args:
            websocket: The client connection.
            client_id: Unique identifier for this client.
            data: Parsed JSON message dict.

        Returns:
            Response dict to send back to client, or None if no response needed.
        """
        msg_type: str = data.get("type", "unknown")
        self._logger.debug(f"Client {client_id} message: {msg_type}")

        if msg_type == "start_episode":
            return await self._handle_start_episode(client_id)
        elif msg_type == "reset":
            return await self._handle_reset(client_id)
        elif msg_type == "observation":
            # Accept both "observation" (web app) and "obs" (legacy)
            obs = data.get("observation") or data.get("obs")
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            return await self._handle_observation(client_id, obs, reward, done)
        elif msg_type == "step":
            inputs = data.get("inputs", {})
            return await self._handle_step(client_id, inputs)
        elif msg_type == "stop":
            return await self._handle_stop(client_id)
        elif msg_type == "status":
            return await self._handle_status(client_id)
        else:
            self._logger.warning(f"Unknown message type: {msg_type}")
            return {"type": "error", "message": f"Unknown message type: {msg_type}"}

    async def _handle_start_episode(self, client_id: int) -> dict[str, Any]:
        """Start a new episode and return initial observation."""
        try:
            self._agent.reset_episode()

            state = self._episode_state[client_id]
            state["episode_active"] = True
            state["episode_num"] += 1
            state["step_num"] = 0
            state["total_reward"] = 0.0
            state["prev_obs"] = None
            state["prev_action"] = None

            obs = self._agent._obs
            inputs = self._agent._inputs

            self._logger.info(f"Episode {state['episode_num']} started for client {client_id}")

            return {
                "type": "episode_start",
                "episode": state["episode_num"],
                "max_episodes": self._max_episodes,
                "max_steps_per_episode": self._max_steps_per_episode,
                "obs": self._serialize_value(obs),
                "inputs": inputs,
            }
        except Exception as e:
            self._logger.error(f"Failed to start episode for {client_id}: {e}")
            return {"type": "error", "message": f"Failed to start episode: {str(e)}"}

    async def _handle_reset(self, client_id: int) -> dict[str, Any]:
        """Reset the agent and start a fresh episode."""
        return await self._handle_start_episode(client_id)

    async def _handle_step(self, client_id: int, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run one agent step and return the action."""
        state = self._episode_state.get(client_id)
        if not state or not state["episode_active"]:
            return {
                "type": "error",
                "message": "No active episode. Call 'start_episode' first.",
            }

        try:
            self._agent._inputs = inputs
            transition = self._agent.step(learn=True)

            action = transition["action"]
            reward = transition["reward"]
            done = transition["terminated"] or transition["truncated"]

            state["step_num"] += 1
            state["total_reward"] += float(reward)

            if done:
                state["episode_active"] = False

            self._logger.debug(
                f"Step {state['step_num']}: action={action}, reward={reward}, done={done}"
            )

            return {
                "type": "step_result",
                "step": state["step_num"],
                "max_episodes": self._max_episodes,
                "max_steps_per_episode": self._max_steps_per_episode,
                "action": self._serialize_value(action),
                "reward": float(reward),
                "terminated": bool(transition["terminated"]),
                "truncated": bool(transition["truncated"]),
                "total_reward": float(state["total_reward"]),
                "episode_done": done,
            }

        except Exception as e:
            self._logger.error(f"Step failed for client {client_id}: {e}")
            state["episode_active"] = False
            return {"type": "error", "message": f"Step failed: {str(e)}"}

    async def _handle_observation(
        self,
        client_id: int,
        obs: Any,
        reward: float,
        done: bool,
    ) -> dict[str, Any]:
        """Accept an observation from the client and return the agent's action.

        This is the primary message type for client-driven loops where the
        environment runs on the client side and the server hosts only the
        agent brain.

        Args:
            client_id: Unique identifier for this client.
            obs: Current environment observation sent by the client.
            reward: Reward received for the previous action (0.0 on first step).
            done: Whether the episode ended after the previous action.

        Returns:
            Response payload containing the chosen action or an error message.
        """
        state = self._episode_state.get(client_id)
        if not state or not state["episode_active"]:
            # Auto-start an episode so the web app doesn't need a separate start_episode call
            if state is None:
                self._episode_state[client_id] = {
                    "episode_active": False,
                    "episode_num": 0,
                    "step_num": 0,
                    "total_reward": 0.0,
                    "prev_obs": None,
                    "prev_action": None,
                }
                state = self._episode_state[client_id]

            state["episode_active"] = True
            state["episode_num"] += 1
            state["step_num"] = 0
            state["total_reward"] = 0.0
            state["prev_obs"] = None
            state["prev_action"] = None
            self._logger.info(f"Auto-started episode {state['episode_num']} for client {client_id}")

        try:
            import numpy as np

            obs_value = obs
            # If the adapter's observation_space is Box, convert dict/list to np.array
            adapter = getattr(self._agent, "_adapter", None)
            obs_space = getattr(adapter, "observation_space", None)

            if obs_space is not None:
                from gymnasium import spaces as gym_spaces

                if isinstance(obs_space, gym_spaces.Box):
                    # If obs is a dict, try to convert to array by value order
                    if isinstance(obs, dict):
                        obs_value = np.array(list(obs.values()), dtype=np.float32)
                    elif isinstance(obs, list):
                        obs_value = np.array(obs, dtype=np.float32)
                elif isinstance(obs_space, gym_spaces.Discrete):
                    # If obs is a dict, extract the first value
                    if isinstance(obs, dict):
                        # Accept dicts with a single key or take the first value
                        if len(obs) == 1:
                            obs_value = next(iter(obs.values()))
                        else:
                            obs_value = list(obs.values())[0]
                    else:
                        obs_value = obs

            state["step_num"] += 1
            state["total_reward"] += float(reward)

            # Q-update for the transition that just completed (prev_action → reward → obs)
            if state["prev_obs"] is not None and state["prev_action"] is not None:
                self._agent.update(
                    state["prev_obs"],
                    state["prev_action"],
                    float(reward),
                    obs_value,
                    done,
                )

            # Feed the observation through the brain so predictions are current
            inputs = self._agent._adapter.observation_to_inputs(obs_value)
            inputs["reward"] = float(reward)
            brain_outputs = self._agent._brain.step(inputs, learn=True)

            action = self._agent.select_action(obs_value, brain_outputs=brain_outputs)

            # Cache for the next update call
            state["prev_obs"] = obs_value
            state["prev_action"] = action

            if done:
                state["episode_active"] = False
                state["prev_obs"] = None
                state["prev_action"] = None

            self._logger.debug(
                f"Observation step {state['step_num']}: action={action}, "
                f"reward={reward}, done={done}"
            )

            return {
                "type": "action",
                "step": state["step_num"],
                "max_episodes": self._max_episodes,
                "max_steps_per_episode": self._max_steps_per_episode,
                "action": self._serialize_value(action),
                "episode_done": done,
                "total_reward": float(state["total_reward"]),
            }
        except Exception as e:
            self._logger.error(f"Observation handling failed for client {client_id}: {e}")
            state["episode_active"] = False
            return {"type": "error", "message": f"Observation handling failed: {str(e)}"}

    async def _handle_stop(self, client_id: int) -> dict[str, Any]:
        """Stop the current episode."""
        state = self._episode_state.get(client_id)
        if state:
            was_active = state["episode_active"]
            state["episode_active"] = False

            summary = {
                "type": "episode_stop",
                "episode": state["episode_num"],
                "max_episodes": self._max_episodes,
                "max_steps_per_episode": self._max_steps_per_episode,
                "steps": state["step_num"],
                "total_reward": float(state["total_reward"]),
                "was_active": was_active,
            }

            self._logger.info(
                f"Episode {state['episode_num']} stopped: "
                f"{state['step_num']} steps, "
                f"reward={state['total_reward']:.2f}"
            )
            return summary

        return {"type": "error", "message": "No active episode to stop"}

    async def _handle_status(self, client_id: int) -> dict[str, Any]:
        """Return the current episode state."""
        state = self._episode_state.get(client_id)
        if state:
            return {
                "type": "status",
                "episode": state["episode_num"],
                "max_episodes": self._max_episodes,
                "max_steps_per_episode": self._max_steps_per_episode,
                "step": state["step_num"],
                "total_reward": float(state["total_reward"]),
                "episode_active": state["episode_active"],
            }
        return {"type": "error", "message": "No episode state available"}

    def _serialize_value(self, value: Any) -> Any:
        """Convert observation/action values to JSON-serializable form."""
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return str(value)

    async def start(self) -> None:
        """Start the WebSocket server and run forever."""
        self._logger.info(f"Starting WebSocket server on ws://{self._host}:{self._port}")

        async with websockets.serve(
            self.handle_client, self._host, self._port, ping_interval=20, ping_timeout=20
        ):
            self._logger.info("WebSocket server is running. Waiting for connections...")
            await asyncio.Future()

    def run(self) -> None:
        """Start the server using asyncio.run() (blocking call)."""
        asyncio.run(self.start())
