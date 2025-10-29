import sys

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any
from numenta.utils import get_logger

sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
from gymnasium import spaces
from numenta.frozen_lake import FrozenLakeEnvironment, GymAdapter
from stable_baselines3 import PPO 
from stable_baselines3.common.evaluation import evaluate_policy

from htm import TemporalPooler

class Agent:
    def __init__(self, L2: TemporalPooler|None=None, L5: TemporalPooler|None=None):
        self.sdr_size = 64
        self.cells_per_column = 4
        self.sparsity = 0.1

        self.sdr = [1] * int(self.sdr_size * self.sparsity) + [0] * (
            self.sdr_size - int(self.sdr_size * self.sparsity)
        )

        # Allow injection of shared temporal poolers so different env instances use the same learner/state
        self.L2 = (
            L2
            if L2 is not None
            else TemporalPooler(
                input_space_size=self.sdr_size,
                column_count=self.sdr_size,
                cells_per_column=self.cells_per_column,
                initial_synapses_per_column=12,
            )
        )

        self.L5 = (
            L5
            if L5 is not None
            else TemporalPooler(
                input_space_size=self.sdr_size + self.L2.column_count*self.L2.cells_per_column,
                column_count=self.sdr_size * 2,
                cells_per_column=self.cells_per_column,
                initial_synapses_per_column=12,
            )
        )

    @property
    def observation_size(self):
        return self.sdr_size * 2 * self.cells_per_column

    def encode_feature(self, feature_vector):
        state = self.L2.run({'X': feature_vector})
        return state['predictive_cells']
    def encode_location_and_feature(self, location_vector, feature_vector):
        visual_feature_vector = self.encode_feature(feature_vector)
        input_data: Dict[str, Any] = {'G': location_vector, 'F': visual_feature_vector}
        state = self.L5.run(input_data)
        return state['active_cells']
        return state['active_cells']

class SDRFrozenLakeEnvironment:
    def __init__(self, render_mode="human", size=8, agent: Agent|None=None):
        """
        Create an SDR-encoded wrapper around FrozenLake.

        Parameters
        - render_mode: gym render mode
        - size: board size
        - L5, L2: optional shared TemporalPooler instances. When provided, both
          training and evaluation environments can share the exact same poolers.
        """
        self.env = FrozenLakeEnvironment(render_mode=render_mode, size=size)
        self.n_possible_locations = size * size
        self.agent = agent if agent is not None else Agent()
        self.action_space = self.env.env.action_space
        self.observation_space = spaces.MultiBinary(self.agent.observation_size)
        self.current_command = 0

        # Base SDR template used to create deterministic permutations per feature/location

        # Keep existing deterministic mapping behavior
        np.random.seed(123)
        self.location_map = {
            location: np.random.permutation(self.agent.sdr)
            for location in range(self.n_possible_locations)
        }
        # Access the desc attribute safely by getting the underlying FrozenLake environment
        frozen_lake_env = self.env.env.unwrapped
        desc = getattr(frozen_lake_env, 'desc', None)
        if desc is not None:
            features = set(desc.flatten().tolist())
        else:
            # Fallback if desc is not available
            features = [b'S', b'F', b'H', b'G']
        
        self.feature_map = {
            str(feature, "utf-8"): np.random.permutation(self.agent.sdr)
            for feature in features
        }

    def _stimuli_to_obs(self, stimuli):
        desc = getattr(self.env.env.unwrapped, 'desc', None)
        if desc is not None:
            vision_sensor = str(desc.flatten()[stimuli], 'utf-8')
        else:
            # Fallback if desc is not available
            vision_sensor = 'F'  # Default to frozen tile
        current_stimuli = {'location': stimuli, 'vision_sensor': vision_sensor}
        current_location = current_stimuli['location']
        current_feature = current_stimuli['vision_sensor']
        encoded = self.agent.encode_location_and_feature(
            self.location_map[current_location],
            self.feature_map[current_feature]
        )
        return np.array(encoded).flatten().tolist()

    def reset(self, seed=None):
        obs, reward, done, truncated, info, surrounding_tiles = self.env.reset(seed=seed)
        obs = self._stimuli_to_obs(obs)
        return obs, reward, done, truncated, info, surrounding_tiles 

    def step(self, action):
        obs, reward, done, truncated, info, surrounding_tiles = self.env.step(action)
        obs = self._stimuli_to_obs(obs)
        return obs, reward, done, truncated, info, surrounding_tiles 

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

class Environment(ABC):
    """ Abstract base class for environments. """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, motor_commands):
        if not motor_commands:
            self.logger.warning("No motor commands provided to execute.")

        if motor_commands:
            if len(motor_commands):
                self.run_commands(motor_commands)

        current_stimuli = self.receive_sensory_stimuli()
        return current_stimuli

    @abstractmethod
    def run_commands(self, motor_commands):
        pass

    @abstractmethod
    def receive_sensory_stimuli(self) -> Dict[str, Any]:
        ''' Receives sensory stimuli from the 
        actual environment. '''
        pass



if __name__ == '__main__':
    # Create a shared Agent so training and evaluation use the same internal L2/L5
    shared_agent = Agent()

    core = SDRFrozenLakeEnvironment(render_mode="rgb_array", agent=shared_agent)
    env = GymAdapter(core)

    core_eval = SDRFrozenLakeEnvironment(render_mode="rgb_array", agent=shared_agent)
    eval_env = GymAdapter(core_eval)
    n_training_envs = 1
    n_eval_envs = 5
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)

    obs, _ = env.reset()
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    env.close()

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(mean_reward, std_reward)