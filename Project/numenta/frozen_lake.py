import sys
from pathlib import Path
from numenta.utils import get_logger
from abc import ABC, abstractmethod

sys.path.append(str(Path(__file__).parents[1]))

import gymnasium as gym
from gymnasium import spaces
from numenta.agent import Environment 
    

class GymAdapter(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, core):
        super().__init__()
        self.core = core                      # your non-gym simulator
        self.observation_space = self.core.observation_space  # map your API here
        self.action_space = self.core.action_space

        # This will catch many common issues
        # try:
        #     check_env(self)
        #     print("Environment passes all checks!")
        # except Exception as e:
        #     print(f"Environment has issues: {e}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, reward, done, truncated, info, surrounding_tiles = self.core.reset(seed=seed)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info, surrounding_tiles = self.core.step(action)  # map your API here
        terminated = bool(done) and not info.get("TimeLimit.truncated", False)
        return obs, reward, done, terminated, info 

    def render(self): return self.core.render()
    def close(self):  self.core.close()

class FrozenLakeEnvironment(Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    col = 0     #Data to hold the current column the agent occupies
    row = 0     # Data to hold the current row the agent occupies
    def __init__(self, render_mode="human", size=8):
        super(FrozenLakeEnvironment, self).__init__()
    #def __init__(self):
        #generating the frozen lake environment
        self.env = gym.make(
            'FrozenLake-v1',
            desc=None,
            map_name="4x4",
            is_slippery=False,
            render_mode=render_mode)

        self.action_space = self.env.action_space  # action_space attribute
        self.observation_space = self.env.observation_space  # observation_space attribute
        self._step_out = None
        self.reset()
        #self.col = 0 #Agents column position
        #self.row = 0 #Agents row position

    #Reseting the environment to start a new episode
    def reset(self, seed=None):
        #interacting with the environment by using Reset()
        state, info = self.env.reset(seed=seed)
        done = False
        truncated = False
        self.col, self.row = 0, 0 #Assuming the agent is started at (0,0)
        surrounding_tiles = self.get_surrounding_tiles(self.row, self.col)
        reward = 0
        return state, reward, done, truncated, info, surrounding_tiles

    # perform an action in environment:
    def step(self, action, render=True):
        if render:
            self.render()
        state, reward, done, truncated, info = self.env.step(action if action else 0) # type: ignore
        self.update_position(state) #updating the agents position based on the action
        surrounding_tiles = self.get_surrounding_tiles(self.row, self.col)
        return state, reward, done, truncated, info, surrounding_tiles     # action chosen by the agent

    def run_commands(self, motor_commands):    
        action = motor_commands['move']
        self._step_out = self.step(action)
    
    def receive_sensory_stimuli(self):
        if self._step_out is None:
            self._step_out = self.reset()
        state, reward, done, truncated, info, surrounding_tiles = self._step_out
        order = ['left', 'down', 'right', 'up']
        order = ['current']
        stimuli = {'vision_sensor': ''.join([surrounding_tiles[x] for x in order]),
                   'location': state, 
                   'reward': reward
                  }
        return stimuli

    # render environment's current state:
    def render(self):
        self.env.render()

    # close the environment:
    def close(self):
        self.env.close()

    def update_position(self, state):
        #updating the agents position based on the action taken
        desc = self.env.unwrapped.desc # type: ignore
        self.row, self.col = state // desc.shape[1], state % desc.shape[1]

    def get_surrounding_tiles(self, row, col):
        #gathering information about the tiles surrounding the agent
        desc = self.env.unwrapped.desc # type: ignore
        surrounding_tiles = {}
        directions = {
            "up":(max(row - 1, 0), col),
            "right":(row,min(col + 1,desc.shape[1] - 1)),
            "down":(min(row + 1, desc.shape[0] - 1), col),
            "left":(row,max(col - 1, 0)),
            "current":(row, col)
        }
        for direction, (r,c) in directions.items():
            surrounding_tiles[direction] = desc[r,c].decode('utf-8') #Decode byte to string
        return surrounding_tiles
    