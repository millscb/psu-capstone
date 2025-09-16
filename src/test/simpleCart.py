
import time
import gymnasium as gym
from stable_baselines3 import PPO



def train():
    """
    Train a RL agent on CartPole-v1 
    """
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    env.close()
    return model


if __name__ == "__main__":

    start = time.time()
    model = train()
    end = time.time()
    print(f"Training took {end - start:.2f} seconds")
    model.save("C:\\repos\\SWENG480\\models\\ppo_cartpole")



