import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
obs = env.reset()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print(obs)


obs = env.reset()