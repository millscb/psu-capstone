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

def watch(model_path: str, episodes: int = 5, fps: int = 60, deterministic: bool = True):
    """
    Load a trained model and render its performance.
    Press Ctrl+C in the terminal to stop early.
    """
    model = PPO.load(model_path)
    env = gym.make("CartPole-v1", render_mode="human")
    frame_delay = 1.0 / fps
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_f = float(reward)
            episode_reward += reward_f
            done = terminated or truncated
            step += 1
            if frame_delay > 0:
                time.sleep(frame_delay)
        print(f"Episode {ep+1}: steps={step} reward={episode_reward:.1f}")
    env.close()

if __name__ == "__main__":

    start = time.time()
    model = train()
    end = time.time()
    print(f"Training took {end - start:.2f} seconds")
    save_path = r"C:\repos\SWENG480\models\ppo_cartpole"
    model.save(save_path)
    print("Saved model to", save_path)
    # Watch it perform
    watch(save_path, episodes=3, fps=60, deterministic=True)



