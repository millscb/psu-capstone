import gymnasium as gym

def main():
    env = gym.make("CartPole-v1", render_mode="human")  # requires pygame (installed via classic-control)
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = env.action_space.sample()  # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    env.close()
    print(f"Episode reward: {total_reward}")

if __name__ == "__main__":
    main()