import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Reduce TF verbosity

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tensorflow import keras  # type: ignore
from pathlib import Path


def play_one_step(env, model, obs):
    """Play one environment step, returning transition and gradients.

    Implements a REINFORCE-style policy gradient for a Bernoulli action head:
      - model outputs probability of taking action 0 ("left").
      - sample action from Bernoulli(p_left).
      - compute loss = -log( prob(sampled_action) ) so that later weighting by reward
        and applying gradient descent performs ascent on expected return.
    """
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(obs, dtype=tf.float32)
        x = tf.reshape(x, (1, -1))
        left_proba = model(x)  # shape (1,1)
        action_left_bool = tf.random.uniform(tf.shape(left_proba)) < left_proba  # bool (1,1)
        # Probability of chosen action:
        prob_action = tf.where(action_left_bool, left_proba, 1.0 - left_proba)
        log_prob = tf.math.log(prob_action + 1e-8)
        loss = -log_prob  # Gradient of this will be -grad(log_prob)

    grads = tape.gradient(loss, model.trainable_variables)
    # Convert sampled boolean to CartPole action id: 0 (left) if True else 1 (right)
    act = 0 if bool(action_left_bool[0, 0].numpy()) else 1
    obs, reward, terminated, truncated, info = env.step(act)
    done = bool(terminated or truncated)
    return obs, reward, done, grads


def play_multiple_episodes(env, model, episodes, max_steps, render=True):

    all_rewards = []
    all_grads = []

    for episode in range(episodes):
        current_rewards = []
        current_grads = []
        obs, info = env.reset()

        print(f"Episode {episode+1} starting...")
        #print(obs)
                
        for step in range(max_steps):
            obs, reward, done, grads = play_one_step(env, model, obs)
            current_rewards.append(reward)
            current_grads.append(grads)
            if render:
                env.render()
            if done:
                print(f"Episode {episode+1} finished after {step+1} steps")
                break

        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    
    return all_rewards, all_grads
       


def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards)-2, -1, -1):
        discounted[step] += discounted[step+1] * discount_rate

    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


def train(
    model,
    env,
    optimizer,
    n_iterations,
    n_episode_per_update,
    n_max_steps,
    discount_rate,
    render=True,
    early_stop_mean=200.0,
    early_stop_patience=2,
):
    """Train policy with simple early stopping.

    Stops if mean episode length >= early_stop_mean for early_stop_patience
    consecutive iterations.
    """
    consecutive_hits = 0
    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(
            env, model, n_episode_per_update, n_max_steps, render=render
        )
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        mean_ep_len = np.mean([len(r) for r in all_rewards])
        print(f"Iteration {iteration}: mean episode length {mean_ep_len:.1f}")

        # Aggregate gradients weighted by normalized returns
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [
                    final_reward * all_grads[episode_index][step][var_index]
                    for episode_index, final_rewards in enumerate(all_final_rewards)
                    for step, final_reward in enumerate(final_rewards)
                ],
                axis=0,
            )
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

        # Early stopping check
        if mean_ep_len >= early_stop_mean:
            consecutive_hits += 1
            print(
                f"Early stop counter: {consecutive_hits}/{early_stop_patience} (mean >= {early_stop_mean})"
            )
            if consecutive_hits >= early_stop_patience:
                print(
                    f"Early stopping triggered at iteration {iteration}: mean episode length {mean_ep_len:.1f}"
                )
                break
        else:
            consecutive_hits = 0
    return model








if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and save a CartPole policy gradient model.")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations (policy updates)")
    parser.add_argument("--episodes-per-update", type=int, default=10, help="Episodes collected per update")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount rate (gamma)")
    parser.add_argument("--no-render", action="store_true", help="Disable environment rendering")
    parser.add_argument("--save-name", type=str, default="cartpole_policy.keras", help="Filename inside models/ directory")
    parser.add_argument("--load-model", type=str, default=None, help="Path to existing .keras model to continue training (relative or absolute)")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    enviroment = gym.make("CartPole-v1", render_mode=render_mode)
    observation, info = enviroment.reset()

    # Hyperparameters / architecture
    n_inputs = 4
    n_hidden = [8, 4]
    n_outputs = 1
    n_iterations = args.iterations
    n_episode_per_update = args.episodes_per_update
    n_max_steps = args.max_steps
    discount_rate = args.discount

    # Build or load model
    if args.load_model:
        load_path = Path(args.load_model)
        if not load_path.is_absolute():
            load_path = (Path(__file__).resolve().parents[2] / load_path).resolve()
        if not load_path.exists():
            raise FileNotFoundError(f"--load-model path not found: {load_path}")
        print(f"Loading existing model from {load_path}")
        model = keras.models.load_model(load_path)
    else:
        model = keras.Sequential(
            [
                keras.layers.Dense(n_hidden[0], activation="elu", input_shape=[n_inputs]),
                keras.layers.Dense(n_hidden[1], activation="relu"),
                keras.layers.Dense(n_outputs, activation="sigmoid"),
            ]
        )

    # If resuming, keep optimizer fresh unless you need stateful resume; for full resume
    # you'd have to have saved with optimizer state (which .keras does). We can rebuild
    # and recompile; gradients do not require compile here because we use custom tape.
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    model = train(
        model,
        enviroment,
        optimizer,
        n_iterations,
        n_episode_per_update,
        n_max_steps,
        discount_rate,
        render=not args.no_render,
    )

    model.summary()

    # === Simple single-file save ===
    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    simple_path = models_dir / args.save_name
    model.save(simple_path)
    print(f"Model saved to: {simple_path}")


  

     
        

       
       

  


 
   