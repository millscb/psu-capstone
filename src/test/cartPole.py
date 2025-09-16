import gymnasium as gym
import numpy as np
import tensorflow as tf

from tensorflow import keras #type: ignore
from pathlib import Path


def play_one_step(env, model, obs, loss_fn):
    
    with tf.GradientTape() as tape:
       x = tf.convert_to_tensor(obs, dtype=tf.float32)  
       x = tf.reshape(x, (1, -1))                       
       left_proba = model(x) # shape (1,1)

       action_bool = tf.random.uniform(tf.shape(left_proba)) < left_proba # bool (1,1)
       y_target = tf.cast(tf.logical_not(action_bool), tf.float32) # 1.0 if go left, else 0.0
       loss = loss_fn(y_target, left_proba)

    
    grads = tape.gradient(loss, model.trainable_variables)
    act = int(action_bool[0,0].numpy())
    obs, reward, terminated, truncated, info = env.step(act)
    done = bool(terminated or truncated)
    
    return obs, reward, done, grads


def play_multiple_episodes(env, model, loss_fn, episodes, max_steps):

    all_rewards = []
    all_grads = []

    for episode in range(episodes):
        current_rewards = []
        current_grads = []
        obs, info = env.reset()

        print(f"Episode {episode+1} starting...")
        #print(obs)
                
        for step in range(max_steps):
            obs, reward, done, grads = play_one_step(env, model, obs, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
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


def train(loss,opt,env, ops):
    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(env,model,loss, n_episode_per_update, n_max_steps)
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        print(f"Iteration {iteration}: mean reward {np.mean([len(rewards) for rewards in all_rewards])}")

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
            #print(f"Mean gradients for variable {var_index}: {mean_grads.numpy()}")
        opt.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    return model








if __name__ == "__main__":

  
    enviroment = gym.make("CartPole-v1")

    observation, info = enviroment.reset()

    n_inputs = 4
    n_hidden = 4
    n_outputs = 1
    n_iterations = 150
    n_episode_per_update = 10
    n_max_steps = 200
    discount_rate = 0.95

    model = keras.Sequential(
        [
            keras.layers.Dense(n_hidden, activation="elu", input_shape=[n_inputs]),
           
            keras.layers.Dense(n_outputs, activation="sigmoid"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)


    model = train(loss_fn,optimizer,enviroment, observation)

    print(model.summary())


  

     
        

       
       

  


 
   