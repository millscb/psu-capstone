import tensorflow as tf
import gymnasium as gym
import numpy as np

from tensorflow import keras # type: ignore

env = gym.make('CartPole-v1', render_mode='human')
obs, info = env.reset()

env.render()

print(obs)

inputs = 4 # shape of observation space

model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(inputs,), activation='elu'),
    keras.layers.Dense(1, activation='sigmoid')
])

def play_one_step(env, model, obs, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1,1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss_value = loss_fn(y_target, left_proba)
    grads = tape.gradient(loss_value, model.trainable_variables)
    # Gymnasium returns (obs, reward, terminated, truncated, info)
    obs, reward, terminated, truncated, info = env.step(int(action[0,0].numpy()))
    done = bool(terminated or truncated)
    return obs, reward, done, grads, loss_value
    