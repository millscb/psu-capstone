import tensorflow as tf
import gymnasium as gym

#from tensorflow import keras

env = gym.make('CartPole-v1')
obs = env.reset()

env.render()

print(obs)

inputs = 4 # shape of observation space

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, input_shape=(inputs,), activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

