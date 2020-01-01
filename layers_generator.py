import tensorflow as tf
from tensorflow.keras import layers

def one_hidden_layer(nb_percep,feature_layer):
     """
     return a model with one hidden layer.
     """
     return  tf.keras.Sequential([
                            feature_layer,
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(1, activation='sigmoid')
                            ])

def tow_hidden_layers(nb_percep,activationFunction,feature_layer):
     """
     return a model with two hidden layers.
     """
     return  tf.keras.Sequential([
                            feature_layer,
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(1, activation='sigmoid')
                            ])
                         
def three_hidden_layers(nb_percep,activationFunction,feature_layer):
     """
     return a model with three hidden layers.
     """
     return  tf.keras.Sequential([
                            feature_layer,
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(1, activation='sigmoid')
                            ])
def four_hidden_layers(nb_percep,activationFunction,feature_layer):
     """
     return a model with four hidden layers.
     """
     return  tf.keras.Sequential([
                            feature_layer,
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(1, activation='sigmoid')
                            ])

def five_hidden_layers(nb_percep,activationFunction,feature_layer):
     """
     return a model with five hidden layers.
     """
     return  tf.keras.Sequential([
                            feature_layer,
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(nb_percep, activation='relu'),
                            layers.Dense(1, activation='sigmoid')
                            ])
