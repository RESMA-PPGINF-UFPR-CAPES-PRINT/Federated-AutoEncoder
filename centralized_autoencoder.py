import pandas as pd
import numpy as np
import argparse

import os

import pandas as pd
import matplotlib.pyplot as plt
import cv2

# import pickle
import pickle5 as pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


from sklearn.manifold import TSNE

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def generate_autoencoder_model(data_shape):
  
    # model = Sequential([
    #   layers.Rescaling(1./255, input_shape=dataX[0].shape),
    #   layers.Conv2D(16, 3, padding='same', activation='relu'),
    #   layers.MaxPooling2D(),
    #   # layers.Conv2D(32, 3, padding='same', activation='relu'),
    #   # layers.MaxPooling2D(),
    #   # layers.Conv2D(64, 3, padding='same', activation='relu'),
    #   # layers.MaxPooling2D(),
    #   layers.Flatten(),
    #   layers.Dense(128, activation='relu'),
    #   layers.Dense(num_classes)
    # ])


    model = layers.Input(shape=data_shape)
    model_input = model
    # model = layers.Rescaling(1./255)(model)
    model = layers.Conv2D(16, 3, padding='same', activation='relu')(model)

    model = layers.MaxPooling2D()(model)
    conv_model = model
    conv_model_shape = conv_model.shape
    
    model = layers.Flatten()(model)
    model = layers.Dense(135*3, activation='relu')(model)
    
    model = layers.Reshape((15,9,3))(model)
    model = layers.Conv2DTranspose(16, 3, strides=(9,15), activation="relu", padding="same")(model)
    model = layers.Conv2D(3, 3, activation="relu", padding="same")(model)

    # model = layers.Dense(conv_model_shape[1]*conv_model_shape[2]*conv_model_shape[3], activation='relu')(model)
    # model = layers.Reshape((conv_model_shape[1],conv_model_shape[2],conv_model_shape[3]))(model)
    # model = layers.Dense(135*135*3, activation='relu')(model)

    # model = layers.Rescaling(255)(model)

    model = tf.keras.Model(model_input, model)

    opt = keras.optimizers.Adam()
    # opt = keras.optimizers.SGD()
    # opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.SGD(learning_rate=0.01)

    # model.compile(optimizer=opt,
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #             # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #             # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    #             metrics=['accuracy'],)

    
    model.compile(optimizer=opt,
                loss='mse')

    # model.fit(X_train, y_train, epochs=100, shuffle=False, batch_size=1)
    # model.fit(X_train, y_train, epochs=100, shuffle=False, batch_size=16)
    # model.fit(X_train, y_train, epochs=1, shuffle=False, batch_size=1)

    return model

def plot_model_latent_space(model, X_train, y_train):
  inter_output_model = keras.Model(model.input, model.get_layer(index = 4).output )
  # inter_output_model = keras.Model(model.input, model.get_layer(index = 8).output )
  
  

  # test_eval = model.evaluate(X_test, y_test)
  # print(test_eval)

  latent_layer = inter_output_model.predict(X_train)
  emb2d_train = TSNE(n_components=2).fit_transform(latent_layer)
  shuffled_order = np.random.choice(np.arange(len(emb2d_train)), len(emb2d_train), replace=False)
  
  fig, ax = plt.subplots(1,1)
  
  for y in set(list(y_train)):
    ax.scatter(emb2d_train[y_train==y, 0], emb2d_train[y_train==y, 1], label=y)
  
  plt.legend()
  plt.show()

  data_to_file = (X_train, y_train, emb2d_train)
  with open('tsne_emb_centralized_autoencoder.pickle', 'wb') as handle:
    pickle.dump(data_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  # parser.add_argument('-f', '--input_file', dest='input_file', type=str,
  #                     required=True)
  # parser.add_argument('-l','--list', nargs='+', dest='list', help='List of ',
  #                     type=int)
  # parser.add_argument('-s', dest='silent', action='store_true')

  # parser.set_defaults(list=[])    
  # parser.set_defaults(silent=False)
  
  args = parser.parse_args()
  # print(args.input_file, args.list, args.silent)


  with open('caravelas_dataset_rescaled.pickle', 'rb') as handle:
    caravelas_data = pickle.load(handle)

  dataX, dataY = caravelas_data

  dataY = dataY == 'ACEITA'
  print(dataX.shape)

  

  num_classes = len(set(list(dataY)))

  n_clients = 10
  kf = StratifiedKFold(n_splits=n_clients)
  splits = list(kf.split(dataX, dataY))

  model = generate_autoencoder_model(dataX[0].shape)

  X_train, X_test, y_train, y_test = train_test_split(
    dataX, dataY, test_size=0.33, random_state=42)

  model.fit(X_train, X_train, epochs=3, shuffle=False, batch_size=1)

  plot_model_latent_space(model, X_train, y_train)

if __name__ == "__main__":
  main()