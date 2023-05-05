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

def generate_autoencoder_model(data_shape, num_classes):
  
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


    model = Sequential([
      layers.Rescaling(1./255, input_shape=data_shape),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      # layers.Conv2D(32, 3, padding='same', activation='relu'),
      # layers.MaxPooling2D(),
      # layers.Conv2D(64, 3, padding='same', activation='relu'),
      # layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    opt = keras.optimizers.Adam()

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'],)
                
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
  with open('tsne_emb_fl_classifier.pickle', 'wb') as handle:
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

  local_models = [generate_autoencoder_model(dataX[0].shape, len(set(dataY))) for i in range(n_clients)]
  global_model = generate_autoencoder_model(dataX[0].shape, len(set(dataY)))
  total_rounds = 5

  for round_id in range(total_rounds):
    print("Round ", round_id+1)
    for x in local_models:
      x.set_weights(global_model.get_weights())

    for i, (train_index, test_index) in enumerate(splits):

      X_train, X_test, y_train, y_test = train_test_split(
      dataX[test_index], dataY[test_index], test_size=0.1, random_state=42)

      

      # model = generate_autoencoder_model(dataX[0].shape)
      model = local_models[i]

      
      # model.fit(X_train, X_train, epochs=2, shuffle=False, batch_size=1)
      model.fit(X_train, y_train, epochs=100, shuffle=False, batch_size=16)

      # plot_model_latent_space(model, X_train, y_train)
      # local_models.append(model)
      # print(np.array(model.get_weights()[0], dtype=object).shape)


    avg_weights = sum([np.array(x.get_weights(), dtype=object) for x in local_models])/len(local_models)
    # avg_weights = sum([np.array(models[x].get_weights())*avg_contribution[x] for x in models])/len(models)
    global_model.set_weights(avg_weights)
    

  plot_model_latent_space(global_model, dataX, dataY)
  test_eval = global_model.evaluate(X_test, y_test)
  print(test_eval)

if __name__ == "__main__":
  main()