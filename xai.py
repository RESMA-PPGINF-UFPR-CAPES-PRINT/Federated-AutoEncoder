# https://towardsdatascience.com/explainable-ai-xai-a-guide-to-7-packages-in-python-to-explain-your-models-932967f0634b
# https://towardsdatascience.com/hands-on-tutorial-for-applying-grad-cams-for-explaining-image-classifiers-using-keras-and-cbdcef68bb89


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

import matplotlib.cm as c_map
from tensorflow.keras.preprocessing import image
from IPython.display import Image, display

def get_heatmap(vectorized_image, model, last_conv_layer, pred_index=None):
    '''
    Function to visualize grad-cam heatmaps
    '''
    gradient_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    # Gradient Computations
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = gradient_model(vectorized_image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # Normalize the heatmap
    return heatmap.numpy()



def superimpose_gradcam(img, heatmap, output_path="grad_cam_image.jpg", alpha=0.4):
    '''
    Superimpose Grad-CAM Heatmap on image
    '''
    # img = image.load_img(img_path)
    # img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap) # Back scaling to 0-255 from 0 - 1
    jet = c_map.get_cmap("jet") # Colorizing heatmap
    jet_colors = jet(np.arange(256))[:, :3] # Using RGB values
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    
    superimposed_img = jet_heatmap * alpha + img # Superimposing the heatmap on original image
    superimposed_img = np.concatenate((img*255, superimposed_img), axis=1)
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(output_path) # Saving the superimposed image
    display(Image(output_path)) # Displaying Grad-CAM Superimposed Image



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

  X_train, X_test, y_train, y_test = train_test_split(
    dataX, dataY, test_size=0.33, random_state=42)

  print(X_train.shape)
  print(X_test.shape)

  num_classes = len(set(list(dataY)))

  model = Sequential([
    layers.Rescaling(1./255, input_shape=dataX[0].shape),
    layers.Conv2D(16, 3, padding='same', activation='relu', name='last_conv_layer'),
    layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(64, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dense(num_classes)
    layers.Dense(num_classes, activation='softmax')
  ])

  opt = keras.optimizers.Adam()
  # opt = keras.optimizers.Adam(learning_rate=0.0001)
  # opt = keras.optimizers.SGD(learning_rate=0.01)
  model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'],)


  categoricalY = tf.keras.utils.to_categorical(y_train, num_classes = num_classes)

  # model.fit(X_train, y_train, epochs=100, shuffle=False, batch_size=1)
  # model.fit(X_train, categoricalY, epochs=100, shuffle=False, batch_size=16)
  model.fit(X_train, categoricalY, epochs=1000, shuffle=False, batch_size=16)
  # model.fit(X_train, y_train, epochs=1, shuffle=False, batch_size=16)
  
  last_conv_layer = 'last_conv_layer'


  for i, (x, y) in enumerate(zip(X_train, y_train)):
    l = 'Positivo' if y else 'Negativo'
    
    # plt.matshow(get_heatmap([X_train[0].reshape(-1, *X_train[0].shape)], model, last_conv_layer))
    output_path = 'xai/img_train_{}_{}.jpg'.format(l, i)
    superimpose_gradcam(x, get_heatmap([x.reshape(-1, *x.shape)], model, last_conv_layer), output_path=output_path)
    # plt.show()

  for i, (x, y) in enumerate(zip(X_test, y_test)):
    l = 'Positivo' if y  else 'Negativo'
    
    # plt.matshow(get_heatmap([X_train[0].reshape(-1, *X_train[0].shape)], model, last_conv_layer))
    output_path = 'xai/img_test_{}_{}.jpg'.format(l, i)
    superimpose_gradcam(x, get_heatmap([x.reshape(-1, *x.shape)], model, last_conv_layer), output_path=output_path)
    # plt.show()

  # exit()

  inter_output_model = keras.Model(model.input, model.get_layer(index = 4).output )
  # inter_output_model = keras.Model(model.input, model.get_layer(index = 8).output )

  
  categoricalY = tf.keras.utils.to_categorical(y_test, num_classes = num_classes)
  test_eval = model.evaluate(X_test, categoricalY)
  print(test_eval)

  latent_layer = inter_output_model.predict(X_train)
  emb2d_train = TSNE(n_components=2).fit_transform(latent_layer)
  shuffled_order = np.random.choice(np.arange(len(emb2d_train)), len(emb2d_train), replace=False)
  
  fig, ax = plt.subplots(1,1)
  
  for y in set(list(y_train)):
    ax.scatter(emb2d_train[y_train==y, 0], emb2d_train[y_train==y, 1], label=y)
  
  plt.legend()
  plt.show()

  data_to_file = (X_train, y_train, emb2d_train)
  with open('tsne_emb_centralized_classifier.pickle', 'wb') as handle:
    pickle.dump(data_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
  main()