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

import cv2

from matplotlib import offsetbox

def main(exp_id=0):
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=False, default='tsne_emb_fl_autoencoder.pickle')
  # parser.add_argument('-l','--list', nargs='+', dest='list', help='List of ',
  #                     type=int)
  # parser.add_argument('-s', dest='silent', action='store_true')

  # parser.set_defaults(list=[])    
  # parser.set_defaults(silent=False)
  
  args = parser.parse_args()
  # print(args.input_file, args.list, args.silent)


  with open(args.input_file, 'rb') as handle:
    caravelas_data = pickle.load(handle)

  X_train, y_train, emb2d_train = caravelas_data

  fig, ax = plt.subplots(1,1, figsize=(12,6))
  
  for y in set(list(y_train)):
    label_name = 'Rejected as man-of-war'
    if y:
      label_name = 'Accepted as man-of-war'
    ax.scatter(emb2d_train[y_train==y, 0], emb2d_train[y_train==y, 1], label=label_name)
  # ax.scatter(emb2d_train[:, 0], emb2d_train[:, 1])
  
  # shuffled_order = np.random.choice(np.arange(len(emb2d_train)), len(emb2d_train), replace=False)

  negative_random_index = np.random.choice(np.where(y_train == False)[0], 5, replace=False)
  positive_random_index = np.random.choice(np.where(y_train == True)[0], 5, replace=False)
  
  shuffled_order = np.concatenate((negative_random_index, positive_random_index))
  for i in range(10):
    bounding_box_min = np.min(emb2d_train,axis=0)
    bounding_box_max = np.max(emb2d_train,axis=0)

    # point_idx, img_frame = frames[np.argmax([dataY[i[0]] for i in frames])]
    
    point_idx = shuffled_order[i]
    img_frame = X_train[point_idx]

    xybox = bounding_box_max
    negshift = (bounding_box_max[0] - bounding_box_min[0])*0.1*(i+int(i>=5))
    xybox[0]-=negshift
    imagebox = offsetbox.AnnotationBbox(
        offsetbox.OffsetImage(cv2.resize(img_frame,(50,50))),
        box_alignment=(0,0),
        arrowprops=dict(arrowstyle="->"),
        # xy=tuple(emb2d[point_idx]),
        # xycoords='figure fraction',
        # xy=(0.3,0.3),
        # xybox=(0.75, 0.75))
        xy=emb2d_train[point_idx],
        xybox=xybox)
        
    ax.add_artist(imagebox)


  plt.legend(loc='lower left')
  # plt.show()
  plt.savefig('test_{}.png'.format(exp_id))


if __name__ == "__main__":
  # for i in range(10):
  #   main(i)
  main()