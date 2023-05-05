import pandas as pd
import numpy as np
import argparse

import os

import pandas as pd
import matplotlib.pyplot as plt
import cv2

import pickle


from skimage.transform import resize

from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  # parser.add_argument('-f', '--input_file', dest='input_file', type=str,
  #                     required=True)
  # parser.add_argument('-l','--list', nargs='+', dest='list', help='List of ',
  #                     type=int)
  # parser.add_argument('-s', dest='silent', action='store_true')
  parser.add_argument('-p', dest='process', action='store_true')
  parser.add_argument('-r', dest='rescale', action='store_true')

  # parser.set_defaults(list=[])    
  # parser.set_defaults(silent=False)
  
  args = parser.parse_args()
  # print(args.input_file, args.list, args.silent)

  if args.process:
    negative_factor = 1.0

    total_data_X = []
    total_data_Y = []
    total_negative = 0
    for file in os.listdir('.'):
      # check only text files
      if file.endswith('.xlsx'):
        hashtag_name = "#"+file.split("_")[1]
        print(file, hashtag_name)
        # df = pd.read_excel(file, encoding = "ISO-8859-1")
        df = pd.read_excel(file)
        # print(df.head())
        data = df[['TIMESTAMP', 'ROTULO_BINARIO_HELO']].to_numpy()

        for fname, label in data:
          fpath = 'raw-data/{}/{}.jpg'.format(hashtag_name, fname)
          if not os.path.exists(fpath):
            continue

          if label == 'REJEITADA':
            continue
          else:
            total_negative += 1
          
          img = cv2.imread(fpath)
          total_data_X.append(img)
          total_data_Y.append(label)

          # # Output img with window name as 'image'
          # cv2.imshow('image', img)
          # # Maintain output window until
          # # user presses a key
          # cv2.waitKey(0)       
          # # Destroying present windows on screen
          # cv2.destroyAllWindows()

    
    print(len(total_data_Y))

    total_negative_to_select = int(negative_factor*len(total_data_Y))
    selected_negative = np.random.choice(np.arange(total_negative), total_negative_to_select, replace=False)

    current_negative = 0
    for file in os.listdir('.'):
      if file.endswith('.xlsx'):
        hashtag_name = "#"+file.split("_")[1]
        print(file, hashtag_name)
        df = pd.read_excel(file)
        data = df[['TIMESTAMP', 'ROTULO_BINARIO_HELO']].to_numpy()

        for fname, label in data:
          fpath = 'raw-data/{}/{}.jpg'.format(hashtag_name, fname)
          if not os.path.exists(fpath):
            continue

          if label == 'REJEITADA':
            if current_negative in selected_negative:
              img = cv2.imread(fpath)
              total_data_X.append(img)
              total_data_Y.append(label)
            
            current_negative += 1

    total_data_X = np.array(total_data_X)
    total_data_Y = np.array(total_data_Y)
    print(len(total_data_Y))


    data_to_file = (total_data_X, total_data_Y)
    with open('caravelas_dataset.pickle', 'wb') as handle:
      pickle.dump(data_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

  if args.rescale:
    with open('caravelas_dataset.pickle', 'rb') as handle:
      caravelas_data = pickle.load(handle)

    dataX, dataY = caravelas_data

    new_dataX = np.array([resize(x, (135, 135)) for x in tqdm(dataX)])
    dataX = new_dataX

    data_to_file = (dataX, dataY)
    with open('caravelas_dataset_rescaled.pickle', 'wb') as handle:
      pickle.dump(data_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('.pickle', 'rb') as handle:
#   b = pickle.load(handle)






if __name__ == "__main__":
  main()