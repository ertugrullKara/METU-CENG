import scipy.signal as signal

import numpy as np
import os
import h5py


def load_dataset(filename, input_size=1000):
  X, Y = [], []
  data = []
  with h5py.File(filename, 'r') as matfile:
    data = [matfile[element[0]][:] for element in matfile['Part_1']]
  np.random.shuffle(data)
  for idx, cell in enumerate(data):
    print "{}/{} ,".format(str(idx), str(len(data))), cell.shape
    prev_idx = 0
    for idx in range(input_size, cell.shape[0], 1): # range(1000, cell.shape[0], int(input_size/4.))
      if idx - prev_idx != 1000:
        continue
      if idx%1000 == 0:
        print len(X)
      x_window = cell[prev_idx:idx, 0]
      X.append(x_window)
      y_window = cell[prev_idx:idx, 1]
      peaks = signal.find_peaks_cwt(y_window, np.arange(1, 20))
      peaks = y_window[peaks]
      # print peaks
      Y.append([np.max(peaks), np.min(peaks)])
      if(len(Y) >= 10000):
        return np.array(X), np.array(Y)
      # print np.max(peaks), np.min(peaks)
      prev_idx += 1
    break
  return np.array(X), np.array(Y)
  
if __name__ == '__main__':
	# TODO: You can fill in the following part to test your function(s)/dataset from the command line
	filename='dataset/part1.mat'
	X, Y = load_dataset(filename, 54)

