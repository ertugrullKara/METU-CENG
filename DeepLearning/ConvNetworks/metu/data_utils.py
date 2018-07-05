import scipy.signal as signal

import numpy as np
import os
import h5py


def load_dataset(filename, input_size=1000, step_size=100):
  X, Y = [], []
  data = []
  with h5py.File(filename, 'r') as matfile:
    data = [matfile[element[0]][:] for element in matfile['Part_1']]
  np.random.shuffle(data)
  while 1:
    for idx, cell in enumerate(data):
      if len(X) % 1000 == 0:
        print len(X)
      if len(data) < input_size:
        continue
      # print "{}/{} ,".format(str(idx), str(len(data))), cell.shape
      idx = np.random.randint(len(data)-input_size)
      x_window = cell[idx:idx+input_size, 0]
      X.append(x_window)
      y_window = cell[idx:idx+input_size, 1]
      peaks = signal.find_peaks_cwt(y_window, np.arange(1, 20))
      peaks = y_window[peaks]
      # print peaks
      Y.append([np.max(peaks), np.min(peaks)])
      if(len(Y) >= 30000):
        return np.array(X), np.array(Y)
      # print np.max(peaks), np.min(peaks)
      break
      # break
  return np.array(X), np.array(Y)
  
if __name__ == '__main__':
	# TODO: You can fill in the following part to test your function(s)/dataset from the command line
	filename='dataset/part1.mat'
	X, Y = load_dataset(filename, 54)

