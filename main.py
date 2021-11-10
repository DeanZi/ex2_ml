import sys
import numpy as np


class Model:
  def __init__(self, data_x, data_y, test_data, k=None):
    self.data_x = data_x
    self.data_y = data_y
    self.test_data = test_data
    self.k = k

  def train(self):
    pass

  def predict(self):
    pass

class KNN(Model):

  def predict(self):







def receive_data(examples_file, examples_labels_file, test_data_file):
  examples_x = np.loadtxt(examples_file, dtype=float)
  examples_y = np.loadtxt(examples_labels_file, dtype=int)
  test_data = np.loadtxt(test_data_file, dtype=float)
  return examples_x, examples_y, test_data

if __name__ == '__main__':
  data_x, data_y, test_data = receive_data(sys.argv[0], sys.argv[1], sys.argv[2])
  output_file_name = sys.argv[3]
  knn = KNN(data_x, data_y, test_data)






