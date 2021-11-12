import sys
import numpy as np


class Model:
    def __init__(self, data_x, data_y, test_data, k=None, learning_rate=1):
        self.data_x = data_x
        self.data_y = data_y
        self.test_data = test_data
        self.k = k
        self.learning_rate = learning_rate

    def train(self):
        pass

    def predict(self):
        pass


class KNN(Model):

    def predict(self):
        predictions = []
        for flower in self.test_data:
            distances = {}
            for i, example in enumerate(self.data_x):
                distances[i] = np.linalg.norm(flower - example)
            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            labels_of_neigbours = []
            for i in range(self.k):
                labels_of_neigbours.append(self.data_y[list(sorted_distances.keys())[i]])
            predictions.append(max(set(labels_of_neigbours), key=labels_of_neigbours.count))
        return predictions


class Perceptron(Model):

    def train(self):
        weights = np.zeros([3, 5])
        for epoch in range(1000):
            for example, label in zip(self.data_x, self.data_y):
                y_hat = np.argmax(np.dot(weights, example.transpose()))
                if y_hat != label:
                    weights[label] = weights[label] + self.learning_rate * example
                    weights[y_hat] = weights[y_hat] - self.learning_rate * example
        return weights

    def predict(self, weights):
        predictions = []
        for test_example in self.test_data:
            predictions.append(np.argmax(np.dot(weights, test_example.transpose())))
        return predictions


def receive_data(examples_file, examples_labels_file, test_data_file):
    tmp_data_examples_x = []
    with open(examples_file) as tmp_file:
        for line in tmp_file:
            tmp_data_examples_x.append([list(map(float, x.split(','))) for x in line.split(' ')])

    examples_x = np.array(tmp_data_examples_x)
    examples_y = np.loadtxt(examples_labels_file, dtype=int)

    tmp_data_test_x = []
    with open(test_data_file) as tmp_file:
        for line in tmp_file:
            tmp_data_test_x.append([list(map(float, x.split(','))) for x in line.split(' ')])

    test_data = np.array(tmp_data_test_x)
    return examples_x, examples_y, test_data


if __name__ == '__main__':
    data_x, data_y, test_data = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    output_file_name = sys.argv[4]
    knn = KNN(data_x, data_y, test_data, 3)
    knn_test_predictions = knn.predict()
    #print(knn_test_predictions)
    perceptron = Perceptron(data_x, data_y, test_data, learning_rate=0.5)
    perceptron_weights = perceptron.train()
    perceptron_test_predictions = perceptron.predict(perceptron_weights)
    print(perceptron_test_predictions)

    '''
    TODO :
    
    - Try adding bias 
    
    
    
    
    '''
