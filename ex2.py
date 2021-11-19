import math
import sys
import numpy as np


class Model:
    def __init__(self, k=None, learning_rate=1, lambda_svm=1):
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_svm = lambda_svm

    def predict(self, weights, test_data):
        predictions = []
        for test_example in test_data:
            predictions.append(np.argmax(np.dot(weights, test_example.transpose())))
        return predictions


class KNN(Model):

    def predict(self, data_x, data_y, test_data):
        predictions = []
        for flower in test_data:
            distances = {}
            for i, example in enumerate(data_x):
                distances[i] = np.linalg.norm(flower - example)
            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            labels_of_neigbours = []
            for i in range(self.k):
                labels_of_neigbours.append(data_y[list(sorted_distances.keys())[i]])
            predictions.append(max(set(labels_of_neigbours), key=labels_of_neigbours.count))
        return predictions


class Perceptron(Model):

    def train(self, data_x, data_y):
        weights = np.zeros([3, 5])
        bias = 0
        initial_learning_rate = self.learning_rate
        for epoch in range(2000):
            # random_state = np.random.get_state()
            # np.random.shuffle(data_x)
            # np.random.set_state(random_state)
            # np.random.shuffle(data_y)
            self.learning_rate = (1 / (1 + epoch)) * initial_learning_rate
            for example, label in zip(data_x, data_y):
                y_hat = np.argmax(np.dot(weights, example.transpose()) + bias)
                if y_hat != label:
                    weights[label] = weights[label] + self.learning_rate * example
                    weights[y_hat] = weights[y_hat] - self.learning_rate * example
                    bias += label
        return weights


class SVM(Model):

    def train(self, data_x, data_y):
        weights = np.zeros([3, 5])
        classifications = [0, 1, 2]
        bias = 0
        initial_learning_rate = self.learning_rate
        for epoch in range(2000):
            # random_state = np.random.get_state()
            # np.random.shuffle(data_x)
            # np.random.set_state(random_state)
            # np.random.shuffle(data_y)
            self.learning_rate = (1 / (1 + epoch)) * initial_learning_rate
            for example, label in zip(data_x, data_y):
                y_hat = np.argmax(np.dot(weights, example.transpose()) + bias)
                if y_hat != label:
                    weights[label] = (1 - self.learning_rate * self.lambda_svm) * weights[
                        label] + self.learning_rate * example
                    weights[y_hat] = (1 - self.learning_rate * self.lambda_svm) * weights[
                        y_hat] - self.learning_rate * example
                    bias += label
                    for classification in classifications:
                        if classification != label and classification != y_hat:
                            weights[classification] = (1 - self.learning_rate * self.lambda_svm) * weights[
                                classification]
                # else:
                #     for classification in classifications:
                #         weights[classification] = (1 - self.learning_rate * self.lambda_svm) * weights[classification]
        return weights


class PA(Model):

    def train(self, data_x, data_y):
        weights = np.zeros([3, 5])
        bias = 0
        for epoch in range(15):
            random_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(random_state)
            np.random.shuffle(data_y)
            for example, label in zip(data_x, data_y):
                y_hat = np.argmax(np.dot(weights, example.transpose()) + bias)
                if y_hat != label:
                    hinge_loss = max(0, 1 - np.dot(weights[label], example.transpose()) + np.dot(weights[y_hat],
                                                                                                 example.transpose()))
                    tau = hinge_loss / (2 * (np.linalg.norm(example) ** 2))
                    weights[label] = weights[label] + tau * example
                    weights[y_hat] = weights[y_hat] - tau * example
                    bias += label
        return weights


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


def print_output_file(knn_test_predictions, perceptron_test_predictions, svm_test_predictions, pa_test_predictions,
                      output_file):
    with open(output_file, 'w') as output_file:
        for knn_yhat, perceptron_yhat, svm_yhat, pa_yhat in zip(knn_test_predictions, perceptron_test_predictions,
                                                                svm_test_predictions, pa_test_predictions):
            output_file.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")


def validate(model, data_x, data_y, k=5):
    accuracy_per_fold = []
    len_of_validation = math.ceil(len(data_x) / k)
    for _ in range(k):
        # random_state = np.random.get_state()
        # np.random.shuffle(data_x)
        # np.random.set_state(random_state)
        # np.random.shuffle(data_y)
        validation_x = []
        validation_y = []
        for index_to_validation in range(len_of_validation):
            validation_x.append(data_x[index_to_validation])
            validation_y.append(data_y[index_to_validation])
        trainning_x = []
        trainning_y = []
        for index_to_trainning in range(len_of_validation, len(data_x)):
            trainning_x.append(data_x[index_to_trainning])
            trainning_y.append(data_y[index_to_trainning])
        if not isinstance(model, KNN):
            model_weights = model.train(trainning_x, trainning_y)
            model_predictions = model.predict(model_weights, validation_x)
        else:
            model_predictions = model.predict(trainning_x, trainning_y, validation_x)
        successes = sum(1 for i, j in zip(validation_y, model_predictions) if i == j)
        accuracy_per_fold.append((successes / len_of_validation) * 100)
    return sum(accuracy_per_fold) / len(accuracy_per_fold)



def find_minmax(data):
    minmax_per_feature = list()
    for i in range(5):
        all_values_per_feature = []
        for flower in data:
            all_values_per_feature.append(flower[0][i])
        min_value_for_feature = min(all_values_per_feature)
        max_value_for_feature = max(all_values_per_feature)
        minmax_per_feature.append([min_value_for_feature, max_value_for_feature])
    return minmax_per_feature


def normalize_data(data):
    minmax_per_feature = find_minmax(data)
    for flower in data:
        for i in range(5):
            flower[0][i] = (flower[0][i] - minmax_per_feature[i][0]) / (
                        minmax_per_feature[i][1] - minmax_per_feature[i][0])


if __name__ == '__main__':
    data_x, data_y, test_data = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    output_file_name = sys.argv[4]
    # normalize_data(data_x)
    knn = KNN(k=3)
    knn_accuracy = validate(knn, data_x, data_y)
    print(knn_accuracy, "THIS IS KNN ACC")
    knn_test_predictions = knn.predict(data_x, data_y, test_data)
    perceptron = Perceptron(learning_rate=0.02)
    perceptron_accuracy = validate(perceptron, data_x, data_y)
    print(perceptron_accuracy, "THIS IS PERCEPTRON ACC")
    perceptron_weights = perceptron.train(data_x, data_y)
    perceptron_test_predictions = perceptron.predict(perceptron_weights, test_data)
    svm = SVM(learning_rate=0.02, lambda_svm=0.08)
    svm_accuracy = validate(svm, data_x, data_y)
    print(svm_accuracy, "THIS IS SVM ACC")
    svm_weights = svm.train(data_x, data_y)
    svm_test_predictions = svm.predict(svm_weights, test_data)
    pa = PA()
    pa_accuracy = validate(pa, data_x, data_y)
    print(pa_accuracy, "THIS IS PA ACC")
    pa_weights = pa.train(data_x, data_y)
    pa_test_predictions = pa.predict(pa_weights, test_data)
    print_output_file(knn_test_predictions, perceptron_test_predictions, svm_test_predictions, pa_test_predictions,
                      output_file_name)

    '''
    TODO :
    
    - Understand why results differ so much, maybe needed seed?
    - Try feature selection ?
    - Create the report (understand how to choose hyper parameters, how many epochs?)
    - Do I need k-fold cross validation?
    
    '''
