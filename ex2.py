import math
import sys
import numpy as np


class Model:
    def __init__(self, k=None, learning_rate=1, lambda_svm=1, epochs=1000):
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_svm = lambda_svm
        self.epochs = epochs

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
                sum_for_distance = 0
                for feature_data_x, feature_test in zip(example[0].tolist(), flower[0].tolist()):
                    sum_for_distance += (feature_test-feature_data_x)**2
                distances[i] = math.sqrt(sum_for_distance)
            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            labels_of_neighbours = []
            for i in range(self.k):
                labels_of_neighbours.append(data_y[list(sorted_distances.keys())[i]])
            predictions.append(max(set(labels_of_neighbours), key=labels_of_neighbours.count))
        return predictions


class Perceptron(Model):

    def train(self, data_x, data_y):
        weights = np.zeros([3, 5])
        best_weights = []
        initial_learning_rate = self.learning_rate
        min_loss = sys.maxsize
        for epoch in range(self.epochs):
            random_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(random_state)
            np.random.shuffle(data_y)
            self.learning_rate = (1 / (1 + epoch)) * initial_learning_rate
            loss = 0
            for example, label in zip(data_x, data_y):
                y_hat = np.argmax(np.dot(weights, example.transpose()))
                if y_hat != label:
                    loss += 1
                    weights[label] = weights[label] + self.learning_rate * example
                    weights[y_hat] = weights[y_hat] - self.learning_rate * example
            if loss < min_loss:
                min_loss = loss
                best_weights = weights
        return best_weights


class SVM(Model):

    def train(self, data_x, data_y):
        weights = np.zeros([3, 5])
        classifications = [0, 1, 2]
        best_weights = []
        initial_learning_rate = self.learning_rate
        min_loss = sys.maxsize
        for epoch in range(self.epochs):
            random_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(random_state)
            np.random.shuffle(data_y)
            self.learning_rate = (1 / (1 + epoch)) * initial_learning_rate
            loss = 0
            for example, label in zip(data_x, data_y):
                y_hat = np.argmax(np.dot(weights, example.transpose()))
                hinge_loss = max(0, 1 - np.dot(weights[label], example.transpose()) + np.dot(weights[y_hat],
                                                                                             example.transpose()))
                if hinge_loss > 0:
                    loss += 1
                    weights[label] = (1 - self.learning_rate * self.lambda_svm) * weights[
                        label] + self.learning_rate * example
                    weights[y_hat] = (1 - self.learning_rate * self.lambda_svm) * weights[
                        y_hat] - self.learning_rate * example
                    for classification in classifications:
                        if classification != label and classification != y_hat:
                            weights[classification] = (1 - self.learning_rate * self.lambda_svm) * weights[
                                classification]
            if loss < min_loss:
                min_loss = loss
                best_weights = weights

        return best_weights


class PA(Model):

    def train(self, data_x, data_y):
        weights = np.zeros([3, 5])
        best_weights = []
        min_loss = sys.maxsize
        for epoch in range(self.epochs):
            random_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(random_state)
            np.random.shuffle(data_y)
            loss = 0
            for example, label in zip(data_x, data_y):
                y_hat = np.argmax(np.dot(weights, example.transpose()))
                hinge_loss = max(0, 1 - np.dot(weights[label], example.transpose()) + np.dot(weights[y_hat],
                                                                                             example.transpose()))
                tau = hinge_loss / (2 * (np.linalg.norm(example) ** 2))
                if hinge_loss > 0:
                    loss += 1
                    weights[label] = weights[label] + tau * example
                    weights[y_hat] = weights[y_hat] - tau * example
            if loss < min_loss:
                min_loss = loss
                best_weights = weights

        return best_weights


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
        random_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(random_state)
        np.random.shuffle(data_y)
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


def calculate_f_score_per_feature(data_x, data_y):
    features_f_score = []
    for feature in range(5):
        label_0 = []
        label_1 = []
        label_2 = []
        for example, label in zip(data_x, data_y):
            if label == 0:
                label_0.append(example[0][feature])
            if label == 1:
                label_1.append(example[0][feature])
            if label == 2:
                label_2.append(example[0][feature])
        avg_of_all = (sum(label_0) + sum(label_1) + sum(label_2)) / len(data_x)
        avg_label_0 = sum(label_0) / len(label_0)
        avg_label_1 = sum(label_1) / len(label_1)
        avg_label_2 = sum(label_2) / len(label_2)
        numerator = (len(label_0) * pow((avg_label_0 - avg_of_all),2)) + (len(label_1) * pow((avg_label_1 - avg_of_all),2)) + (len(label_2) * pow((avg_label_2 - avg_of_all),2))
        sum_of_square_0 = sum([pow((x-avg_label_0),2) for x in label_0])
        sum_of_square_1 = sum([pow((x-avg_label_1),2) for x in label_1])
        sum_of_square_2 = sum([pow((x-avg_label_2),2) for x in label_2])
        len_of_groups_minus_one = len(data_x) - 3
        denominator = (sum_of_square_0 + sum_of_square_1 + sum_of_square_2) / len_of_groups_minus_one
        features_f_score.append(numerator/denominator)
    return features_f_score


def clean_features_from_data(data_x):
    new_data_x = []
    for flower in data_x:
        flower = np.delete(flower, 4, 1)
        new_data_x.append(flower)
    return new_data_x


def remove_outliers(new_data_x, data_y):
    outliers = []
    for flower_x in new_data_x:
        for index in range(len(new_data_x[0])):
            mean = np.average(new_data_x[:, index])
            std = np.std(new_data_x[:, index])
            z_score = (flower_x[0][index] - mean) / std
            if abs(z_score) > 3:
                if index not in outliers:
                    outliers.append(index)
    new_data_x = np.delete(new_data_x, outliers, 0)
    new_data_y = np.delete(data_y, outliers, 0)
    return new_data_x, new_data_y


def normalize_data(data_x, data_y, test_data):

    data_x = np.array(data_x)
    test_data = np.array(test_data)
    for feature_index in range(len(data_x[0])):
        mean = np.average(data_x[:, feature_index])
        std = np.std(data_x[:, feature_index])
        data_x[:, feature_index] = (data_x[:, feature_index] - mean) / std
        test_data[:, feature_index] = (test_data[:, feature_index] - mean) / std

    new_data_x, new_data_y = remove_outliers(data_x, data_y)

    return new_data_x, new_data_y, test_data


def add_bias_to_data(data_x, test_data):
    new_data_x = []
    new_test_data = []
    for flower in data_x:
        flower = np.append(flower, [[1]], 1)
        new_data_x.append(flower)
    for flower in test_data:
        flower = np.append(flower, [[1]], 1)
        new_test_data.append(flower)
    return new_data_x, new_test_data


if __name__ == '__main__':
    data_x, data_y, test_data = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    output_file_name = sys.argv[4]
    # features_f_score = calculate_f_score_per_feature(data_x, data_y)
    # data_x, data_y, test_data = normalize_data(data_x, data_y, test_data)
    # data_x, test_data = add_bias_to_data(data_x, test_data)
    # data_x = clean_features_from_data(data_x)
    # test_data = clean_features_from_data(test_data)
    # print(features_f_score)
    perceptron = Perceptron(learning_rate=0.0001, epochs=3000)
    perceptron_accuracy = validate(perceptron, data_x, data_y)
    print(perceptron_accuracy, f"THIS IS PERCEPTRON ACC")
    # perceptron_weights = perceptron.train(data_x, data_y)
    # perceptron_test_predictions = perceptron.predict(perceptron_weights, test_data)
    svm = SVM(learning_rate=0.0001, lambda_svm=1, epochs=3000)
    svm_accuracy = validate(svm, data_x, data_y)
    print(svm_accuracy, f"THIS IS SVM ACC")
    # svm_weights = svm.train(data_x, data_y)
    # svm_test_predictions = svm.predict(svm_weights, test_data)
    pa = PA(epochs=3000)
    pa_accuracy = validate(pa, data_x, data_y)
    print(pa_accuracy, f"THIS IS PA ACC")
    # pa_weights = pa.train(data_x, data_y)
    # pa_test_predictions = pa.predict(pa_weights, test_data)
    knn = KNN(k=7)
    print(f"KNN: ", validate(knn, data_x, data_y), '%')
    # knn_test_predictions = knn.predict(data_x, data_y, test_data)
    # print_output_file(knn_test_predictions, perceptron_test_predictions, svm_test_predictions, pa_test_predictions,
    #                   output_file_name)


