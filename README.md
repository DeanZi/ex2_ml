# ex2_ml
### Machine Learning Models Implementation
This code provides an implementation of four machine learning models: K-Nearest Neighbors (KNN),
Support Vector Machine (SVM), Perceptron, and Passive Aggressive (PA).

### Requirements
* Python 3.x
* NumPy
* scikit-learn

### Models

The implemented models are:

* K-Nearest Neighbors (KNN): a non-parametric method used for classification and regression. The k-NN algorithm assumes that similar things exist in close proximity, so it predicts the label of a new instance based on the labels of the k-nearest neighbors in the training data.

* Support Vector Machine (SVM): a supervised learning model used for classification and regression analysis. The SVM algorithm tries to find the best hyperplane that separates the different classes of data points in the training set.

* Perceptron: a linear classifier used for binary classification problems. The perceptron algorithm tries to find the weights that minimize the classification error on the training data.

* Passive Aggressive (PA): a family of algorithms used for online learning. PA algorithms update the model parameters based on the observed training data in an incremental fashion.

### Dataset
The code uses the Iris flowers dataset, a commonly used dataset in machine learning.

### Report

Implementation details:

*	I have created a model class with train and predict methods, made each model inherit and override the methods.

*	Implemented k-fold cross validation (k=5), left around 10-20 percent of the train dataset to validation. Was able to check out the accuracy of different models with different hyperparameters.

*	Tried min max normalization – didn’t improve even degraded the accuracy.

•	Tried Z-score normalization – did not seem to affect much.

•	Implemented Anova f-score feature selection and found out that feature 5 is not contributing much for the classification process, hence got rid of that feature in train and test datasets.

•	To choose my hyperparameters I ran a search on some different values.

### Results

* K for KNN:

  KNN ACC for k=7:  95.41666666666667 %![image](https://user-images.githubusercontent.com/45519333/227004749-440cae96-f354-4cb9-9503-a3d1647eb644.png)


* Learning rate:

  PERCEPTRON (l_r = 0.01) ACC:  90.0 %![image](https://user-images.githubusercontent.com/45519333/227004708-b2fc7874-d9fb-405f-a10b-306e166306f7.png)
  SVM (l_r = 0.005) ACC:  87.5 %![image](https://user-images.githubusercontent.com/45519333/227004647-31cc8113-1c4b-4fad-adb6-7b1d3de81024.png)

* Lambda:

  SVM (lambda = 1) ACC:  88.75 % ![image](https://user-images.githubusercontent.com/45519333/227004867-38e2dc5f-0a41-4c90-9011-4458434141ce.png)

* Epochs:

  PERCEPTRON (epochs = 3000) ACC:  88.33333333333333 % ![image](https://user-images.githubusercontent.com/45519333/227004942-a4e185a7-e90c-44b4-91b8-4765981c52a6.png)
  SVM (epochs = 3000) ACC:  90.0 % ![image](https://user-images.githubusercontent.com/45519333/227004979-37505f0d-234b-4f7d-905f-7f1a322f0b28.png)
  PA (epochs = 2000) ACC:  87.08333333333333 % ![image](https://user-images.githubusercontent.com/45519333/227005021-6a144104-50c1-4f45-80e5-377590303cd7.png)
