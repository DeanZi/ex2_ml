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

•	I have created a model class with train and predict methods, made each model inherit and override the methods.

•	Implemented k-fold cross validation (k=5) as advised in tirgul (to leave around 10-20 percent of the train dataset to validation) to be able to check out the accuracy of different models with different hyperparameters.

•	Also tried min max normalization – didn’t improve even degraded the accuracy.

•	Tried Z-score normalization – did not seem to affect much.

•	Implemented Anova f-score feature selection (see code below) and found out that feature 5 is not contributing much for the classification process, hence got rid of that feature in train and test datasets.

•	To choose my hyperparameters I ran a search on some different values, see results below.

### Results

Choosing K for KNN:

KNN ACC for k=1:  91.25 %
KNN ACC for k=2:  90.41666666666666 %
KNN ACC for k=3:  92.91666666666667 %
KNN ACC for k=4:  89.58333333333333 %
KNN ACC for k=5:  91.66666666666667 %
KNN ACC for k=6:  90.0 %
KNN ACC for k=7:  95.41666666666667 %
KNN ACC for k=8:  90.83333333333334 %
KNN ACC for k=9:  91.66666666666667 %
KNN ACC for k=10:  90.83333333333333 %
KNN ACC for k=11:  92.5 %
KNN ACC for k=12:  89.16666666666667 %
KNN ACC for k=13:  89.16666666666667 %
KNN ACC for k=14:  91.25 %
KNN ACC for k=15:  93.33333333333334 %
KNN ACC for k=16:  90.41666666666667 %
KNN ACC for k=17:  92.08333333333334 %
KNN ACC for k=18:  89.58333333333333 %
KNN ACC for k=19:  93.33333333333333 %
KNN ACC for k=20:  92.08333333333333 %
KNN ACC for k=21:  93.75000000000003 %
KNN ACC for k=22:  92.08333333333334 %
KNN ACC for k=23:  92.08333333333333 %
KNN ACC for k=24:  91.25 %
KNN ACC for k=25:  92.08333333333333 %
KNN ACC for k=26:  89.16666666666667 %
KNN ACC for k=27:  91.25 %
KNN ACC for k=28:  91.25000000000003 %
KNN ACC for k=29:  90.0 %

Choosing learning rate:

PERCEPTRON (l_r = 0.001) ACC:  82.91666666666667 %
PERCEPTRON (l_r = 0.005) ACC:  84.58333333333333 %
PERCEPTRON (l_r = 0.01) ACC:  90.0 %
PERCEPTRON (l_r = 0.02) ACC:  85.83333333333333 %
PERCEPTRON (l_r = 0.05) ACC:  89.16666666666667 %
PERCEPTRON (l_r = 0.1) ACC:  84.16666666666667 %
PERCEPTRON (l_r = 0.5) ACC:  85.41666666666667 %
PERCEPTRON (l_r = 1) ACC:  87.5 %
SVM (l_r = 0.001) ACC:  85.41666666666667 %
SVM (l_r = 0.005) ACC:  87.5 %
SVM (l_r = 0.01) ACC:  85.41666666666667 %
SVM (l_r = 0.02) ACC:  84.16666666666667 %
SVM (l_r = 0.05) ACC:  86.66666666666666 %
SVM (l_r = 0.1) ACC:  82.08333333333333 %
SVM (l_r = 0.5) ACC:  82.91666666666667 %
SVM (l_r = 1) ACC:  82.91666666666667 %


Choosing lambda for SVM:

SVM (lambda = 0.001) ACC:  87.91666666666667 % 
SVM (lambda = 0.005) ACC:  82.08333333333334 % 
SVM (lambda = 0.01) ACC:  85.83333333333334 % 
SVM (lambda = 0.02) ACC:  87.08333333333333 % 
SVM (lambda = 0.05) ACC:  85.83333333333333 % 
SVM (lambda = 0.08) ACC:  87.08333333333334 % 
SVM (lambda = 0.1) ACC:  82.08333333333334 % 
SVM (lambda = 0.2) ACC:  86.25 % 
SVM (lambda = 0.5) ACC:  86.25 % 
SVM (lambda = 1) ACC:  88.75 % 


EPOCHS check:

PERCEPTRON (epochs = 500) ACC:  81.25 % 
PERCEPTRON (epochs = 1000) ACC:  84.58333333333333 % 
PERCEPTRON (epochs = 2000) ACC:  83.33333333333333 % 
PERCEPTRON (epochs = 3000) ACC:  88.33333333333333 % 
PERCEPTRON (epochs = 4000) ACC:  83.33333333333334 % 
PERCEPTRON (epochs = 5000) ACC:  82.91666666666667 % 
PERCEPTRON (epochs = 6000) ACC:  82.08333333333333 % 
PERCEPTRON (epochs = 7000) ACC:  82.91666666666667 % 
SVM (epochs = 500) ACC:  85.83333333333334 % 
SVM (epochs = 1000) ACC:  84.16666666666667 % 
SVM (epochs = 2000) ACC:  80.41666666666667 % 
SVM (epochs = 3000) ACC:  90.0 % 
SVM (epochs = 4000) ACC:  86.25 % 
SVM (epochs = 5000) ACC:  89.58333333333333 % 
SVM (epochs = 6000) ACC:  87.5 % 
SVM (epochs = 7000) ACC:  83.33333333333333 % 
PA (epochs = 500) ACC:  85.83333333333333 % 
PA (epochs = 1000) ACC:  85.41666666666667 % 
PA (epochs = 2000) ACC:  87.08333333333333 % 
PA (epochs = 3000) ACC:  86.66666666666667 % 
PA (epochs = 4000) ACC:  84.75 % 
PA (epochs = 5000) ACC:  86.25 % 
PA (epochs = 6000) ACC:  82.5 % 
PA (epochs = 7000) ACC:  84.58333333333333 %
![image](https://user-images.githubusercontent.com/45519333/227003352-acf30038-bb91-4f30-b192-95d48e8736da.png)
