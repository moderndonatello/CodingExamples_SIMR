import csv # for reading the csv file
import pandas as pd # uses dataframes from pandas
import numpy as np # used for arrays
from sklearn.ensemble import RandomForestClassifier # used for creating FSS score predictors --> Random Forest algorithm
from sklearn.model_selection import KFold # for implementing k-folds
import matplotlib.pyplot as plt # used for plotting the graphs
from sklearn import tree


# FUNCTIONS:

# creates the confusion matrix array by calculating the true and false # positives and negatives
def create_conf_matrix(y_preds, y_reals):
	TP = 0
	FN = 0
	FP = 0
	TN = 0
	falsePos = []
	falseNeg = []
	truePos = []

	for i in range(len(y_preds)):
		pred_val = y_preds[i]
		test_val = y_reals[i]

		if test_val == 1 and pred_val == 1:
			TP += 1
		if test_val == 1 and pred_val == 0:
			FN += 1
		if test_val == 0 and pred_val == 1:
			FP += 1
		if test_val == 0 and pred_val == 0:
			TN += 1
	return [TP, FN, FP, TN]

# calculates the accuracy of model results
def accuracy(matrix):
	TP = matrix[0]
	TN = matrix[3]
acc = (TP + TN) / (matrix[0] + matrix[1] + matrix[2] + matrix[3])
	return acc
# calculates the sensitivity of model results
def sensitivity(matrix):
	TP = matrix[0]
	FN = matrix[1]
	sens = TP / (TP + FN)
	return sens

# calculates the specificity of model results
def specificity(matrix):
	TN = matrix[3]
	FP = matrix[2]
	spec = TN / (TN + FP)
	return spec


# MAIN:
dataset = np.array(pd.read_csv('modelData.csv')) # final dataset using # 								  advisor's classifications
target_data = dataset[:, 19] # only reads column 20 - target variable
dataset = dataset[:, [10, 18, 12, 11, 14, 1, 13, 7, 17]] # 9 features

labels = ["Donor Variance", "Donor End Intensity (4000?)", "ALEX Variance", "Acceptor Variance", "Acceptor Rising(T/F)", "Donor Changes", "Donor Rising(T/F)", "ALEX Changes", "Incomplete ALEX Drop(T/F)"]
classes = ["0", "1"]

# creates 10-fold splits
kfold = KFold(n_splits = 10, shuffle = True, random_state = 110) 

photo_preds = [] # contains all predictions
photo_truths = [] # contains all truths

for train_index, test_index in kfold.split(dataset):

	x_train, x_test = dataset[train_index], dataset[test_index]
y_train, y_test = target_data[train_index], target_data[test_index]

classifier = RandomForestClassifier(max_depth = 6, min_samples_leaf = 3, min_samples_split = 10, class_weight = 'balanced', bootstrap = False, max_features = "auto", random_state = 60) # generates random forest model

	classifier.fit(x_train, y_train) # trains the modes
	y_pred = classifier.predict(x_test) # makes predictions
	for i in range(len(y_pred)):
		photo_preds.append(y_pred[i])
		photo_truths.append(y_test[i])

matrix = create_conf_matrix(photo_preds, photo_truths)
print("[TP, FN, FP, TN] = ", end = "")
print(matrix)
print("\nAccuracy: " + str(accuracy(matrix)))
print("Sensitivity: " + str(sensitivity(matrix)))
print("Specificity: " + str(specificity(matrix)))

# END OF PROGRAM
