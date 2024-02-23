import csv # for reading the csv file
import sys # for max integer num in forward stepwise feature selection
import pandas as pd # uses dataframes from pandas
import numpy as np # used for arrays
from sklearn import svm # used for creating mortality models --> Support Vector Machine algorithm
from sklearn import metrics # for accuracy calculation
from sklearn.ensemble import RandomForestClassifier # used for creating FSS score predictors --> Random Forest algorithm
from sklearn.preprocessing import LabelEncoder # for converting the strs to ints
import statistics # used for calculating standard deviation
from sklearn.metrics import hinge_loss # used to calculating the hinge loss
from sklearn.model_selection import KFold # for implementing k-folds
import matplotlib.pyplot as plt # used for plotting the graphs


# FUNCTIONS

# creates an array that converts the values from 'Mortality' to 1s, and everything else becomes a 0
def death_updater(target_data, num_ppl):
	deaths = []
	for i in range(num_ppl):
		if target_data[i] == 'Mortality':
			deaths.append(1) # 1 = dead
		else:
			deaths.append(0) # 0 = alive
	return deaths

# adds the features that needed to be summed together and stored in a new column within the mortality dataset
def add_mort_features(dataset, feat_sum_vals, num_ppl):
	cardiac_arrest = []
	for i in range(num_ppl):
		cardiac_sum = 0
		for j in range(6):
			cardiac_sum += feat_sum_vals[i][j]
		cardiac_arrest.append([cardiac_sum])
	new = np.append(dataset, cardiac_arrest, axis = 1)
	return new

def change_target_size(target_data, num_ppl):
	updated_target = np.arange(num_ppl)
	for i in range(num_ppl):
		updated_target[i] = target_data[i]
	return updated_target

# creates the confusion matrix array by calculating the true and false positives and negatives
def create_conf_matrix(y_preds, y_reals):
	TP = 0
	FN = 0
	FP = 0
	TN = 0

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

# calculates the accuracy of mortality model results
def accuracy(matrix, num_ppl):
	TP = matrix[0]
	TN = matrix[3]
	acc = (TP + TN) / num_ppl
	return acc

# calculates the sensitivity of mortality model results
def sensitivity(matrix):
	TP = matrix[0]
	FN = matrix[1]
	sens = TP / (TP + FN)
	return sens

# calculates the specificity of mortality model results
def specificity(matrix):
	TN = matrix[3]
	FP = matrix[2]
	spec = TN / (TN + FP)
	return spec

# replaces the features that have non-integers in them using LabelEncoder
def replace_vals(str_cols, dataset, num_ppl):
	for col in str_cols:
		str_list = [] # contains the strings for each column
		for i in range(num_ppl):
			str_list.append(dataset[i][col])

		label_encoder = LabelEncoder()
		new_vals = label_encoder.fit_transform(str_list) # converts strs to ints
		for i in range(num_ppl):
			dataset[i][col] = int(new_vals[i]) # replaces the strs with ints from new_vals

	for i in range(num_ppl):
		for j in range(len(dataset[0])):
			if np.isnan(dataset[i][j]):
				dataset[i][j] = -1 # if any value is empty, then replace the values with -1
			else:
				dataset[i][j] = int(dataset[i][j]) # convert all other values to integers
	return dataset

# finds the features that contain non-integers
def str_to_int_cols(dataset, num_ppl):
	str_cols = set() # columns that need to be changed from str to int
	for i in range(num_ppl):
		for j in range(len(dataset[i])):
			if type(dataset[i][j]) == str:
				str_cols.add(j)
	return str_cols

# replaces the empty FSS scores (stored as NaN) with 1s
def replace_fss_nans(fss_scores, num_ppl):
	for i in range(num_ppl):
		for j in range(6):
			if np.isnan(fss_scores[i][j]):
				fss_scores[i][j] = 1
	return fss_scores

# adds the features to each FSS domain dataset that require it --> FSS Motor, Feeding, and Respiratory 
def add_fss_features(ct_only, dataset, feat_sum_vals, num_ppl):
	if not ct_only:
		pairs = [[0, 13], [1, 14], [2, 15], [3, 16]]
		for i in range(len(pairs)):
			col_sum = []
			for j in range(num_ppl):
				pair_sum = feat_sum_vals[j][pairs[i][0]] + feat_sum_vals[j][pairs[i][1]]
				col_sum.append([pair_sum])
			new = np.append(dataset, col_sum, axis = 1)
			dataset = new

		cardiac_arrest = []
		for i in range(num_ppl):
			cardiac_sum = 0
			for j in range(17, 23):
				cardiac_sum += feat_sum_vals[i][j]
			cardiac_arrest.append([cardiac_sum])
		new = np.append(dataset, cardiac_arrest, axis = 1)
		dataset = new

		ct_array = []
		for i in range(num_ppl):
			ct_sum = 0
			for j in range(4, 13):
				ct_sum += feat_sum_vals[i][j]
			ct_array.append([ct_sum])
		new = np.append(dataset, ct_array, axis = 1)
		dataset = new

	else:
		ct_array = []
		for i in range(num_ppl):
			ct_sum = 0
			for j in range(9):
				ct_sum += feat_sum_vals[i][j]
			ct_array.append([ct_sum])
		new = np.append(dataset, ct_array, axis = 1)
		dataset = new
	return dataset

# calculates the mean squared error between the model predictions and the actual labels
def mse_val(pred, real):
	mse_sum = 0
	for i in range(len(pred)):
		mse_sum += (pred[i] - real[i]) ** 2
	return (mse_sum / len(pred))

def add_stepwise_features(dataset, feat_sum_vals, num_ppl):
	cardiac_arrest = []
	for i in range(num_ppl):
		cardiac_sum = 0
		for j in range(len(feat_sum_vals[0])):
			cardiac_sum += feat_sum_vals[i][j]
		cardiac_arrest.append([cardiac_sum])
	new = np.append(dataset, cardiac_arrest, axis = 1)
	return new

# converts the features from their integer indices to actual feature names
def feat_converter(arr, final_feats):
	feats = []
	for num in arr:
		feats.append(final_feats[num])
	return feats

# gets the feature importances of the features 
def get_importances(arr, final_feats):
	forest_importances = pd.Series(arr, index = final_feats)
	return forest_importances

# adds the standard deviation to the array containing the standard deviation sums
def add_std(std_sums, std):
	for i in range(len(std_sums)):
		std_sums[i] += std[i]
	return std_sums


# completes forward stepwise feature selection
def do_forward_stepwise():
	top_feats = 2 # checks only the top N feats determined by mutual information, or else runtime becomes too high
	num_folds = 5 # 5-fold cross validation
	features = ['age', 'admittoentnut', 'entnutyn', 'hosplos', 'puplrcticu', 'gcsicu', 'cardiacarrestyn', 'gcsmotoricu', 'cardiacarresticu', 'admittoicudc1', 'rxinotrvas', 'admittocathstart2', 'gcseyeicu', 'cathtype2', 'admittoext', 'admittocathend2', 'gcsed', 'cathtype1', 'gcsmotored', 'ctce', 'cardiacarrestprehosp', 'admittocathstart1', 'admittocathend1', 'injurymech']
	final_feats = ['age', 'admittoentnut', 'entnutyn', 'hosplos', 'puplrcticu', 'gcsicu', 'cardiacarrestyn', 'gcsmotoricu', 'cardiacarresticu', 'admittoicudc1', 'rxinotrvas', 'admittocathstart2', 'gcseyeicu', 'cathtype2', 'admittoext', 'admittocathend2', 'gcsed', 'cathtype1', 'gcsmotored', 'ctce', 'cardiacarrestprehosp', 'admittocathstart1', 'admittocathend1', 'injurymech', 'cardiacarrestsum']
	sum_cols = ['cardiacarrestyn', 'cardiacarrestprehosp', 'cardiacarrested', 'cardiacarrestor', 'cardiacarresticu', 'cardiacarrestother']
	num_features = len(final_feats)

	dataset = np.array(pd.read_csv('training.csv', usecols = features)) # only reads columns in features
	feat_sum_vals = np.array(pd.read_csv('training.csv', usecols = sum_cols)) # only reads columns in sum_cols
	target_data = np.array(pd.read_csv('training.csv', usecols = ['hospdisposition'])) # gets target data
	num_ppl = len(dataset) # number of people

	dataset = add_stepwise_features(dataset, feat_sum_vals, num_ppl) # sums the features that need to be added

	str_cols = set() # columns that need to be changed from str to int
	for i in range(num_ppl):
		for j in range(num_features):
			if type(dataset[i][j]) == str:
				str_cols.add(j)

	dataset = replace_vals(str_cols, dataset, num_ppl) # replace the strings with integers

	target_data = death_updater(target_data, num_ppl) # converts mortalities to 0's and 1's
	target_data = change_target_size(target_data, num_ppl)

	kfold = KFold(n_splits = num_folds, shuffle = True, random_state = 100)
	curr_fold = 0
	errors = np.empty(shape = (num_folds, top_feats)) # stores the hinge losses

	for train_index, test_index in kfold.split(dataset):
		x_train, x_test = dataset[train_index], dataset[test_index]
		y_train, y_test = target_data[train_index], target_data[test_index]

		added_feats = [] # stores the best features to use in ranked (descending) order
		min_train_loss = sys.maxsize # largest int value
		test_loss = 0

		for max_feats in range(top_feats):
			best_feat = 0;
			min_train_loss = sys.maxsize # largest int --> 2^31 - 1

			for feat in range(num_features):
				if feat not in added_feats:
					curr_feats = added_feats.copy()
					curr_feats.append(feat)

					new_clf = svm.SVC(kernel = 'linear', random_state = 100)
					new_clf.fit(x_train[:, curr_feats], y_train)
					confs = new_clf.decision_function(x_train[:, curr_feats])
					loss = hinge_loss(y_true = y_train, pred_decision = confs) # calculates hinge loss

					if(loss < min_train_loss): # if the loss is the new best loss, update the relevant variables
						min_train_loss = loss
						test_confs = new_clf.decision_function(x_test[:, curr_feats])
						test_loss = hinge_loss(y_true = y_test, pred_decision = test_confs)
						best_feat = feat

			added_feats.append(best_feat) # add the best feature from this iteration to the added feats
			errors[curr_fold][max_feats] = test_loss
		curr_fold += 1
		print("\nFold " + str(curr_fold) + " - top " + str(top_feats) + " features: ", end = "")
		print(feat_converter(added_feats, final_feats))

	averages = [] # stores the loss averages
	stdevs = [] # stores the standard deviations of the losses
	for i in range(top_feats):
		averages.append(np.mean(errors[:, [i]]))
		stdevs.append(np.std(errors[:, [i]]))

	x_axis = [] # x axis values for the graph
	for i in range(1, top_feats + 1):
		x_axis.append(i)

	plt.plot(x_axis, averages, color = 'blue')
	plt.errorbar(x_axis, averages, yerr = stdevs, ecolor = 'red')
	plt.title("SVM Forward Stepwise Feature Selection")
	plt.xlabel("Number of Features (Model Complexity)")
	plt.ylabel("Hinge Loss Value")
	plt.show() # shows the graph to the user


# makes mortality predictions
def mort_predictor():
	features = ['age', 'admittoentnut', 'entnutyn', 'hosplos', 'puplrcticu', 'gcsicu', 'cardiacarrestyn'] # features used from dataset
	final_features = ['age', 'admittoentnut', 'entnutyn', 'hosplos', 'puplrcticu', 'gcsicu', 'cardiacarrestyn', 'cardiacarrest_sum'] # final features used
	num_features = len(features) + 1
	sum_cols = ['cardiacarrestyn', 'cardiacarrestprehosp', 'cardiacarrested', 'cardiacarrestor', 'cardiacarresticu', 'cardiacarrestother']
	accuracies = [] # contains the accuracies for each fold

	TP = 0
	FN = 0
	FP = 0
	TN = 0

	dataset = np.array(pd.read_csv('training.csv', usecols = features)) # only reads columns in features
	feat_sum_vals = np.array(pd.read_csv('training.csv', usecols = sum_cols)) # only reads columns in sum_cols
	target_data = np.array(pd.read_csv('training.csv', usecols = ['hospdisposition'])) # gets target data
	num_ppl = len(dataset) # number of people

	dataset = add_mort_features(dataset, feat_sum_vals, num_ppl) # sums the features that need to be added

	str_cols = set() # columns that need to be changed from str to int
	for i in range(num_ppl):
		for j in range(num_features):
			if type(dataset[i][j]) == str:
				str_cols.add(j)

	dataset = replace_vals(str_cols, dataset, num_ppl) # replace the strings with integers

	target_data = death_updater(target_data, num_ppl) # converts mortalities to 0's and 1's
	target_data = change_target_size(target_data, num_ppl)

	mort_preds = []
	mort_reals = []

	kfold = KFold(n_splits = 5, shuffle = True, random_state = 25) # creates 5-fold splits

	for train_index, test_index in kfold.split(dataset):
		x_train, x_test = dataset[train_index], dataset[test_index]
		y_train, y_test = target_data[train_index], target_data[test_index]

		classifier = svm.SVC(kernel = 'linear', random_state = 100) # generates SVM model
		classifier.fit(x_train, y_train) # trains the model on the training sets
		y_pred = classifier.predict(x_test) # model predicts the data from the test dataset patients

		for i in range(len(y_pred)):
			mort_preds.append(y_pred[i])
			mort_reals.append(y_test[i])

		accuracies.append(metrics.accuracy_score(y_test, y_pred))
		matrix = create_conf_matrix(mort_preds, mort_reals)

	print("Accuracies on Each Fold: ", end = "")
	print(accuracies)

	print("Accuracy: " + str(accuracy(matrix, num_ppl)))
	print("Sensitivity: " + str(sensitivity(matrix)))
	print("Specificity: " + str(specificity(matrix)))

# calculates the feature importances of all features and plots them on a graph
def calculate_feat_imp():
	all_feats = ['studyid', 'age', 'female', 'sourceinj', 'injurytoadmit', 'injurymech', 'gcsyned', 'gcseyeed', 'gcsverbaled', 'gcsmotored', 'gcsed', 'gcsetted', 'gcsseded', 'gcspared', 'gcseyeobed', 'eddisposition', 'admittoct', 'ctskullfrac', 'ctce', 'ctmidlineshift', 'ctcompress', 'ctintraparhem', 'ctsubarchhem', 'ctintraventhem', 'ctsubhematoma', 'ctepihematoma', 'sourceicu', 'puplrcticu', 'gcsynicu', 'gcseyeicu', 'gcsverbalicu', 'gcsmotoricu', 'gcsicu', 'gcsetticu', 'gcssedicu', 'gcsparicu', 'gcseyeobicu', 'admittoicudc1', 'admittoicuadmit2', 'admittoicudc2', 'admittoicuadmit3', 'admittoicudc3', 'ventyn', 'admittoint', 'admittoext', 'icpyn1', 'icptype1', 'icptype2', 'icptype3', 'admittoicpstart1', 'admittoicpend1', 'admittoicpstart2', 'admittoicpend2', 'admittoicpstart3', 'admittoicpend3', 'cathtype1', 'cathtype2', 'cathtype3', 'cathtype4', 'admittocathstart1', 'admittocathstart2', 'admittocathstart3', 'admittocathstart4', 'admittocathend1', 'admittocathend2', 'admittocathend3'	, 'admittocathend4', 'newtrachyn', 'admittotrach', 'newgastyn', 'admittogast', 'decomcranyn', 'admittocrani', 'lmbrdrainyn', 'admittolmbdrain', 'epihemyn', 'admittoedhevac', 'subhemyn', 'admittosdhevac', 'rxhypsal', 'rxmann', 'rxbarb', 'rxinotrvas', 'tpnyn', 'admittotpn', 'entnutyn', 'admittoentnut', 'hosplos', 'hospdisposition', 'cardiacarrestyn', 'cardiacarrestprehosp', 'cardiacarrested', 'cardiacarrestor', 'cardiacarresticu', 'cardiacarrestother']
	classes = ['fssmental', 'fsssensory', 'fsscommun', 'fssmotor', 'fssfeeding', 'fssresp']
	sum_cols = ['gcsetted', 'gcsseded', 'gcspared', 'gcseyeobed', 'ctskullfrac', 'ctce', 'ctmidlineshift', 'ctcompress', 'ctintraparhem', 'ctsubarchhem', 'ctintraventhem', 'ctsubhematoma', 'ctepihematoma', 'gcsetticu', 'gcssedicu', 'gcsparicu', 'gcseyeobicu', 'cardiacarrestyn', 'cardiacarrestprehosp', 'cardiacarrested', 'cardiacarrestor', 'cardiacarresticu', 'cardiacarrestother']
	titles = ['FSS Mental', 'FSS Sensory', 'FSS Communication', 'FSS Motor', 'FSS Feeding', 'FSS Respiratory']
	final_feats = ['studyid', 'age', 'female', 'sourceinj', 'injurytoadmit', 'injurymech', 'gcsyned', 'gcseyeed', 'gcsverbaled', 'gcsmotored', 'gcsed', 'gcsetted', 'gcsseded', 'gcspared', 'gcseyeobed', 'eddisposition', 'admittoct', 'ctskullfrac', 'ctce', 'ctmidlineshift', 'ctcompress', 'ctintraparhem', 'ctsubarchhem', 'ctintraventhem', 'ctsubhematoma', 'ctepihematoma', 'sourceicu', 'puplrcticu', 'gcsynicu', 'gcseyeicu', 'gcsverbalicu', 'gcsmotoricu', 'gcsicu', 'gcsetticu', 'gcssedicu', 'gcsparicu', 'gcseyeobicu', 'admittoicudc1', 'admittoicuadmit2', 'admittoicudc2', 'admittoicuadmit3', 'admittoicudc3', 'ventyn', 'admittoint', 'admittoext', 'icpyn1', 'icptype1', 'icptype2', 'icptype3', 'admittoicpstart1', 'admittoicpend1', 'admittoicpstart2', 'admittoicpend2', 'admittoicpstart3', 'admittoicpend3', 'cathtype1', 'cathtype2', 'cathtype3', 'cathtype4', 'admittocathstart1', 'admittocathstart2', 'admittocathstart3', 'admittocathstart4', 'admittocathend1', 'admittocathend2', 'admittocathend3'	, 'admittocathend4', 'newtrachyn', 'admittotrach', 'newgastyn', 'admittogast', 'decomcranyn', 'admittocrani', 'lmbrdrainyn', 'admittolmbdrain', 'epihemyn', 'admittoedhevac', 'subhemyn', 'admittosdhevac', 'rxhypsal', 'rxmann', 'rxbarb', 'rxinotrvas', 'tpnyn', 'admittotpn', 'entnutyn', 'admittoentnut', 'hosplos', 'hospdisposition', 'cardiacarrestyn', 'cardiacarrestprehosp', 'cardiacarrested', 'cardiacarrestor', 'cardiacarresticu', 'cardiacarrestother', 'gcsett', 'gcssed', 'gcspar', 'gcseyeob', 'cardiacarrestsum', 'ctsum']

	mortality_arr = np.array(pd.read_csv('training.csv', usecols = [88]))
	num_ppl = len(mortality_arr) # number of people
	skips = []
	for i in range(num_ppl):
		if mortality_arr[i][0] == 'Mortality' or mortality_arr[i][0] == 'Discharge TO or WITH Hospice':
			skips.append(i + 1)

	full = np.array(pd.read_csv('training.csv', usecols = all_feats, skiprows = skips))
	sum_vals = np.array(pd.read_csv('training.csv', usecols = sum_cols, skiprows = skips))
	fss_scores = np.array(pd.read_csv('training.csv', usecols = classes, skiprows = skips))
	num_ppl = len(full)

	full = add_fss_features(False, full, sum_vals, num_ppl)

	str_cols = str_to_int_cols(full, num_ppl)
	full = replace_vals(str_cols, full, num_ppl)

	fss_scores = replace_fss_nans(fss_scores, num_ppl)

	predictions = []
	real_scores = []


	total_predictions = []
	total_scores = []

	num_folds = 5
	kfold = KFold(n_splits = num_folds, shuffle = True, random_state = 70) # creates 5-fold splits

	for i in range(6):
		curr_fold = 0
		target_data = fss_scores[:, i]
		dataset = full

		for train_index, test_index in kfold.split(dataset):

			x_train, x_test = dataset[train_index], dataset[test_index]
			y_train, y_test = target_data[train_index], target_data[test_index]
			
			classifier = RandomForestClassifier(max_depth = 6, min_samples_leaf = 1, min_samples_split = 3, class_weight = 'balanced', bootstrap = False, max_features = "auto", random_state = 150) # generates random forest model
			classifier.fit(x_train, y_train) # trains the model using the training sets
			y_pred = classifier.predict(x_test) # predicts the data for the test dataset

			importances = classifier.feature_importances_
			forest_importances = get_importances(importances, final_feats) # gets feature importances of eqch feature
			std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis = 0) # calculates the standard deviation

			if curr_fold == 0:
				total_importances = forest_importances
				std_sums = std
			else:
				total_importances = total_importances.add(forest_importances, fill_value = 0)
				std_sums = add_std(std_sums, std)

			curr_fold += 1

		total_importances.div(num_folds) # averages out the feature importances across all the folds
		std_sums = std_sums / num_folds # averages the standard deviatoins aross all the folds

		fig, ax = plt.subplots()
		total_importances.plot.bar(yerr = std_sums, ax = ax) # plots the feature importance graphs; one for each of the FSS domains
		ax.set_title("Feature Importances for " + titles[i] + " Domain in the Random Forest")
		ax.set_ylabel("Feature Importance")
		fig.tight_layout()
		plt.show() # displays the graph


# makes FSS score predictions across all FSS domains
def fss_predictor():
	fss_mental_feats = ['age', 'hosplos', 'admittoicudc1', 'admittoicpend1', 'admittocathend2', 'admittocathend1', 'admittoext']
	fss_sensory_feats = ['age', 'hosplos', 'admittoicudc1', 'admittocathend3', 'admittocathend2', 'admittoicpend1', 'admittoext']
	fss_commun_feats = ['age', 'hosplos', 'admittoicudc1', 'admittoicpend1', 'admittocathend2', 'admittoext', 'admittocathend1', 'admittocathend3', 'admittocathstart2', 'admittoint', 'admittogast', 'admittoicpstart1', 'icptype1', 'gcsed', 'admittocathstart3']
	fss_motor_feats = ['age', 'female', 'injurymech', 'gcsyned', 'gcsed', 'admittoct', 'puplrcticu', 'gcsynicu', 'gcsicu', 'admittoicudc1', 'ventyn', 'icpyn1', 'subhemyn', 'entnutyn', 'hosplos']
	fss_feeding_feats = ['age', 'female', 'injurymech', 'gcsyned', 'gcsed', 'admittoct', 'puplrcticu', 'gcsynicu', 'gcsicu', 'admittoicudc1', 'ventyn', 'icpyn1', 'subhemyn', 'entnutyn', 'hosplos']
	fss_resp_feats = ['age', 'hosplos', 'admittoicudc1', 'admittoext', 'admittocathend2', 'admittogast', 'admittoicpend1', 'admittocathend1', 'admittoint', 'admittotrach', 'admittocathend3', 'newtrachyn', 'admittocathstart3', 'gcsed', 'admittoicpend2', 'gcsicu']

	classes = ['fssmental', 'fsssensory', 'fsscommun', 'fssmotor', 'fssfeeding', 'fssresp']
	titles = ['FSS Mental Predictions:', 'FSS Sensory Predictions:', 'FSS Communication Predictions:', 'FSS Motor Predictions:', 'FSS Feeding Predictions:', 'FSS Respiratory Predictions:']
	motor_and_feeding_sumcols = ['gcsetted', 'gcsseded', 'gcspared', 'gcseyeobed', 'ctskullfrac', 'ctce', 'ctmidlineshift', 'ctcompress', 'ctintraparhem', 'ctsubarchhem', 'ctintraventhem', 'ctsubhematoma', 'ctepihematoma', 'gcsetticu', 'gcssedicu', 'gcsparicu', 'gcseyeobicu', 'cardiacarrestyn', 'cardiacarrestprehosp', 'cardiacarrested', 'cardiacarrestor', 'cardiacarresticu', 'cardiacarrestother']
	fss_resp_sumcols = ['ctskullfrac', 'ctce', 'ctmidlineshift', 'ctcompress', 'ctintraparhem', 'ctsubarchhem', 'ctintraventhem', 'ctsubhematoma', 'ctepihematoma']

	mortality_arr = np.array(pd.read_csv('training.csv', usecols = [88]))
	num_ppl = len(mortality_arr) # number of people
	skips = [] # contains mortalities of patients, and skips these rows because dead people don't have FSS scores
	for i in range(num_ppl):
		if mortality_arr[i][0] == 'Mortality' or mortality_arr[i][0] == 'Discharge TO or WITH Hospice':
			skips.append(i + 1)

	# datasets for each FSS domain
	fss_mental = np.array(pd.read_csv('training.csv', usecols = fss_mental_feats, skiprows = skips))
	fss_sensory = np.array(pd.read_csv('training.csv', usecols = fss_sensory_feats, skiprows = skips))
	fss_commun = np.array(pd.read_csv('training.csv', usecols = fss_commun_feats, skiprows = skips))
	fss_motor = np.array(pd.read_csv('training.csv', usecols = fss_motor_feats, skiprows = skips))
	fss_feeding = np.array(pd.read_csv('training.csv', usecols = fss_feeding_feats, skiprows = skips))
	fss_resp = np.array(pd.read_csv('training.csv', usecols = fss_resp_feats, skiprows = skips))
	num_ppl = len(fss_mental)

	resp_sum_cols = np.array(pd.read_csv('training.csv', usecols = fss_resp_sumcols, skiprows = skips))
	feeding_motor_sum_vals = np.array(pd.read_csv('training.csv', usecols = motor_and_feeding_sumcols, skiprows = skips))
	fss_scores = np.array(pd.read_csv('training.csv', usecols = classes, skiprows = skips))

	fss_motor = add_fss_features(False, fss_motor, feeding_motor_sum_vals, num_ppl)
	fss_feeding = add_fss_features(False, fss_feeding, feeding_motor_sum_vals, num_ppl)
	fss_resp = add_fss_features(True, fss_resp, resp_sum_cols, num_ppl)

	fss_datasets = [fss_mental, fss_sensory, fss_commun, fss_motor, fss_feeding, fss_resp]

	for i in range(len(fss_datasets)):
		str_cols = str_to_int_cols(fss_datasets[i], num_ppl) # finds the columns that have non-integers in them
		fss_datasets[i] = replace_vals(str_cols, fss_datasets[i], num_ppl) # replaces non-integer values with integers


	fss_scores = replace_fss_nans(fss_scores, num_ppl) # replaces the empty FSS scores with 1s

	total_predictions = []
	total_scores = []

	kfold = KFold(n_splits = 5, shuffle = True) # creates shuffled 5-fold splits

	for i in range(6):
		print(titles[i])

		fss_pred = []
		fss_real = []

		target_data = fss_scores[:, i]
		dataset = fss_datasets[i]

		for train_index, test_index in kfold.split(dataset):

			x_train, x_test = dataset[train_index], dataset[test_index]
			y_train, y_test = target_data[train_index], target_data[test_index]
			
			classifier = RandomForestClassifier(max_depth = 6, min_samples_leaf = 1, min_samples_split = 3, class_weight = 'balanced', bootstrap = False, max_features = "auto", random_state = 150) # generates random forest model
			classifier.fit(x_train, y_train) # trains the model using the training sets
			y_pred = classifier.predict(x_test) # predicts the data for the test dataset

			for j in range(len(y_pred)):
				fss_pred.append(y_pred[j])
				fss_real.append(y_test[j])

		total_predictions.append(fss_pred)
		total_scores.append(fss_real)

		y_pred2 = classifier.predict(x_train)
		print("MSE training: " + str(mse_val(y_pred2, y_train)))
		print("MSE testing: " + str(mse_val(fss_pred, fss_real)))
		print("Standard Deviation: " + str(statistics.stdev(y_pred)) + "\n")
		# print(y_pred)

	fss_predictions = []
	fss_reals = []
	for i in range(len(total_predictions[0])):
		pred_sum = 0
		real_sum = 0
		for j in range(len(total_predictions)):
			pred_sum += total_predictions[j][i]
			real_sum += total_scores[j][i]
		fss_predictions.append(pred_sum)
		fss_reals.append(real_sum)

	print("MSE testing total: " + str(mse_val(fss_predictions, fss_reals)))
	print("Final Standard Deviation: " + str(statistics.stdev(fss_predictions)) + "\n")


# MAIN:

# print("\nFORWARD STEPWISE FEATURE SELECTION . . .")
# do_forward_stepwise() # this method takes some time to run because it is iterating through all the features and selecting the best one

print("\n\nMORTALITY PREDICTIONS . . .\n")
mort_predictor() # predicts whether a patient is a mortality or a survivor, and prints out the accuracy (across all folds as well), specificty, and sensitivity

print("\n\nFEATURE IMPORTANCE GRAPHS . . .\n")
calculate_feat_imp() # calculates the feature importance for all features, and graphs it

print("\nFSS SCORE PREDICTIONS . . .\n")
fss_predictor() # predicts the FSS scores of each TBI patient, and prints out the MSE across each fold, as well as the final MSE

print("PROGRAM COMPLETED!\n")

# END OF PROGRAM
