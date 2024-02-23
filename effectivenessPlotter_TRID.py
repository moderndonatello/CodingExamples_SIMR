import csv # for reading the csv file
import pandas as pd # uses dataframes from pandas
import numpy as np # used for arrays
import matplotlib.pyplot as plt # used for plotting the graphs
import scipy.stats as stats # used for Mann-Whitney U-Test


# FUNCTIONS:

def createPrefixSum(arr): # creates the cumulative data so that it shows at time t, how many traces have dropped already
	prefix = [0.0]
	arrSum = 0
	for i in range(len(arr)):
		arrSum += arr[i]
		prefix.append(arrSum)
	return prefix

def reformatData(arr): # reformats the data into the best form 
	newArr = []
	for i in range(len(arr)):
		newArr.append(arr[i][0])
	return newArr

def createDropLocArray(dropLocs): # total drops across time (s)
	allDrops = np.zeros(dropLocs[len(dropLocs) - 1])

	for i in range(len(dropLocs)):
		loc = dropLocs[i] - 1
		allDrops[loc] += 1
	return allDrops


# MAIN:

dataBoth = np.array(pd.read_csv('dropLocsBoth.csv')) # when model predictions and truths both are 1s
dataTruths = np.array(pd.read_csv('dropLocsTruths.csv')) # when only truths are 1s
dataPreds = np.array(pd.read_csv('dropLocsPreds.csv')) # when only predictions are 1s

dropLocsBoth = reformatData(dataBoth)
dropLocsTruths = reformatData(dataTruths)
dropLocsPreds = reformatData(dataPreds)

dropLocsBoth.sort()
dropLocsTruths.sort()
dropLocsPreds.sort()

allDropsBoth = createDropLocArray(dropLocsBoth)
allDropsTruths = createDropLocArray(dropLocsTruths)
allDropsPreds = createDropLocArray(dropLocsPreds)

dropSumsBoth = createPrefixSum(allDropsBoth)
dropSumsTruths = createPrefixSum(allDropsTruths)
dropSumsPreds = createPrefixSum(allDropsPreds)

numDropsBoth = max(dropSumsBoth)
numDropsTruths = max(dropSumsTruths)
numDropsPreds = max(dropSumsPreds)

timesBoth = []
timesTruths = []
timesPreds = []

for i in range(len(dropSumsBoth)):
	dropSumsBoth[i] = dropSumsBoth[i] / numDropsBoth
	timesBoth.append(i)

for i in range(len(dropSumsTruths)):
	dropSumsTruths[i] = dropSumsTruths[i] / numDropsTruths
	timesTruths.append(i)

for i in range(len(dropSumsPreds)):
	dropSumsPreds[i] = dropSumsPreds[i] / numDropsPreds
	timesPreds.append(i)

#performs Mann-Whitney U-Test
p = stats.mannwhitneyu(dropSumsTruths, dropSumsPreds, alternative = "two-sided") # p-value = 0.999

plt.plot(timesTruths, dropSumsTruths, alpha = 1, linewidth = 3)
plt.plot(timesPreds, dropSumsPreds, alpha = 0.7, linewidth = 3)

plt.legend(["Truths", "Predictions"], loc = "lower right")
plt.xlabel("Time (s)")
plt.ylabel("Percent of Traces that Dissociated")
plt.title("Ataluren's Suppression of Release Factor Activity Over Time")
plt.show(

# END OF PROGRAM
