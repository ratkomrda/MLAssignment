import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, Normalizer
import math

def printDivision():
    print("====================================================")
    return


def getPredictedTarget(results):
    # How many elements are there in results
    index = len(results)
    # Access the last element added to results and the y predicition is the final value stored in each classifers
    # results so -1 can be used to index it
    ypred = results[index - 1][-1]
    return ypred

def benchmark(name, clf, Xtrain, ytrain, Xtest, ytest):
    # Get the current time
    t0 = time()
    # Train the classifier
    clf.fit(Xtrain, ytrain)
    # The difference between the time now and t0 is the time taken to train the classifier
    train_duration = time() - t0
    printDivision()
    print("Train Duration: %0.3fs" % train_duration)
    # Get the current time
    t0 = time()
    # Test the classifier by using the model on the training data and returning the results
    ypred = clf.predict(Xtest)
    # The difference between the time now and t0 is the time taken to test the classifier
    test_duration = time() - t0
    print("Test Duration:  %0.3fs" % test_duration)
    # Calculate the accuracy
    accuracy = metrics.accuracy_score(ytest, ypred)
    return name, accuracy, train_duration, test_duration, ypred

def printPlotMetrics(ytest, Xtest, ypred, model):
    # The confusion matrix for the logistic regression classifier
    cnf_matrix = metrics.confusion_matrix(ytest, ypred)
    printDivision()
    print("Confusion Matrix")
    printDivision()
    print(cnf_matrix)
    printDivision()
    print("Classification Results")
    printDivision()
    print(metrics.classification_report(ytest, ypred, target_names=['Agressive','Even']))
    # The mean squared error
    printDivision()
    print("Mean squared error: %.2f"
          % mean_squared_error(ytest, ypred))
    printDivision()
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ytest, ypred))
    printDivision()
    print('Actual Values:', ytest[0:20])
    print('Predicted Values', ypred[0:20])
    printDivision()

    y_prediction_probability = model.predict_proba(Xtest)[:,1]
    print("Prediction Probability", y_prediction_probability)
    printDivision()

    plt.hist(y_prediction_probability)
    plt.title("Prediction Probability")
    plt.show()

    fp, tp, thresholds = metrics.roc_curve(ytest, y_prediction_probability)
    plt.plot(fp, tp)
    plt.grid(True)
    plt.title("ROC Curve")
    plt.show()
    print("ROC ACU Score", metrics.roc_auc_score(ytest, y_prediction_probability)) # useful even with high class imbalance
    printDivision()
    return


def plotHeatmap(X, title):
    # Plot a heat map of all features and look for correlations
    fig, ax = plt.subplots()

    corr = X.corr()
    midpoint = (corr.values.max() - corr.values.min()) / 2
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt='.2g',
        square=True,
        cmap='coolwarm',
        center = 0.0,
        vmin = -1.0,
        vmax = 1.0
    )
    plt.title(title)
    plt.show()
    return


def plotFeatures(dataFrame):
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('tab20')
    num = 0
    for column in dataFrame.drop('drivingStyle', axis=1):
        num += 1
        plt.plot(dataFrame[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
        # Add legend
        plt.legend(loc=2, ncol=2)
    plt.title('Unscaled ' + car_variants[dataset_num] + ' Data Set')
    plt.show()
    return


def printDataInfo(dataFrame, featureName):
    print("Mean: %0.3f" % dataFrame[featureName].mean())
    print('Max: %0.3f' % dataFrame[featureName].max())
    print('Min:  %0.3f' % dataFrame[featureName].min())
    print('Median:  %0.3f' % dataFrame[featureName].median())
    return

# This function reads comma-separated values (csv) file into a DataFrame.
# As my data is stored in a folder inside the project one folder up form the source I use ..\ to navigate out od source
# then \data\ to get inside the data folder. It has a sub folder kaggle_input. The csv files use I have used either use
# , or a ; as a delimiter
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
readgroupc = 1
readorig = 0
if readgroupc == 1:
    printDivision()
    print("Using the dataset provide by Sean with mods made to remove multi sensor failure entries")

    df1 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/opel_corsa_01.csv', sep=";")
    df2 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/opel_corsa_02.csv', sep=";")
    df3 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/peugeot_207_01.csv', sep=";")
    df4 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/peugeot_207_02.csv', sep=";")
elif readorig == 1:
    printDivision()
    print("Using the original Kaggle dataset ")
    df1 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/original/opel_corsa_01.csv', sep=";", decimal=',')
    df2 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/original/opel_corsa_02.csv', sep=";", decimal=',')
    df3 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/original/peugeot_207_01.csv', sep=";", decimal=',')
    df4 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/original/peugeot_207_02.csv', sep=";", decimal=',')
else:
    printDivision()
    print("Using the dataset provide by Sean unmodified")
    df1 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/Sean/opel_corsa_01.csv', sep=";")
    df2 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/Sean/opel_corsa_02.csv', sep=";")
    df3 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/Sean/peugeot_207_01.csv', sep=";")
    df4 = pd.read_csv('https://raw.githubusercontent.com/ratkomrda/MLAssignment/master/data/kaggle_input/Sean/peugeot_207_02.csv', sep=";")

#Choose the files you want to use
frames = [df1, df2, df3, df4]

feature_names = list(frames[0].head(0))
# Remove the last three columns form this list as they are not features they are classes
feature_names = feature_names[:len(feature_names)-3]
if readorig != 1:
    # in the case where the data was modified by Sean the first column is an index and needs to be removed
    len(feature_names)
    feature_names = feature_names[1:]
    len(feature_names)

# style
car_variants = ['Opel_Corsa_01', 'Opel_Corsa_02', 'Peugeot_207_01', 'Peuggeot_207_02']
dataset_num = 0




for df in frames:
    printDivision()
    print(car_variants[dataset_num])
    if readorig != 1:
        df.drop(['Unnamed: 0', 'roadSurface', 'traffic'], axis=1, inplace=True)
    else:
        df.drop(['roadSurface', 'traffic'], axis=1, inplace=True)

    for feature_index in feature_names:
        printDivision()
        print(feature_index)
        printDataInfo(df, feature_index)
        printDivision()


#    plotFeatures(df)
    # Add titles
    for feature_index in feature_names:
        # Find all missing or NAN values and replace with the mean as calculated for this car.
        num_nulls = df[feature_index].isnull()
        sum_nulls = num_nulls.sum().sum()
        if (sum_nulls > 0):
            print(feature_index + ' has ' + str(sum_nulls) + ' nulls')
        df[feature_index].fillna(df[feature_index].mean(), inplace=True)
#    plotHeatmap(df, 'Unscaled ' + car_variants[dataset_num] + ' Features Heatmap')
    dataset_num += 1
    printDivision()
    # display
    plt.show()

#concatente multiple data csv files
data = pd.concat(frames)
#######################################################################################################################
# Take the data and
#######################################################################################################################
# Purely integer-location based indexing for selection by position.
# From importing the files provided Sean into excel and the data provided in the pdf I saw that each file has 18
# columns which are integer indexed from 0 - 17. The first column whose index is 0 is not data from a sensor it
# is row count left over form the original conversion. I skipped this by starting to index form 1. The last 3 columns
# contain the classes so the features end at column 14. Having looked at the data I opted to go with the drivingStyle
# as it had only two classes EvenPaceStyle and AggressiveStyle each of the others had three classes.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# Xorig contains all of the feature columns
Xorig = data.iloc[:, 0:14]

plt.style.use('seaborn-darkgrid')
# create a color palette
palette = plt.get_cmap('tab20')
num = 0
for column in Xorig:
    num += 1
    plt.plot(Xorig[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
    # Add legend
    plt.legend(loc=2, ncol=2)
plt.title("Unscaled Combined Features")
plt.show()

# y contains the labels for the drivingStyle class it has either
# if you get the error IndexError: single positional indexer is out-of-bounds on this line check the delimiter
yorig = data.iloc[:, 14]

plotHeatmap(Xorig, "Unscaled Combined Features Heatmap")
# Create a series of boolean values which are true for all locations in y that have the label EvenPaceStyle and false
# otherwise
positive = yorig == 'EvenPaceStyle'
# Create a series of boolean values which are true for all locations in y that have the label AggressiveStyle and false
# otherwise
negative = yorig == 'AggressiveStyle'

# Create a new array of given shape and type, filled with zeros.
# Create a 1D array the same length as yfilled with zeros named yinput
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
target = np.zeros((len(yorig), 1)).ravel()

# This will set the value of yinput to 1 for the indices of positive which where had true written to them at line 36
target[positive] = 1
total_positive = np.sum(positive)
# This is unneeded as yinput was initialised with 0s but I include it for completeness. It will set the value of yinput
# to 0 for the indices of negative that were set to true at line 39
target[negative] = 0
total_negative = np.sum(negative)

# look at scaling
X_scaler = PowerTransformer(method='yeo-johnson')
#X_scaler = Normalizer()
#X_scaler = StandardScaler(with_mean=True, with_std=True)
#X_scaler = RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True)

Xorig = X_scaler.fit_transform(Xorig)
plt.plot(Xorig)
plt.title("Scaled Combined Features")
plt.show()

printDivision()
print('Class 0:', total_negative)
print('Class 1:', total_positive)
print('Proportion:', round(total_negative / total_positive, 2), ': 1')
printDivision()

# Split arrays or matrices into random train and test subsets
# Split the data set using the value which test_size is set to determine the proportion of the split. The data is split
# randomly using .4 means that 40% of the data goes to test 60% goes to train. random_state=some_number guarantees that
# the output of the split will be always the same if this is not set then each time the split will be different
# preventing reproducible results
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(Xorig, target, test_size=.4, random_state=23)

# First check what the accuracy would be if Even driving is predicted everytime
ones_percentage = y_test.mean()
zeros_percentage = 1 - y_test.mean()
# This will not work for multivariate classifier only for binary
null_accuracy = max(ones_percentage, zeros_percentage)
print("====================================================")
print("Accuracy when Even driving is predicted 100%", null_accuracy)

# Balance teh data set based on the negative class count which is fewer.

df_class_0 = data[data['drivingStyle'] == 'AggressiveStyle']
df_class_1 = data[data['drivingStyle'] == 'EvenPaceStyle']

# Randomly select the number of negative samples from the positive class to balance the dataset
df_class_1_under = df_class_1.sample(total_negative, random_state=23)
undersampleddata = pd.concat([df_class_0, df_class_1_under], axis=0)

Xunder = undersampleddata.iloc[:, 0:13]
# y contains the labels for the drivingStyle class it has either
# if you get the error IndexError: single positional indexer is out-of-bounds on this line check the delimiter
yunder = undersampleddata.iloc[:, 14]
plt.plot(Xunder)
plt.title("Under-sampled Scaled Combined Features")
plt.show()
plotHeatmap(Xunder, "Under-sampled Scaled Combined Features Heatmap")
# Create a series of boolean values which are true for all locations in y that have the label EvenPaceStyle and false
# otherwise
positive = yunder == 'EvenPaceStyle'
# Create a series of boolean values which are true for all locations in y that have the label AggressiveStyle and false
# otherwise
negative = yunder == 'AggressiveStyle'

# Create a new array of given shape and type, filled with zeros.
# Create a 1D array the same length as yfilled with zeros named yinput
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
targetunder = np.zeros((len(yunder), 1)).ravel()

# This will set the value of yinput to 1 for the indices of positive which where had true written to them at line 36
targetunder[positive] = 1
total_positive = np.sum(positive)
# This is unneeded as yinput was initialised with 0s but I include it for completeness. It will set the value of yinput
# to 0 for the indices of negative that were set to true at line 39
targetunder[negative] = 0
total_negative = np.sum(negative)
printDivision()
print('Class 0:', total_negative)
print('Class 1:', total_positive)
print('Proportion:', round(total_negative / total_positive, 2), ': 1')
printDivision()


# Split arrays or matrices into random train and test subsets
# Split the data set using the value which test_size is set to determine the proportion of the split. The data is split
# randomly using .4 means that 40% of the data goes to test 60% goes to train. random_state=some_number guarantees that 
# the output of the split will be always the same if this is not set then each time the split will be different 
# preventing reproducible results
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(Xunder, targetunder, test_size=.4, random_state=23)

# First check what the accuracy would be if Even driving is predicted everytime
ones_percentage = y_test.mean()
zeros_percentage = 1 - y_test.mean()
# This will not work for multivariate classifier only for binary
null_accuracy = max(ones_percentage, zeros_percentage)
print("====================================================")
print("Accuracy when Even driving is predicted 100%", null_accuracy)

# Logistic Regression classifier. This class implements regularized logistic regression using the ‘liblinear’ library,
# ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is applied by default. Based on the
# documentation which recommended liblinear for small data sets.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
logreg = LogisticRegression(solver='liblinear')

# Fit the model according to the given training data
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit
#logreg.fit(X_train, y_train)
# Predict class labels for samples in the test data X_test
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict
#ypred_logreg = logreg.predict(X_test)
calssifier_results = []
# benchmark runs the train and predict functions and times their duration it also returns accuracy and all three results
# will be plotted and compare once they have been added to classifer_results
logresults = benchmark("Logistic Regression", logreg, X_train, y_train, X_test, y_test)
calssifier_results.append(logresults)

# Print and plot metrics for the classifier
printDivision()
print("Logistic Regression")
printPlotMetrics(y_test, X_test, getPredictedTarget(calssifier_results), logreg)

printDivision()
#print("Logistic Regression Entire Dataset Results")
#ypred_orig = logreg.predict(Xorig)
#printPlotMetrics(target, Xorig, ypred_orig, logreg)

# Random Forest classifier. A random forest is a meta estimator that fits a number of decision  tree classifiers on
# various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement
# if bootstrap=True which is the default. n_estimators default=10 The number of trees in the forest. random_state
# is a number used to initialise the random generator default=None
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf = RandomForestClassifier(n_estimators=200, random_state=23)

calssifier_results.append(benchmark("Random Forest", rf, X_train, y_train, X_test, y_test))


# Print and plot metrics for the classifier
printDivision()
print("Random Forest Results")
printPlotMetrics(y_test, X_test, getPredictedTarget(calssifier_results), rf)

printDivision()
#print("Random Forest Entire Dataset Results")
#ypred_orig = rf.predict(Xorig)
#printPlotMetrics(target, Xorig, ypred_orig, rf)

# Create a series of the the features and plot their importance
feat_important = pd.Series(rf.feature_importances_, index=X_train.columns)
indices = np.argsort(feat_important)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), feat_important[indices], align='center')
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
# Meta-transformer for selecting features based on importance weights.
# Reduce the Feature selection based on their importance
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
features = SelectFromModel(rf)
features.fit(X_train, y_train)

printDivision()
print("The reduced features to be used in the Random Forest")
# Print the features that will be used in the reduced set
for feature_list_index in features.get_support(indices=True):
    print(feature_names[feature_list_index])
# Create a reduced set of features from those selected by SelectFromModel ensure that the training and test set are both
# updated
X_train_sel = features.transform(X_train)
X_test_sel = features.transform(X_test)
#X_test_orig_sel = features.transform(Xorig)

# Random Forest classifier. A random forest is a meta estimator that fits a number of decision  tree classifiers on
# various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement
# if bootstrap=True which is the default. n_estimators default=10 The number of trees in the forest. random_state
# is a number used to initialise the random generator default=None
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf_sel = RandomForestClassifier(n_estimators=200, random_state=23)

calssifier_results.append(benchmark("Random Forest Reduced Features", rf_sel, X_train_sel, y_train, X_test_sel, y_test))

printDivision()
print("Random Forest Reduced Features Results")
printPlotMetrics(y_test, X_test_sel, getPredictedTarget(calssifier_results), rf_sel)

printDivision()
#print("Random Forest Reduced Features Entire Dataset Results")
#ypred_orig = rf_sel.predict(X_test_orig_sel)
#printPlotMetrics(target, X_test_orig_sel, ypred_orig, rf_sel)

indices = np.arange(len(calssifier_results))
calssifier_results = [[x[i] for x in calssifier_results] for i in range(4)]
# Initialise the variables with the values in calssifier_results
clfnames, accuracy, train, test = calssifier_results
train = np.array(train) / np.max(train)
test = np.array(test) / np.max(test)

plt.title("Accuracy, Training and Test Times")
plt.barh(indices, accuracy, .2, label="Accuracy", color='blue')
plt.barh(indices + .3, train, .2, label="Training Duration", color='green')
plt.barh(indices + .6, test, .2, label="Test Duration", color='orange')
plt.yticks(())
plt.grid(True)
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clfnames):
    plt.text(-.3, i, c)
plt.show()
