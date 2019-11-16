#https://www.youtube.com/watch?v=V0u6bxQOUJ8
import numpy as np
from sklearn.impute import SimpleImputer
import glob, os
import pandas as pd
from itertools import combinations
from sklearn.impute import SimpleImputer
import sklearn.feature_selection

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from IPython.display import Image
pd.set_option('display.max_colwidth', -1) # displays content from given folder



#================================
#function to rename given files
#def rename(dir, pathAndFilename, pattern, titlePatteren):
#    os.remane(pathAndFilename, os.path.join(dir, titlePatteren)) # methdoe to do the rename

# search for csv files in working folder
#path = os.path.expanduser("../data/kaggle_input/*.csv") #pass folder name here

#iterate and remane them one by one with the number of the iteration
#for i, fname in enumerate(glob.glob(path)): # change file name to add numbee
#    rename(os.path.expanduser('../data/kaggle_input/'), fname, r'*.csv', r'test{}.csv'.format(i))# change file name without changing extention e.g ".csv"


#Change spearator for CSV file
df1 = pd.read_csv('../data/kaggle_input/opel_corsa_01.csv', sep=";")
df2 = pd.read_csv('../data/kaggle_input/opel_corsa_02.csv', sep=";")
df3 = pd.read_csv('../data/kaggle_input/peugeot_207_01.csv', sep=";")
df4 = pd.read_csv('../data/kaggle_input/peugeot_207_02.csv', sep=";")
#Choose the files you want to use
frames = [df1, df2, df3, df4]

#concatente multiple data csv files
data = pd.concat(frames)

#print(df1.shape)
#print(df2.shape)
#print(df3.shape)
#print(df4.shape)
#print(data.shape)

#result = pd.DataFrame() # creates empty data frame
#================================
#===========================
# Simple way to fill in missing data
#Working with missing data
print(data.isnull().sum())

#===============================================================
#Impute Mean
VehicleSpeedInstantaneous = data['VehicleSpeedInstantaneous'].mean()
data['VehicleSpeedInstantaneous'].fillna(VehicleSpeedInstantaneous, inplace=True)
print(VehicleSpeedInstantaneous)

EngineRPM = data['EngineRPM'].mean()
data['EngineRPM'].fillna(EngineRPM, inplace=True)
print(EngineRPM)

IntakeAirTemperature = data['IntakeAirTemperature'].mean()
data['IntakeAirTemperature'].fillna(IntakeAirTemperature, inplace=True)
print(IntakeAirTemperature)

EngineCoolantTemperature = data['EngineCoolantTemperature'].mean()
data['EngineCoolantTemperature'].fillna(EngineCoolantTemperature, inplace=True)
print(EngineCoolantTemperature)

#===================================================
#Impute Median
EngineLoad = data['EngineLoad'].median()
data['EngineLoad'].fillna(EngineLoad, inplace=True)
print(EngineLoad)

MassAirFlow = data['MassAirFlow'].median()
data['MassAirFlow'].fillna(MassAirFlow, inplace=True)
print(MassAirFlow)

FuelConsumptionAverage = data['FuelConsumptionAverage'].median()
data['FuelConsumptionAverage'].fillna(FuelConsumptionAverage, inplace=True)
print(FuelConsumptionAverage)

ManifoldAbsolutePressure = data['ManifoldAbsolutePressure'].median()
data['ManifoldAbsolutePressure'].fillna(ManifoldAbsolutePressure, inplace=True)
print(ManifoldAbsolutePressure)

print(data.isnull().sum())

#================================
#print(data.head(5))

#print(data['drivingStyle'].value_counts())

data['drivingStyle'] = [0 if x == 'AggressiveStyle' else 1 for x in data['drivingStyle']]
data.drop(['Unnamed: 0', 'roadSurface','traffic'], axis =1, inplace=True)
X = data.drop('drivingStyle',1)
y = data.drivingStyle


#print (X.head(5))
#print(y.head(5))

#Categorical features
#raodsurface is either Smooth, uneven, full of holes
#print((X['roadSurface'].head(5)))
#========================================================
#print(pd.get_dummies(X['roadSurface']).head(5))
#print(pd.get_dummies(X['traffic']).head(5))

#========================================================
#How many uniqiue categories you have in each column
#for col_name in X.columns:
#    if X[col_name].dtype == 'object':
#        unique_cat = len(X[col_name].unique())
        #print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
#=========================================================
#List high to low with category has more in it
#print(X['roadSurface'].value_counts().sort_values(ascending=False).head(10))
#print(X['traffic'].value_counts().sort_values(ascending=False).head(10))
#=========================================================
# if low frequecy categories as "Other" anything that isn't "LowCongestionCondistion" will be other, should only be used for once off columns
#X['traffic'] = ['LowCongestionCondition' if x == 'LowCongestionCondition' else 'other' for x in X['traffic']]
#print(X['traffic'].value_counts().sort_values(ascending=False))
#==========================================================
#Create a list of features to dummy of all the string columns
#todummy_list = ['roadSurface','traffic','drivingStyle']
# Function to dummy all categorical variables used for model
#def dummy_data(data, todummy_list):
#    for x in todummy_list:
#        dummies = pd.get_dummies(data[x], prefix= x, dummy_na=False)
#        data = data.drop(x,1)
#        data = pd.concat([data, dummies], axis=1)
#        return data

#X = dummy_data(X, todummy_list)





#X.isnull().sum ().sort_values(ascending=False).head()
#impute missing values using imputer in sklearn.preprocessing

#def add_interactions(data):
#    combos = list(combinations(list(data.columns),2))
#    colnames = list(data.columns) + ['_'.join(x) for x in combos]

#    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
#    data = poly.fit_transform(data)
#    data = pd.DataFrame(data)
#    data.comlumns = colnames

#    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
#    data = data.drop(data.colums[noint_indicies], axis=1)
#    return data

#x = add_interactions(data)
#print(X.head(5))

#=========================================================================
#This is the magic part where we let the program hide data from itself
#=========================================================================
#Train and Test splitting data
#X_train = data set we are going to use to train Classifier with
#X_test = information we don't let he program see, so it can test itself
#y_train = the training data giving the answers we want
#y_test = data we test to see if program gives us the right answer
#Test split, part of SKlearn lib
#test size is the % of data to test
#Random state is just a random seed number, just grabs random numbers in the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

#Applying a stander scaler to optimise the results, imported with sklearn lib above
#fit transform is to fit the data better
sc = StandardScaler()
X_train = sc.fit_transform(X_train)#this reducs bias in columns that have high numbers or low numbers (changes most data in the X_training data to a value between 0 - 1)
X_test  = sc.transform(X_test)# we want to keep the train and test data the same
X_train[:10]#print out the 1st 10 data on the training data, this is just default

#==============================
#Random Forest Classifier
#Look mammy I'm programming AI, (Classifier with is a fancy way to say organise the data)
#used for medium size data set
#==============================
#Object label = Classifier (how many trees do you want)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)# just does a simple fit of the data we seperated out for training
pred_rfc = rfc.predict(X_test)# predect the test values
#How the Forest Classifier preforms
print("Random Forst")
print(classification_report(y_test, pred_rfc)) # how the test data compares to the predected values
print(confusion_matrix(y_test, pred_rfc))# this give us a matrix on the mislabels between good  and bad

#=================================
##SVM Classifier
#Support Vector Model
#Libary is pretty much the same as other libs
#=================================
clf = svm.SVC()#calling the function
clf.fit(X_train, y_train)# just does a simple fit of the data we seperated out for training
pred_clf = clf.predict(X_test)# predect the test values
#How the CLF model preformes
print("SVM Classifaction")
print(classification_report(y_test, pred_clf))# how the test data compares to the predected values
print(confusion_matrix(y_test, pred_clf))# this give us a matrix on the mislabels between good  and bad


#=================================
##Neural Network
#hidden layers is the nodes in the NN
#Good for text based code or big data sets, picture processing
#==================================
#object = Classifier(how many nodes in each layer, max many iterations
mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
#How the NN model preformes
print("Nerual Net")
print(classification_report(y_test, pred_mlpc))# how the test data compares to the predected values
print(confusion_matrix(y_test, pred_mlpc))# this give us a matrix on the mislabels between good  and bad

#Score the AI
from sklearn.metrics import accuracy_score #Test scrore
bn = accuracy_score(y_test, pred_rfc) #Labelling code for printing
dm = accuracy_score(y_test, pred_clf) #Labelling code for printing
cm = accuracy_score(y_test, pred_mlpc) #Labelling code for printing
print(bn, ' is the Forest score')
print(dm, ' is the Classifier score')
print(cm, ' is the Neural Network score')

