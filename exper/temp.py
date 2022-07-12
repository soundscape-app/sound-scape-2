# mlp for multi-output regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
df = pd.read_csv('result_data.csv', index_col = 0)
# get the dataset
def get_dataset():
	X = df[["LAeq", "LA5-95","Loudness", "Green", "Sky", "Grey", ]]
	y = df[["Fascination","Being_away_from","Negative", "Valence", "Arousal"]]
	#y = df[["Fascination"]]
    
	X = X.values
	y = y.values
	X_scaled = X
	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)
    
	return X_scaled, y
 
# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mae', optimizer='adam')
	return model
 
# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		#model = LinearRegression()
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		#model.fit(X_train, y_train)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		#r_sq = model.score(X_train, y_train)
		# store result
		print('>%.3f' % mae)
		#print('>%.3f' % r_sq)
		results.append(mae)
		#results.append(r_sq)
		#print('intercept:', model.intercept_)
		#print('slope:', model.coef_)
		
	return results
 # load dataset
X, y = get_dataset()
n_inputs, n_outputs = X.shape[1], y.shape[1]
model = nn.Sequential(
    nn.Linear(1,5),
    nn.LeakyReLU(0.2),
    nn.Linear(5,10),
    nn.LeakyReLU(0.2),
    nn.Linear(10,10),
    nn.LeakyReLU(0.2),    
    nn.Linear(10,10),
    nn.LeakyReLU(0.2),        
    nn.Linear(10,5),
    nn.LeakyReLU(0.2),          
    nn.Linear(5,1),
)
model.fit(X, y, verbose=0, epochs=100)
print(model.evaluate(X, y, verbose=0))
model.save("model2.h5")
# poly = PolynomialFeatures(degree=2, include_bias=True)
# X_train_poly = poly.fit_transform(X.reshape(-1, 1))
# lin_reg = LinearRegression()
# lin_reg.fit(X_train_poly, y)
# y_pred = lin_reg.predict(X_train_poly)
# print(lin_reg.score(X, y))
# evaluate model
#results = evaluate_model(X, y)
# summarize performance
#print('mae: %.3f (%.3f)' % (mean(results), std(results)))

