'''TaxiFare
Problem Statement: To prepare a regrassion model to predict the amount of the taxifare.

The goal of this project is to predict with reasonable accuracy, the taxi fare of new york city based on the dataset

Target: The target column fare_amount is a float dollar amount of the cost of the text ride.'''

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
palette = sns.color_palette("rainbow", 8)

# Regressors

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Model Evalution Tools

from sklearn.metrics import max_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

# Data Processing functions

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("TaxiFare (2) (1).csv")
print(df.head(5))

# find the shapre of the Data set
print(df.shape)

# find the info about dataset
print(df.info())

# data describing 
print(df.describe(include='all'))

# finding the NULL values
print(df.isnull().sum())

# finding the Duplicate and sum them

duplicate = df.duplicated()
duplicate.sum()
print(duplicate)


# Feature Engineering

df['date_time_of_pickup'] = pd.to_datetime(df['date_time_of_pickup'], format ="%Y-%m-%d %H:%M:%S UTC")
df['year'] = df.date_time_of_pickup.apply(lambda t: t.year)
df['weekday'] = df.date_time_of_pickup.apply(lambda t:t.weekday())
df['hour'] = df.date_time_of_pickup.apply(lambda t:t.hour)

def distance(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


df['distance'] = distance(df.latitude_of_pickup, df.longitude_of_pickup, df.latitude_of_dropoff, df.longitude_of_dropoff)

df = df.dropna()
df = df[df.amount > 0]
df = df[df.distance > 0]
df = df.drop(['unique_id','date_time_of_pickup'], axis=1)

print(df.head())

''' install this packages for all kind of visualization 
1.pip install autoviz
2.pip install xlrd
'''

from autoviz.AutoViz_Class import AutoViz_Class
import matplotlib.pyplot as plt
import seaborn as sns
AV = AutoViz_Class()

filename = "TaxiFare (2) (1).csv"
sep = ","
dft = AV.AutoViz(
    filename,
    sep=",",
    depVar="",
    dfte=None,
    header=0,
    verbose=0,
    lowess=False,
    chart_format="svg"
)


# checking the null values

df.isnull().sum()
print(df)
print(df.head(10))

# dropping the unwanted columns
df=df.drop(["longitude_of_pickup","latitude_of_pickup","longitude_of_dropoff","latitude_of_dropoff"],axis=1)
X = df.drop(["amount","weekday"],axis = 1)
Y = df["amount"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

# Linear Regrassion
model1 = LinearRegression()
model1.fit(X_train,Y_train)
model1.score(X_train,Y_train)
model1.score(X_test,Y_test)

from sklearn.metrics import r2_score

Y_test_pred = model1.predict(X_test)
score = r2_score(Y_test, Y_test_pred)
print("The accuracy of our model1 is {}%".format(round(score,2)*100))

# Decision Tree Regression
model2 = DecisionTreeRegressor()
model2.fit(X_train,Y_train)
model2.score(X_train,Y_train)
model2.score(X_test,Y_test)

from sklearn.metrics import r2_score

Y_test_pred = model2.predict(X_test)
score = r2_score(Y_test,Y_test_pred)
print("The Accuracy of our model2 is {}%".format(round(score, 2)*100))

# AdaBoostRegrassion
model3=AdaBoostRegressor()
model3.fit(X_train,Y_train)
model3.score(X_train,Y_train)
model3.score(X_test,Y_test)

from sklearn.metrics import r2_score
Y_test_pred = model3.predict(X_test)
score = r2_score(Y_test, Y_test_pred)
print("The accuracy of the model3 is {}%".format(round(score,2)*100))

# GradiantBoostingRegressor
model4=GradientBoostingRegressor()
model4.fit(X_train,Y_train)
model4.score(X_train,Y_train)
model4.score(X_test,Y_test)

from sklearn.metrics import r2_score
Y_test_pred = model4.predict(X_test)
score = r2_score(Y_test, Y_test_pred)
print("The accuracy of the model4 is {}%".format(round(score,2)*100))

# RandomForestRegressor

model5=RandomForestRegressor()
model5.fit(X_train,Y_train)
model5.score(X_train,Y_train)
model5.score(X_test,Y_test)

from sklearn.metrics import r2_score
Y_test_pred = model5.predict(X_test)
score = r2_score(Y_test, Y_test_pred)
print("The accuracy of the model5 is {}%".format(round(score,2)*100))

# SVR (Support Vector Regrasso)

model6=SVR()
model6.fit(X_train,Y_train)
model6.score(X_train,Y_train)
model6.score(X_test,Y_test)

from sklearn.metrics import r2_score
Y_test_pred = model6.predict(X_test)
score = r2_score(Y_test, Y_test_pred)
print("The accuracy of the model6 is {}%".format(round(score,2)*100))