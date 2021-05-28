import pandas as pd
import numpy as np
dataset = pd.read_csv("Salary_Data.csv")
Y = dataset["Salary"]
X = dataset["YearsExperience"]
X = np.array(X).reshape(30,1)
Y = np.array(Y).reshape(30,1)
from sklearn.linear_model import LinearRegression
mind = LinearRegression()
mind.fit(X,Y)
print("Weight Is :",mind.coef_)
print("Bias Is : ",mind.intercept_)
import joblib
joblib.dump(mind,"model_salary_predict.pk1")
