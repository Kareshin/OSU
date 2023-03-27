import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('data_C02_emission.csv')
X = data[['Engine Size (L)', 'Cylinders', 'Fuel Type', 'Fuel Consumption City (L/100km)',
          'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
y = data['CO2 Emissions (g/km)'].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
linearModel = lm.LinearRegression()
linearModel.fit(X_encoded, y_train)
y_prediction = linearModel.predict(X_test)
print("Mean squared error: ", mean_squared_error(y_test, y_prediction))
print("Mean absolute error: ", mean_absolute_error(y_test, y_prediction))
print("Mean absolute percentage error: ",
      mean_absolute_percentage_error(y_test, y_prediction), "%")
print("R2 score: ", r2_score(y_test, y_prediction))

print("Max error: ", max_error(y_test, y_prediction))
error = abs(y_prediction - y_test)
id = np.argmax(error)
print(data[id]['Make'], ['Model'])