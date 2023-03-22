import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


data = pd.read_csv('data_C02_emission.csv')
y=data['CO2 Emissions (g/km)'].copy()
X=data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]

#a

X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1) 


#b

plt.scatter(X_train['Fuel Consumption City (L/100km)'],y_train, c='b', label='Train', s=1)
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_test, c='r', label='Test', s=1)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

#c

sc = MinMaxScaler () #transform vraca numpy array
X_train_n = pd.DataFrame(sc.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
for col in X_train.columns:
    fig,axs = plt.subplots(2,figsize=(8, 8))
    axs[0].hist(X_train[col])
    axs[0].set_title('Before scaler')
    axs[1].hist(X_train_n[col])
    axs[1].set_xlabel(col)
    axs[1].set_title('After scaler')
    plt.show()
X_test_n = pd.DataFrame(sc.transform(X_test.values), columns=X_test.columns, index=X_test.index)


#d

linearModel=lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(f'Parametri modela:{linearModel.coef_}')

#e

y_prediction = linearModel.predict(X_test_n) #vraca numpy array
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_test, c='b', label='Real values', s=1)
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_prediction, c='r', label='Prediction', s=1)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

#f

print(f'Mean squared error: {mean_squared_error(y_test, y_prediction)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_prediction)}')
print(f'Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_prediction)}%')
print(f'R2 score: {r2_score(y_test, y_prediction)}')

