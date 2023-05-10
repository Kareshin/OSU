import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import MinMaxScaler
from sklearn . preprocessing import OneHotEncoder
import sklearn . linear_model as lm
from sklearn import metrics

#a
data = pd.read_csv('data_C02_emission.csv')

input_variables = ['Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)',
                   'Engine Size (L)','Cylinders']

output_variables = ['CO2 Emissions (g/km)']
X = data[input_variables].to_numpy()
y = data[output_variables].to_numpy()

#podjela na učenje i testiranje u omjeru 80-20
X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )

#b
plt.scatter(X_train[:,0],y_train,c='blue',s=2)
plt.scatter(X_test[:,0],y_test,c='red',s=2)
plt.show()

#c standardizacija
plt.figure()
plt.hist(X_train[:,0])
plt.show()

# min - max skaliranje - kako bi se sve svelo na isti interval
sc = MinMaxScaler () #transform vraca numpy array, zato se mora nazad u dataframe (da se sve radilo s numpy, ne bi bilo potrebe za time)
#X_train_n = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index) //ovako se vraća u dataframe


# transformiranje ulazne veličine 
X_train_n = sc.fit_transform ( X_train ) #fit_transform() ide na training data
X_test_n = sc.transform ( X_test )       #transform() ide na testing data

plt.hist(X_train_n[:,0])
plt.show()

#d

linearModel=lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(f'Parametri modela: {linearModel.coef_}')
print(f'Intercept parametar: {linearModel.intercept_}')

#e
y_test_p = linearModel . predict ( X_test_n )
plt.scatter(y_test, y_test_p, s=2) # što su rezultati sličniji pravcu, to je točnije
plt.show()

#f

# f Najbolji model (savršeno procjenjuje vrijednosti izlazne velicine) za dane podatke ima vrijednost 1
r2s = metrics.r2_score(y_test, y_test_p)
print(r2s)

#g
# # Povecavanjem broja ulaznih podataka imamo bolje učenje, tj. r2_score je blize 1

