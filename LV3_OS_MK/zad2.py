import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

#A
plt.hist(data["CO2 Emissions (g/km)"], bins=50, color='steelblue', edgecolor='gray')
plt.xlabel("Interval of CO2 emissions (g/km)")
plt.ylabel("No. of cars in category")
plt.show()

#B

colors=["r","b","g","k","y"]
gas=['X','Z','D','E','N']

for i in range(0,5):
     data1=data[data['Fuel Type']==gas[i]]
     x=np.array(data1['Fuel Consumption City (L/100km)'])
     y=np.array(data1['CO2 Emissions (g/km)'])
     plt.scatter(x,y, c=colors[i], s=2 )
plt.show()

#C
grouped_fuel_type = data.groupby('Fuel Type')

grouped_fuel_type.boxplot(column=['Fuel Consumption Hwy (L/100km)'])
plt.show()

#D
grouped_fuel_type['Make'].count().plot(kind="bar")
plt.ylabel("No. of cars with fuel type")
plt.show()

#E
grouped_cylinders = data.groupby("Cylinders").mean()
grouped_cylinders["CO2 Emissions (g/km)"].plot(kind="bar")
plt.ylabel("Average CO2 Emissions (g/km)")
plt.show()
