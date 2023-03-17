import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

#A
plt.hist(data["CO2 Emissions (g/km)"], bins=50, color='steelblue', edgecolor='gray')
plt.xlabel("Interval of CO2 emissions (g/km)")
plt.ylabel("No. of cars in category")
plt.show()

#B
colors = { 'X': 'blue', 'Z': 'green', 'D': 'red', 'E': 'orange', 'N': 'black' }
data.plot.scatter(x='Fuel Consumption City (L/100km)',
                    y='CO2 Emissions (g/km)',
                    c=data['Fuel Type'].apply(lambda x: colors[x]), cmap ='viridis', s=10)
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