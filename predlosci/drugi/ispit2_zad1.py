import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("titanic.csv")
#arr = np.loadtxt("titanic.csv", delimiter=",", skiprows=1)

print(data)
#A)
print("Broj podataka: ", len(data))
#B)
data_survived = data[data["Survived"]==1]
print("Broj prezivjelih: ", len(data_survived))

#C)
male = data[data["Sex"]=="male"]
survived_male = male[male["Survived"]==1]
print("Broj muskaraca: ", len(male))
print("Broj prezivjelih muskaraca: ", len(survived_male))
postotak_p_m=len(survived_male)/len(male)*100
print("postootak prezivjelih muskaraca: ", postotak_p_m, "%")

female = data[data["Sex"]=="female"]
survived_female = female[female["Survived"]==1]
print("Broj zena: ", len(female))
print("Broj prezivjelih zena: ", len(survived_female))
postotak_p_z=len(survived_female)/len(female)*100
print("postootak prezivjelih zena: ", postotak_p_z, "%")

#plt.figure()

#plt.show()
#prezivjelo je veci postotak zena, nego muskaraca jer su zene i djeca imale prioritet za camce

#D)
print("prosjecna dob prezvjlog muskarca: ", survived_male["Age"].mean())
print("prosjecna dob prezivjele zene: ", survived_female["Age"].mean())

#E)
class1 = survived_male[survived_male["Pclass"]==1]
print(class1["Age"].min())
class2 = survived_male[survived_male["Pclass"]==2]
print(class2["Age"].min())
class3 = survived_male[survived_male["Pclass"]==3]
print(class3["Age"].min())
#najmladi prezvijeli muskarci su bebe od par mjeseci