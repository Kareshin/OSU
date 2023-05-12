from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df)

#A)
versicolor = df[df['species']=="versicolor"]
virginica = df[df['species']=="virginica"]
plt.figure()
plt.scatter(versicolor["petal width (cm)"], versicolor["sepal length (cm)"], color="blue")
plt.scatter(virginica["petal width (cm)"], virginica["sepal length (cm)"], color="red")
plt.title("Odnos duljina casica i latica od virginice i versicolor-a")
plt.xlabel("Duljina latice")
plt.ylabel("Duljina casice")
plt.legend(["versicolor plavo", "virginica crveno"])
plt.show()
#duljina virginice je veca i casice i latice naspram versicolora

#B)
versicolorSepalWidth = versicolor["sepal width (cm)"].mean()
virginicaSepalWidth = virginica["sepal width (cm)"].mean()
setosa = df[df['species']=="setosa"]
setosaSepalWidth = setosa["sepal width (cm)"].mean()
plt.figure()
values = {"versicolor":versicolorSepalWidth, "virginica":virginicaSepalWidth, "setosa":setosaSepalWidth}
values2 = list(values.values())
courses = list(values.keys())
plt.bar(courses, values2)
plt.title("Prosjek sirine latica")
plt.ylabel("Sirina latica")
plt.xlabel("Nazivi")
plt.show()
#Setosa ima najsiru laticu

#C)
vecaSirinaLatice = virginica[virginica["sepal width (cm)"]>virginicaSepalWidth]
print("Ima ih vise: ",len(vecaSirinaLatice))