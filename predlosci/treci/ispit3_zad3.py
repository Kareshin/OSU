from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import load_model

from tensorflow.python.keras.layers import Dense
iris = datasets.load_iris()
from keras.utils import to_categorical
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

sc = StandardScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)
#A)
model = keras.Sequential()
model.add(keras.layers.Input(shape=(4,)))
model.add(keras.layers.Dense(units=10, activation="relu"))
model.add ( keras.layers.Dropout (0.3) )
model.add(keras.layers.Dense(units=7, activation="relu"))
model . add (keras.layers.Dropout (0.3) )
model.add(keras.layers.Dense(units=5, activation="relu"))
model.add(keras.layers.Dense(units=3, activation="softmax"))
model.summary()

#B)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#C)
history = model.fit(X_train, y_train, batch_size=7, epochs=500, validation_split=0.1)
#D)
model.save("Model/")
#E)
model = load_model("Model/")
score = model.evaluate(X_test, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')
#F)
y_predictions = model.predict(X_test)
y_predictions = np.around(y_predictions).astype(np.int32)
cm = confusion_matrix(y_test.argmax(axis=1),y_predictions.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()