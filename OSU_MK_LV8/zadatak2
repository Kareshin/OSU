import numpy as np
from tensorflow import keras
from tensorflow.keras import layers         #iako je podcrtano radi
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

KERAS_MODEL_NAME = "Model/keras.hdf5"
model = keras.models.load_model(KERAS_MODEL_NAME)

x_test_reshaped = np.reshape(x_test, (len(x_test), x_test.shape[1]*x_test.shape[2])) #od x_test 1D radimo 2D array kako bi mogli dalje raditi
#y_test = OneHotEncoder().fit_transform(np.reshape(y_test, (-1, 1))).toarray() //nepotrebna linija
#pretvaranje 7 u binarni format [0,0,0,0,0,0,0,1,0,0], -1 oznacava automatski broj redaka


y_pred = model.predict(x_test_reshaped)
y_pred = np.argmax(y_pred, axis=1)      #kako bi od [0,0,0,0,0,1,0,0,0,0] dobili 5

#y_test = np.argmax(y_test, axis=1) nepotrebna linija koda

print(y_test.shape, y_pred.shape)
y_bool = y_test == y_pred
print(y_bool, y_bool.shape)

indexes = np.where(y_bool==False) #tuper (array([...])) pa je 2D

for i in range(1, 6):
    plt.imshow(x_test[indexes[0][i]])
    title = f"Stvarna klasa: {y_test[indexes[0][i]]}, Predikcija: {y_pred[indexes[0][i]]}" 
    plt.title(title)
    plt.show()