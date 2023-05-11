import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import time

NAME = "cifar10Model-{}".format(int(time.time())) #nije nuzno

# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype ="uint8")
y_test = to_categorical(y_test, dtype ="uint8")

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = "logs1\\{}".format(NAME), #formatiranje nije nuzno
                                update_freq = 100),
    keras.callbacks.EarlyStopping(monitor="val_loss", 
                                patience=5, verbose=1)
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_n,
            y_train,
            epochs = 5,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

#zad1
#preciznost na skupu za učenje: 0.9877
#točnost na testnom skupu: 0.7322
#tijekom učenja je došlo do overfitanja

#zad2 - dodavanje dropout sloja (broj epoha smanjen na 20 radi brzine)
#preciznost na skupu za učenje: 0.9611
#točnost na testnom skupu: 0.7539
#tijekom učenja overfit je malo manji

#zad3 - early stop nakon 5 uzastopnih epoha bez smanjenja prosječne vrijednosti gubitka
#preciznost na skupu za učenje: 0.9389
#točnost na testnom skupu: 0.7333
#staje nakon 10. epohe, i trening i testni skup su lošiji u odnosu na dropout