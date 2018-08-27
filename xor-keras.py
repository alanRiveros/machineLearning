from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential([
    Dense(8, input_shape=(2,), activation='tanh'),
    Dense(1, activation='sigmoid')
])
sgd = SGD(lr=0.1)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x, y, batch_size=1, epochs=1000, shuffle=True, verbose=2)

print(model.predict(np.array([[0,1]])))
print(model.predict(np.array([[1,1]])))