import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Modell definieren
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(64, input_shape=(1, 3)))
model.add(tf.keras.layers.Dense(1))

#Modell kompilieren
model.compile(optimizer='adam', loss='mse')

#Trainingsdaten einlesen und form anpassen und aufteilen in Trainings- und Testdaten
train_inputs = np.genfromtxt('inputdata.csv',delimiter=',')
train_outputs = np.genfromtxt('outputdata.csv',delimiter=',')
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(train_inputs, train_outputs, test_size=0.2, random_state=42)

train_inputs = np.expand_dims(train_inputs, axis=0)
train_inputs = np.transpose(train_inputs, (1, 0, 2))
train_outputs = np.expand_dims(train_outputs, axis=1)

test_inputs = np.expand_dims(test_inputs, axis=0)
test_inputs = np.transpose(test_inputs, (1, 0, 2))
test_outputs = np.expand_dims(test_outputs, axis=1)

#Modell trainieren
model.fit(train_inputs, train_outputs, epochs=10, batch_size=32)

#Modell testen und bewerten
test = model.evaluate(test_inputs, test_outputs)