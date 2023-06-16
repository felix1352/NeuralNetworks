import numpy as np
import tensorflow as tf

#Trainingsdaten generieren
num_samples = 10000;
x_train = np.random.uniform(0, 10, num_samples)
y_train = x_train**2+4*x_train+4

#Modell definieren
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(16, input_shape=(1, 1)))
model.add(tf.keras.layers.Dense(1))

#Modell kompilieren
model.compile(optimizer='adam', loss='mse')

#Modell trainieren
x_train = np.expand_dims(x_train, axis=-1)
model.fit(x_train, y_train, epochs=100, batch_size=32)

#Modell testen
test_input = np.array([[0.5],[1.0],[9]])
test_richtigeserg = test_input**2+4*test_input+4
test_vorhersage = model.predict(test_input)

print('Vorhersage: ', test_vorhersage)
print('Exakter Ergebnis: ', test_richtigeserg)
print('Fehler: ', abs(test_vorhersage-test_richtigeserg))