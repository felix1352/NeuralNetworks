import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
#Das ist ein Feed Forward Neural Network in dem als Input die drei letzten Positionen eingegeben werden und als output die nächste Position bestimmt werden soll

#Modell definieren
model = Sequential()
model.add(Dense(units = 4, input_shape=(3,), activation='relu'))   #Eingangschicht
model.add(Dense(units=8, activation='relu'))                #Mittelschichten
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1))                                   #Ausgangsschicht

#Modell kompilieren
model.compile(optimizer='adam', loss='mse')

#Trainingsdaten einlesen und form anpassen und aufteilen in Trainings- und Testdaten
train_inputs = np.genfromtxt('inputdata.csv',delimiter=',')
train_outputs = np.genfromtxt('outputdata.csv',delimiter=',')
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(train_inputs, train_outputs, test_size=0.2, random_state=42)

#Modell trainieren
history = model.fit(train_inputs[:1000].reshape(-1,3), train_outputs[:1000].reshape(-1,1), epochs=10, batch_size=12, validation_split=0.3)

#Modell testen und bewerten
test = model.evaluate(test_inputs, test_outputs)

model.save('FFNN_Einfachpendel.h5')

plt.figure()
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

print(((model.predict(train_inputs[:1000])-train_outputs[:1000])**2).mean())