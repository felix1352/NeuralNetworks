import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
#Das ist ein Feed Forward Neural Network in dem als Input die drei letzten Positionen eingegeben werden und als output die nächste Position bestimmt werden soll

#Modell definieren
model = Sequential()
model.add(Dense(units = 100, input_shape=(4,), activation='relu'))   #Eingangschicht
model.add(Dense(units = 100, activation='relu'))                #Mittelschichten
#model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1))                                   #Ausgangsschicht

#Modell kompilieren
model.compile(optimizer='adam', loss='mse')

#Trainingsdaten einlesen und form anpassen und aufteilen in Trainings- und Testdaten
train_inputs = np.genfromtxt('inputdata.csv',delimiter=',')
train_outputs = np.genfromtxt('outputdata.csv',delimiter=',')
train_inputs, validate_inputs, train_outputs, validate_outputs = train_test_split(train_inputs, train_outputs, test_size=0.2, random_state=42)

#Modell trainieren
filepath = "\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsFFnn\\weights-{epoch:02d}-{val_loss:.7f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
callbacks_list = [checkpoint]
#history = model.fit(train_inputs.reshape(-1,4), train_outputs.reshape(-1,1), epochs=10, batch_size=1, callbacks=callbacks_list, validation_data=(validate_inputs,validate_outputs), verbose=2)
model.load_weights("\\Users\\Felix Koch\\Documents\\BachelorArbeit\\ModelsFFnn\\weights-06-0.0000525.hdf5")

model.save('FFNN_Einfachpendel.h5')

#Plotten von Trainingsloss und Validationloss
#plt.figure()
#plt.plot(history.epoch,history.history['loss'], label='loss')
#plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
#plt.legend()
#plt.show()