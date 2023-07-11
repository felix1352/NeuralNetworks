import numpy as np
import csv

# Eingang ist der Startwinkel und Ausgang ist der Verlauf des Winkels

# Berechnung des Winkels über die Zeit für Mathematisches Pendel
#Variablen deklarieren
l = 2
m = 1
g = 9.81
b = 0.7
T = 10  #simulation Time
dt = 0.1  # Schrittweite für numerisches Lösen


def f(t, x):
    phi = x[0]
    y = x[1]
    dphidt = y
    dydt = 1/l*(-b*y-g*np.sin(phi))
    return np.array([dphidt,dydt])

def berechne_winkel(phi0):
    t = 0
    phi = phi0
    y = 0
    erg = np.array([])
    while t < T:
        k1 = dt * f(t, [phi, y])
        k2 = dt * f(t + dt / 2, [phi + k1[0] / 2, y + k1[1] / 2])
        k3 = dt * f(t + dt / 2, [phi + k2[0] / 2, y + k2[1] / 2])
        k4 = dt * f(t + dt, [phi + k3[0], y + k3[1]])
        phi += (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y += (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        t += dt
        erg = np.append(erg, phi)
    return erg

#Trainingsdaten generieren
num_samples = 1000;
train_inputs = np.empty((1,3))
train_outputs = np.array([0])
startwinkel_train = np.random.uniform(1.9999, 2.0001, num_samples)
for g in startwinkel_train:
    winkelverlauf = berechne_winkel(g)
    for i in range(2,len(winkelverlauf)-2):
        input = np.array([winkelverlauf[i], winkelverlauf[i-1], winkelverlauf[i-2]])
        output = np.array([winkelverlauf[i + 1]])
        if g == startwinkel_train[0]:
            if i==2:
                train_inputs = input
                train_outputs = output
            else:
                train_inputs = np.vstack((train_inputs, input))
                train_outputs = np.vstack((train_outputs, output))
        else:
            train_inputs = np.vstack((train_inputs, input))
            train_outputs = np.vstack((train_outputs, output))

with open('inputdata.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in train_inputs:
        writer.writerow(row)

with open('outputdata.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in train_outputs:
        writer.writerow(row)
