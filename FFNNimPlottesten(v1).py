import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

#Test im Plot vergleichen

#Variablen deklarieren
l = 2
b = 0.7
g = 9.81
T = 10 #simulation Time
def f(t, x):
    dxdt = x[1]
    dxdtdt = 1/l*(-b*x[1]-g*np.sin(x[0]))
    return np.array([dxdt,dxdtdt])

#RK4 Verfahren

dt = 0.1 #schrittweite
t = 0 #startzeitpunkt
x0 = 2 #startwert für x
y0 = 0  #startwert für y
tdata = np.array([])
xdata = np.array([])
tdata = np.append(tdata,t)
xdata = np.append(xdata,x0)

while t<T:
    k1 = dt * f(t, [x0, y0])
    k2 = dt * f(t+dt/2, [x0 + k1[0]/2, y0+k1[1]/2])
    k3 = dt * f(t+dt/2, [x0+k2[0]/2, y0+k2[1]/2])
    k4 = dt * f(t+dt, [x0+k3[0], y0+k3[1]])
    x0 += (k1[0]+ 2*k2[0] +2*k3[0]+k4[0])/6
    y0 += (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    t += dt
    tdata = np.append(tdata, t)
    xdata = np.append(xdata, x0)

plt.plot(tdata, xdata, label='echter Verlauf')
plt.xlabel('Zeit in s')
plt.ylabel('Auslenkung in rad')

#Verlauf mit RNN vorhersagen

model = load_model('FFNN_Einfachpendel.h5')

x0 = 1.9122742
xm1 = 1.9779199
xm2 = 2
trnn = np.array([])
xrnn = np.array([])
i = 2*dt
while i<T:
    xrnn = np.append(xrnn, x0)
    trnn = np.append(trnn, i)
    input_data = np.array([[x0, xm1, xm2]])
    xplus1 = model.predict(input_data)
    x0, xm1, xm2 = xplus1[0][0], x0, xm1
    i = i+dt

plt.plot(trnn, xrnn, label='vorhergesagter Verlauf')
plt.legend(loc='lower right')
plt.title('Anlernen mit x0 im Bereich 1.9999 & 2.0001')
plt.ylim([-2,2])
plt.show()
