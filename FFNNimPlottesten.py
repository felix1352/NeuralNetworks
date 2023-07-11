import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.stats import pearsonr


#Test im Plot vergleichen

#Variablen deklarieren
l = 2
b = 0.7
g = 9.81
T = 10 #simulation Time
def f(t, x, M):
    dxdt = x[1]
    dxdtdt = 1/l*(-b*x[1]-g*np.sin(x[0])+M)
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
tM = np.arange(0, T+dt, dt)
M = np.sin(tM)
counter = 0

while t<T:
    k1 = dt * f(t, [x0, y0], M[counter])
    k2 = dt * f(t+dt/2, [x0 + k1[0]/2, y0+k1[1]/2], M[counter])
    k3 = dt * f(t+dt/2, [x0+k2[0]/2, y0+k2[1]/2], M[counter])
    k4 = dt * f(t+dt, [x0+k3[0], y0+k3[1]], M[counter])
    x0 += (k1[0]+ 2*k2[0] +2*k3[0]+k4[0])/6
    y0 += (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    t += dt
    tdata = np.append(tdata, t)
    xdata = np.append(xdata, x0)
    counter += 1

plt.subplot(1,2,1)
plt.plot(tdata, xdata, label='echter Verlauf')
plt.xlabel('Zeit in s')
plt.ylabel('Auslenkung in rad')

#Verlauf mit RNN vorhersagen

model = load_model('FFNN_Einfachpendel.h5')

x0 = xdata[2]
xm1 = xdata[1]
xm2 = xdata[0]
tffnn = np.array([])
xffnn = np.array([])
i = 2*dt
counter = 2
while i<T:
    xffnn = np.append(xffnn, x0)
    tffnn = np.append(tffnn, i)
    input_data = np.array([[x0, xm1, xm2, M[counter]]])
    xplus1 = model.predict(input_data)
    x0, xm1, xm2 = xplus1[0][0], x0, xm1
    i = i+dt
    counter += 1

plt.plot(tffnn, xffnn, label='vorhergesagter Verlauf')
plt.legend(loc='lower right')
plt.title('Anlernen mit x0 im Bereich 0 bis pi')
plt.ylim([-2,2])
plt.subplot(1,2,2)
plt.plot(tffnn, abs(xffnn-xdata[2:101]))
plt.title('Absoluter Fehler')
plt.show()

correlation, _ = pearsonr(xffnn, xdata[2:101])
print(correlation)
