import numpy as np
import csv

num_samples = 10000;
x_train = np.random.uniform(0, 10, num_samples)
y_train = x_train**2+4*x_train+4

with open('inputdataeinfachesrnn.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(x_train.reshape(-1,1))

with open('outputdataeinfachesrnn.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(y_train.reshape(-1, 1))
