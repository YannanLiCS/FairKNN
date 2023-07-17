# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,


data = pd.read_csv('propublicaCompassRecividism_data_fairml/propublica_data_for_fairml.csv')


features = list(data.columns[1:])
ori_X = data[features].values
labels = list(data.columns[:1])
y = data[labels].values
y = y.ravel()



X = []
#process X
for ori_x in ori_X:
    x = [0 for i in range(6)]
    x[0:2] = ori_x[0:2]
    # x[2] : age
    if ori_x[2] == 1:
        x[2] = 2
    elif ori_x[3] == 1:
        x[2] = 0
    else:
        x[2] = 1
    
    # x[3] :race
    if ori_x[4] == 1:
        x[3] = 0
    elif ori_x[5] == 1:
        x[3] = 0
    elif ori_x[6] == 1:
        x[3] = 0
    elif ori_x[7] == 1:
        x[3] = 0
    elif ori_x[8] == 1:
        x[3] = 1
    
    x[-2:] = ori_x[-2:]
    X.append(x)



print(X[:10], y[:10])

#scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(np.sum(y), len(y))
print(X[:10], y[:10])


dataset_name = 'compas'
test_percent = 10
from sklearn.model_selection import train_test_split

import numpy as np
filename = '../' + dataset_name + '-' + str(test_percent) + '.npy'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_percent/100.0)
with open(filename, 'wb') as f:
    np.save(f, X_train)
    np.save(f, y_train)
    np.save(f, X_test)
    np.save(f, y_test)
    
print(len(X_train), len(X_test), len(X_train[0]))
for x in X_test[:101]:
    if x[3] > 0 and x[4] > 0:
        print("contains white woman in first 101 datasets")
        
    if x[3] > 0 and x[4] < 0:
        print("contains white man in first 101 datasets")

