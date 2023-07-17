# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing,


data = pd.read_csv('german.data-numeric.csv', header=None)
data = np.array(data)

X = np.array([x[0].rsplit(';')[1:25] for x in data])
y = np.array([x[0].rsplit(';')[25:-1] for x in data])
y = y.ravel()


X = np.asarray(X, dtype=int)

y = [int(l) - 1 for l in y]



#scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(np.sum(y), len(y))


dataset_name = 'german'
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

