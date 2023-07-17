# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,



data = pd.read_excel('default_of_credit_card_clients.xls', header = [0,1])

features = list(data.columns[1:24])
X = data[features].values
labels = list(data.columns[24:])
y = data[labels].values
y = y.ravel()


#scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(np.sum(y), len(y), np.sum(y) / len(y))


dataset_name = 'default'
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

