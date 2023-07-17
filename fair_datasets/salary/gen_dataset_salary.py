# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,

'''
names = [
    'sex',
    'yr',
    'degree',
    'yd',
    'salary',
]
'''


salary_data = np.array(pd.read_csv('salary_character_codes.csv', dtype=object))

X = np.array([x[0].rsplit(';')[:4] for x in salary_data])
y = np.array([x[0].rsplit(';')[4:] for x in salary_data])
y = y.ravel()

# salary > 25K or <= 25K
for i in range(len(y)):
    if y[i] > '25000':
        y[i] = 1
    else:
        y[i] = 0
        


# encode
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X[:, [0, 1, 3]] = enc.fit_transform(X[:, [0, 1, 3]])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
    
print(np.sum(y), len(y))


dataset_name = 'salary'
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


