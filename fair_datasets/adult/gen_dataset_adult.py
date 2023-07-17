# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,

'''
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
]
'''

adult_data = pd.read_csv('adult.data', header=None)


features = list(adult_data.columns[:14])
X = adult_data[features].values
labels = list(adult_data.columns[14:])
y = adult_data[labels].values
y = y.ravel()

print(X[:10])

for i in range(len(X)):
    if 'White' in str(X[i][8]):
        X[i][8] = 0
    else:
        X[i][8] = 1


# encode
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X[:, [1, 3, 5, 6, 7, 9, 13]] = enc.fit_transform(X[:, [1, 3, 5, 6, 7, 9, 13]])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y = le.fit_transform(y)



#scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(np.sum(y), len(y))

print(X[:10])

'''
dataset_name = 'adult'
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
'''
