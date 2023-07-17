1# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,



student_data = pd.read_csv('student-por.csv')

colunm = list(student_data.columns[0:])
student_data = student_data[colunm].values

X = np.array([x[0].rsplit(';')[:30] for x in student_data])
y = np.array([x[0].rsplit(';')[32:] for x in student_data])
y = y.ravel()



from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X[:, [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]] = enc.fit_transform(X[:, [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]])

y = [int(label) for label in y]
for i in range(len(y)):
    if y[i] < 11:
        y[i] = 0
    elif y[i] < 14:
        y[i] = 1
    elif y[i] < 20:
        y[i] = 2
        
    
print(len([y_ for y_ in y if y_ == 0]))
print(len([y_ for y_ in y if y_ == 1]))
print(len([y_ for y_ in y if y_ == 2]))

# scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


dataset_name = 'student'
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

