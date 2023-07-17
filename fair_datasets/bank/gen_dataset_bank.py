# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing,

bank_data = pd.read_csv('bank-full.csv')
colunm = list(bank_data.columns[0:])
bank_data = bank_data[colunm].values


X = np.array([x[0].rsplit(';')[:-1] for x in bank_data])
y = np.array([x[0].rsplit(';')[-1] for x in bank_data])


from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(X)
X = enc.transform(X)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y = le.transform(y)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)                #使用fit_transform(data)一步达成结果

    


dataset_name = 'bank'
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
