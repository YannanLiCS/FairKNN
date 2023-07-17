import datetime
import math
import ast
import numpy as np
import itertools as it
from collections import Counter
# to calculate neighbors quickly
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report


def getFreqLabel(cntr):
    [ori_label, ori_cnt] = cntr.most_common(1)[0]
    if len(cntr.most_common(2)) < 2:
        return ori_label
    for label, cnt in cntr.items():
        if label != ori_label and cnt == ori_cnt:
            ori_label = min(ori_label, label)
    return ori_label
    

def predict(neigh, y_train, x_test):
    nabr_indices = neigh.kneighbors([x_test], return_distance=False)[0]
    targets = [y_train[index] for index in nabr_indices]
    counter = Counter(targets)
    return getFreqLabel(counter)
    


import statistics
# perform 10-fold cross validation
def cross_validation(XTrain, yTrain, k, down_sampling_paras):

    # split train data into 10 fold
    # we guarantee the 10 folds keep the same
    fold_num = 10
    from sklearn.model_selection import KFold
    folds = KFold(fold_num)
    
    errs = []

    for ti, vi in folds.split(XTrain, yTrain):

        XT = XTrain[ti]
        XV = XTrain[vi]
        yT = yTrain[ti]
        yV = yTrain[vi]
                
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(XT)
        
        err_cnt = 0
        weighted_total_cnt = 0
  
        for i in range(XV.shape[0]):
            if yV[i] != predict(neigh, yT, XV[i]):
                err_cnt += down_sampling_paras[yV[i]]
            weighted_total_cnt += down_sampling_paras[yV[i]]

        errs.append(err_cnt / weighted_total_cnt)

    return statistics.mean(errs)
    
    
def getOptimalK(XTrain, yTrain, kset, down_sampling_paras):
    mse = []
    for k in kset:
        mse.append(cross_validation(XTrain, yTrain, k, down_sampling_paras))
    # return best k
    return kset[mse.index(min(mse))]
    
    
def prediction_change(K, x_train, y_train, x_test, predicted_label, protected_indicies, protected_combinations, epsilon):
    new_x_test = np.copy(x_test)
    for comb in protected_combinations:
        for i, att in enumerate(protected_indicies):
            new_x_test[att] = comb[i]
        dist_lbs = [np.sum(np.square(np.maximum(abs(new_x_test - t) - epsilon, np.zeros(len(new_x_test))))) for t in x_train]
        dist_ubs = [np.sum(np.square(abs(new_x_test - t) + epsilon)) for t in x_train]
        #print(dist_lbs[:10], dist_ubs[:10])
        Kth_ub = np.partition(np.array(dist_ubs), K - 1)[K - 1]
        targets = [y_train[index] for index in range(len(x_train)) if dist_lbs[index] <= Kth_ub]
        counter = Counter(targets)
        if len(targets) == 10:
            print([index for index in range(len(x_train)) if dist_lbs[index] <= Kth_ub])
        counter[predicted_label] -= len(targets) - K
        if getFreqLabel(counter) != predicted_label:
            return True
    return False
    

def down_sampling(XTrain, yTrain, down_sampling_paras):
    down_x_train = []
    down_y_train = []
    class_count = {}
    for i in range(len(XTrain)):
        label = yTrain[i]
        if label not in class_count.keys():
            class_count[label] = 0
        else:
            class_count[label] += 1
        if class_count[label] % down_sampling_paras[label] == 0:
            down_x_train.append(XTrain[i])
            down_y_train.append(yTrain[i])
    return np.array(down_x_train), np.array(down_y_train)
    
# =========================
# load train and test set
import sys
import numpy as np
if len(sys.argv) < 6:
    print("args = (filename, down_sampling_paras, protected_indicies, kset, pertubation, pertubation_indicies)")
    exit()
filename = sys.argv[1]
down_sampling_paras = ast.literal_eval(sys.argv[2])
protected_indicies = ast.literal_eval(sys.argv[3])
kset = ast.literal_eval(sys.argv[4])
pertubation = float(sys.argv[5])
pertubation_indicies = ast.literal_eval(sys.argv[6])
with open(filename, 'rb') as f:
    XTrain = np.load(f)
    yTrain = np.load(f)
    X_test = np.load(f)
    y_test = np.load(f)

test_cnt = X_test.shape[0]
train_cnt = XTrain.shape[0]


# down sampling to make the training set balanced
down_x_train, down_y_train = down_sampling(XTrain, yTrain, down_sampling_paras)

'''
# debug print
print(np.sum(yTrain) / len(yTrain))
print(np.sum(down_y_train) / len(down_y_train))
print(len(down_y_train))
print(K)
print(report)
'''

# calculate the epsilon
attribute_max = np.max(down_x_train, axis = 0)
attribute_min = np.min(down_x_train, axis = 0)
epsilon = pertubation * (attribute_max - attribute_min)
for i in range(len(epsilon)):
    if i not in pertubation_indicies:
        epsilon[i] = 0


# tune the hyperparameter K (#neighbors) with Cross Validation
K = getOptimalK(down_x_train, down_y_train, kset, down_sampling_paras)
neigh = NearestNeighbors(n_neighbors = K)
neigh.fit(down_x_train)
predictions = [predict(neigh, down_y_train, x) for x in X_test]
report = classification_report(y_test, predictions)


# certify epsilon-fairness
# calculate all the possible situations
loop_val = []
for index in protected_indicies:
    one_attribute_vals = list(set(x[index] for x in down_x_train))
    loop_val.append(one_attribute_vals)
protected_combinations_iter = it.product(*loop_val)
protected_combinations = [comb for comb in protected_combinations_iter]


starttime = datetime.datetime.now()
unfair = [prediction_change(K, down_x_train, down_y_train, x_test, predictions[i], protected_indicies, protected_combinations, epsilon) for i, x_test in enumerate(X_test[:101])]
endtime = datetime.datetime.now()
print("epsilon =", epsilon)
print("optimal K = ", K)
print(report)
print("fair = {:.4%}".format(1 - np.sum(unfair) / len(unfair)))
total_time_seconds = (endtime - starttime).seconds
print("#test data = ", len(unfair), "total time =", total_time_seconds, "s, avg time = ", total_time_seconds / len(unfair), "s")


