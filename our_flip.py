import random
import numpy as np
import datetime
import ast
import itertools as it
from collections import Counter
# to calculate neighbors quickly
from sklearn.neighbors import NearestNeighbors

from z3 import *

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


# Note: for labels with same cnt, we choose the smaller one
    
def getLabelCnt(nabrs_indices, y_train):
    targets = [y_train[i] for i in nabrs_indices]
    return Counter(targets)
    
    
    
def getFreqLabel(cntr):
    [ori_label, ori_cnt] = cntr.most_common(1)[0]
    if len(cntr.most_common(2)) < 2:
        return ori_label
    for label, cnt in cntr.items():
        if label != ori_label and cnt == ori_cnt:
            ori_label = min(ori_label, label)
    return ori_label
    
    
def getFreqLabelExcept(cntr, omit_label):
    [ori_label, ori_cnt] = cntr.most_common(1)[0]
    if len(cntr.most_common(2)) < 2:
        #return another label with minimal id
        if cntr.most_common(2)[0][0] > 0:
            return 0
        else:
            return 1
    freq_label = -1
    freq_label_cnt = 0
    for label, cnt in cntr.items():
        if label != omit_label and (freq_label == -1 or cnt + (label < freq_label) > freq_label_cnt):
            freq_label = label
            freq_label_cnt = cnt
    return freq_label
    

def predict_with_label(nabrs, correct_label, y_train, k, max_flips, classes_num):

    label_cntr = getLabelCnt(nabrs, y_train)
    
    freq_label = getFreqLabel(label_cntr)
    if freq_label == correct_label:
            best_predict = correct_label
    else:
        freq_wrong_label = getFreqLabelExcept(label_cntr, correct_label)
        if label_cntr[freq_wrong_label] - label_cntr[correct_label] + (freq_wrong_label < correct_label) > max_flips * 2 or sum(label_cntr.values()) - classes_num * label_cntr[correct_label] - (classes_num - correct_label - 1) > max_flips * (classes_num + 1):
            best_predict = -1
        else:
            # calculate best label (as correct as possible)
            Flips = IntVector('Flips', classes_num)
            s = Solver()
            s.add(Flips[correct_label] == 0)
            for label in range(correct_label):
                s.add(label_cntr[label] - Flips[label] < label_cntr[correct_label] + Sum(Flips))
                s.add(label_cntr[label] >= Flips[label])
            for label in range(correct_label+1, classes_num):
                s.add(label_cntr[label] - Flips[label] <= label_cntr[correct_label] + Sum(Flips))
                s.add(label_cntr[label] >= Flips[label])
            s.add(Sum(Flips) <= max_flips)
            if s.check() != unsat:
                best_predict = correct_label
            else:
                # return any wrong label
                best_predict = -1
            
    # calculate worst label (as incorrect as possible)
    freq_wrong_label = getFreqLabelExcept(label_cntr, correct_label)
    label_cntr[correct_label] -= max_flips
    label_cntr[freq_wrong_label] +=  max_flips
    worst_predict = getFreqLabel(label_cntr)
    
    return [worst_predict, best_predict]
    
    


import statistics
# perform 10-fold cross validation
def over_cross_validation(XTrain, yTrain, fold_num, kset, down_sampling_paras, max_flips, classes_num):
    
    # split train data into $fold_num$ fold
    from sklearn.model_selection import KFold
    folds = KFold(fold_num)
    
    err_lb_dict = {k:0.0 for k in kset}
    err_ub_dict = {k:0.0 for k in kset}

    for ti, vi in folds.split(XTrain, yTrain):
        XT = XTrain[ti]
        XV = XTrain[vi]
        yT = yTrain[ti]
        yV = yTrain[vi]
        
        max_k = np.max(kset)
        neigh = NearestNeighbors(n_neighbors = max_k)
        neigh.fit(XT)
        
        err_lb_cnt_dict = {k:0 for k in kset}
        err_ub_cnt_dict = {k:0 for k in kset}
        weighted_total_cnt = 0
  
        for i in range(XV.shape[0]):
            nabrs_over = neigh.kneighbors([XV[i]], return_distance=False)[0]
            for k in kset:
                [worst_predict, best_predict] = predict_with_label(nabrs_over[0:k], yV[i], yT, k, max_flips, classes_num)
                if yV[i] != best_predict:
                    err_lb_cnt_dict[k] += down_sampling_paras[yV[i]]
                if yV[i] != worst_predict:
                    err_ub_cnt_dict[k] += down_sampling_paras[yV[i]]
            weighted_total_cnt += down_sampling_paras[yV[i]]
            
        for k in kset:
            err_lb_dict[k] += err_lb_cnt_dict[k] / weighted_total_cnt
            err_ub_dict[k] += err_ub_cnt_dict[k] / weighted_total_cnt
    
    for k in kset:
        err_lb_dict[k] /= fold_num
        err_ub_dict[k] /= fold_num
    
    return [err_lb_dict, err_ub_dict]
    
    
def getOptKSet(XTrain, yTrain, cross_validation_fold, kset, down_sampling_paras, max_flips, classes_num):
    
    [err_lb_dict, err_ub_dict] = over_cross_validation(XTrain, yTrain, cross_validation_fold, kset, down_sampling_paras, max_flips, classes_num)
    
    opt_Ks = []
    import operator
    min_upbound = min(err_ub_dict.items(), key=operator.itemgetter(1))[1]
    
    for k, lb in err_lb_dict.items():
        if (lb <= min_upbound):
            opt_Ks.append(k)
    
    return opt_Ks
    

# ---- original KNN
def predict(neigh, y_train, x_test):
    nabr_indices = neigh.kneighbors([x_test], return_distance=False)[0]
    targets = [y_train[index] for index in nabr_indices]
    counter = Counter(targets)
    return getFreqLabel(counter)

import statistics
# perform 10-fold cross validation
def original_cross_validation(XTrain, yTrain, fold_num, k, down_sampling_paras):

    # split train data into 10 fold
    # we guarantee the 10 folds keep the same
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

def original_getOptimalK(XTrain, yTrain, cross_validation_fold, kset, down_sampling_paras):
    mse = []
    for k in kset:
        mse.append(original_cross_validation(XTrain, yTrain, cross_validation_fold, k, down_sampling_paras))
    # return best k
    return kset[mse.index(min(mse))]
    
    
def getSafeSet(k, k_step, k_min, k_max, safe_cnt):
    bias_step = safe_cnt // k_step
    if bias_step <= 0:
        return {k}
    safe_k_lb = max(k_min, k - bias_step * k_step)
    safe_k_ub = min(k_max, k + bias_step * k_step)
    return set(range(safe_k_lb, safe_k_ub + k_step, k_step))
    

def isUnfair(opt_Ks, neigh, y_train, _test, predicted_label, max_flips):
    for k in opt_Ks:
        indicies = neigh.kneighbors([x_test], return_distance=False)[0][:k]
        targets = [y_train[i] for i in indicies]
        counter = Counter(targets)
        mst_freq_different_label = getFreqLabelExcept(counter, predicted_label)
        counter[predicted_label] -= max_flips
        counter[mst_freq_different_label] += max_flips
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
def getParameters():
    if len(sys.argv) < 5:
        print("args = (filename, down_sampling_paras, cross_validation_fold, kset, max_flip)")
        exit()
    filename = sys.argv[1]
    down_sampling_paras = ast.literal_eval(sys.argv[2])
    cross_validation_fold = int(sys.argv[3])
    kset = ast.literal_eval(sys.argv[4])
    max_flips = int(sys.argv[5])
    with open(filename, 'rb') as f:
        XTrain = np.load(f)
        yTrain = np.load(f)
        X_test = np.load(f)
        y_test = np.load(f)
    
    return XTrain, yTrain, X_test, y_test, down_sampling_paras, cross_validation_fold, kset, max_flips


if __name__ == "__main__":
    XTrain, yTrain, X_test, y_test, down_sampling_paras, cross_validation_fold, kset, max_flips = getParameters()
    classes_num = np.max(yTrain) + 1
    ori_train_cnt = len(yTrain)

    # down sampling to make the training set balanced
    down_x_train, down_y_train = down_sampling(XTrain, yTrain, down_sampling_paras)
    down_train_cnt = len(down_y_train)
     
    # calculate default prediction labels
    ori_K = original_getOptimalK(down_x_train, down_y_train, cross_validation_fold, kset, down_sampling_paras)
    neigh = NearestNeighbors(n_neighbors = ori_K)
    neigh.fit(down_x_train)

    X_test = X_test[:101]
    y_test = y_test[:101]
    test_cnt = len(X_test)
    predictions = [predict(neigh, down_y_train, x) for x in X_test]
    report = classification_report(y_test, predictions)

    # certify fairness
    starttime = datetime.datetime.now()
    if max_flips < down_train_cnt / cross_validation_fold:
        opt_Ks = getOptKSet(down_x_train, down_y_train, cross_validation_fold, kset, down_sampling_paras, max_flips, classes_num)
    else:
        opt_Ks = kset

    # overapproximation on prediction results
    unfair_indicies = []
    neigh = NearestNeighbors(n_neighbors = opt_Ks[-1])
    neigh.fit(down_x_train)
    for i, x_test in enumerate(X_test):
        if isUnfair(opt_Ks, neigh, down_y_train, x_test, predictions[i], max_flips) == True:
            unfair_indicies.append(i)

    endtime = datetime.datetime.now()
    #print("#train/#cnt:", ori_train_cnt, "/", test_cnt, "max_flip = ", max_flips, ",fold_num =", cross_validation_fold)
    #print("original optimal K = ", ori_K)
    #print(report)
    #print("optimal Ks during flipping:", len(opt_Ks), opt_Ks)
    print("fair = {:.1%}".format(1 - len(unfair_indicies) / test_cnt))
    total_time_seconds = (endtime - starttime).seconds
    print("#test data = ", test_cnt, "total time = {:.1f}s, avg time = {:.1f}s".format(total_time_seconds, total_time_seconds / test_cnt))
    #print("unfair_indicies =", unfair_indicies)

