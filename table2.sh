#!/bin/bash
echo ======= salary =======
echo ---- baseline, label-fipping ---
python3 baseline_flip.py fair_datasets/salary-10.npy "{0:1, 1:1}" 10 "[7,8,9,10,11]" 1
echo ----------------------
echo ---- our method, label-fipping ---
python3 our_flip.py fair_datasets/salary-10.npy "{0:1, 1:1}" 10 "[7,8,9,10,11]" 1
echo ----------------------
echo ---- baseline, label-fipping+individual ---
python3 baseline_flip_indiv.py fair_datasets/salary-10.npy "{0:1, 1:1}" 10 "[7,8,9,10,11]" "[0]" 1
echo ----------------------
echo ---- our method, label-fipping+individual ---
python3 our_all.py fair_datasets/salary-10.npy "{0:1, 1:1}" 10 "[7,8,9,10,11]" "[0]" 0.0 "[]" 1
echo ----------------------

echo ======= student =======
echo ---- baseline, label-fipping ---
python3 baseline_flip.py fair_datasets/student-10.npy "{0:1, 1:1, 2:1}" 10 "[30,31,32,33,34]" 1
echo ----------------------
echo ---- our method, label-fipping ---
python3 our_flip.py fair_datasets/student-10.npy "{0:1, 1:1, 2:1}" 10 "[30,31,32,33,34]" 1
echo ----------------------
echo ---- baseline, label-fipping + individual ---
python3 baseline_flip_indiv.py fair_datasets/student-10.npy "{0:1, 1:1, 2:1}" 10 "[30,31,32,33,34]" "[1]" 1
echo ----------------------
echo ---- our method, label-fipping+individual ---
python3 our_all.py fair_datasets/student-10.npy "{0:1, 1:1, 2:1}" 10 "[30,31,32,33,34]" "[1]" 0.0 "[]" 1
echo ----------------------

