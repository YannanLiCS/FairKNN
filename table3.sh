#!/bin/bash
echo ======= salary, gender =======
echo ---- label-fipping ---
python3 our_flip.py fair_datasets/salary-10.npy "{0:1, 1:1}" 10 "[7,8,9,10,11]" 1
echo ----------------------
echo ---- label-fipping+individual ---
python3 our_all.py fair_datasets/salary-10.npy "{0:1, 1:1}" 10 "[7,8,9,10,11]" "[0]" 0.0 "[]" 1
echo ----------------------
echo ---- label-fipping+individual+epsilon ---
python3 our_all.py fair_datasets/salary-10.npy "{0:1, 1:1}" 10 "[7,8,9,10,11]" "[0]" 0.01 "[29]" 1
echo ----------------------
echo
echo ======= student, gender =======
echo ---- label-fipping ---
python3 our_flip.py fair_datasets/student-10.npy "{0:1, 1:1, 2:1}" 10 "[30,31,32,33,34]" 1
echo ----------------------
echo ---- label-fipping+individual ---
python3 our_all.py fair_datasets/student-10.npy "{0:1, 1:1, 2:1}" 10 "[30,31,32,33,34]" "[1]" 0.0 "[]" 1
echo ----------------------
echo ---- label-fipping + individual+epsilon ---
python3 our_all.py fair_datasets/student-10.npy "{0:1, 1:1, 2:1}" 10 "[30, 31, 32, 33, 34]" "[1]"  0.01 "[29]" 1
echo ----------------------
echo
echo ======= german, gender =======
echo -------- label-fipping --------
python3 our_flip.py fair_datasets/german-10.npy "{0:2, 1:1}" 10 "[190, 200, 210, 220, 230]" 10
echo ----------------------
echo -------- label-fipping+individual --------
python3 our_all.py fair_datasets/german-10.npy "{0:2, 1:1}" 10 "[190, 200, 210, 220, 230]" "[6]" 0.0 "[4]" 10
echo ----------------------
echo -------- label-fipping+individual+epsilon --------
python3 our_all.py fair_datasets/german-10.npy "{0:2, 1:1}" 10 "[190, 200, 210, 220, 230]" "[6]" 0.01 "[4]" 10
echo ----------------------
echo
echo ======= compas, race =======
echo -------- label-fipping --------
python3 our_flip.py fair_datasets/compas-10.npy "{0:1, 1:1}" 10 "[570,580,590,600,610]" 10
echo ----------------------
echo -------- label-fipping+individual --------
python3 our_all.py fair_datasets/compas-10.npy "{0:1, 1:1}" 10 "[570,580,590,600,610]" "[3]" 0.0 "[0]" 10
echo ----------------------
echo -------- label-fipping+individual --------
python3 our_all.py fair_datasets/compas-10.npy "{0:1, 1:1}" 10 "[570,580,590,600,610]" "[3]" 0.01 "[0]" 10
echo ----------------------
echo
echo ======= compas, gender =======
echo -------- label-fipping --------
python3 our_flip.py fair_datasets/compas-10.npy "{0:1, 1:1}" 10 "[570,580,590,600,610]" 10
echo ----------------------
echo -------- label-fipping+individual --------
python3 our_all.py fair_datasets/compas-10.npy "{0:1, 1:1}" 10 "[570,580,590,600,610]" "[4]" 0.0 "[0]" 10
echo ----------------------
echo -------- label-fipping+individual --------
python3 our_all.py fair_datasets/compas-10.npy "{0:1, 1:1}" 10 "[570,580,590,600,610]" "[4]" 0.01 "[0]" 10
echo ----------------------
echo
echo ======= Default, gender =======
echo -------- label-fipping --------
python3 our_flip.py fair_datasets/default-10.npy "{0:3, 1:1}" 10 "[520,530,540,550,560]" 50
echo ----------------------
echo -------- label-fipping+individual --------
python3 our_all.py fair_datasets/default-10.npy "{0:3, 1:1}" 10 "[520,530,540,550,560]" "[1]" 0.0 "[11]" 50
echo ----------------------
echo -------- label-fipping+individual+epsilon --------
python3 our_all.py fair_datasets/default-10.npy "{0:3, 1:1}" 10 "[520,530,540,550,560]" "[1]" 0.01 "[11]" 50
echo ----------------------
echo
echo ======= adult, race =======
echo -------- label-fipping --------
python3 our_flip.py fair_datasets/adult-10.npy "{0:3, 1:1}" 10 "[470,480,490,500,510]" 50
echo  ----------------------
echo  -------- adult, bias+race --------
python3 our_all.py fair_datasets/adult-10.npy "{0:3, 1:1}" 10 "[470,480,490,500,510]" "[8]" 0.0 "[2]" 50
echo  ----------------------
echo  -------- adult, bias+race+epsilon --------
python3 our_all.py fair_datasets/adult-10.npy "{0:3, 1:1}" 10 "[470,480,490,500,510]" "[8]" 0.01 "[2]" 50
echo  ----------------------
echo
echo ======= adult, gender =======
echo -------- label-fipping --------
python3 our_flip.py fair_datasets/adult-10.npy "{0:3, 1:1}" 10 "[470,480,490,500,510]" 50
echo  ----------------------
echo  -------- adult, bias+gender --------
python3 our_all.py fair_datasets/adult-10.npy "{0:3, 1:1}" 10 "[470,480,490,500,510]" "[9]" 0.0 "[2]" 50
echo  ----------------------
echo  -------- adult, bias+race+epsilon --------
python3 our_all.py fair_datasets/adult-10.npy "{0:3, 1:1}" 10 "[470,480,490,500,510]" "[9]" 0.01 "[2]" 50
echo  ----------------------

