import csv
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

src = "Train_transform"    # new format
dat_ref = np.array(list(csv.reader(open("../data/reference_all.csv", 'r'))))
# dat_ref = np.unique(dat_ref, axis=1)
print(dat_ref.shape)

def normalizing(input_pc, dat_ref):
    global num_col
    global num_row

    # input_pc = np.vstack((np.array(input_pc[2:7]), np.array(input_pc[8:]))) #
    temp_ind = []

    for a in range(len(dat_ref[0])):
        index_num = np.where(dat_ref[0, a] == input_pc[0, :])[0][0]
        temp_ind.append(index_num)
    sample = input_pc[:, temp_ind]
    # sample_val = sample[5:].astype('float32')
    sample_val = sample[1:].astype('float32')
    num_col = sample.shape[1]
    num_row = sample_val.shape[0]
    return sample_val

# ##
# cases = []
# for root, dirs, files in os.walk("../data/" + src + "/"):
#     files.sort()
#     for i, fname in enumerate(files):
#         print(i, fname)
#         temp = open(os.path.join(root, fname), 'r', encoding='ISO-8859-1')
#         cases.append(list(csv.reader(temp))[:788])
#         temp.close()
#
# norm_cases = []
# for case in cases:
#     temp = normalizing(case, dat_ref)
#     norm_cases.append(temp[600:780])
# ##

cases = []
for root, dirs, files in os.walk("../data/" + src + "/"):
    files.sort()
    for i, fname in enumerate(files):
        print(i, fname)
        df = pd.read_csv("../data/" + src + "/" +fname, index_col=0, nrows=180)  # read file
        df = df.replace('.*', 0, regex=True).fillna(0)  # character and NA to zero value
        df = df.iloc[0:180]  # event time after row read
        cases.append(df)
        # temp.delete()

norm_cases = []
for case in cases:
    case_ind = np.array(list(case))
    case = case.to_numpy()
    case = np.vstack([case_ind, case])
    temp = normalizing(case, dat_ref)
    norm_cases.append(temp[:])

norm_cases = np.array(norm_cases)
print(norm_cases.shape)

total = norm_cases.reshape(-1, num_col)
new_ref = dat_ref
eq = [] #
for i in range(0, num_col):
    mini = min(total[:, i])
    maxi = max(total[:, i])
    # if mini == maxi or maxi < mini + 0.0001:
    if mini == maxi :
        eq.append(i)
        maxi = maxi + 1
    new_ref[3][i] = mini
    new_ref[4][i] = maxi
new_ref = np.array(new_ref)
eq = np.array(eq)
print("equal :", eq, eq.shape)
new_ref = np.delete(new_ref, eq, axis=1) #
print("new_ref : ", new_ref.shape)

with open("../data/reference_" + str(new_ref.shape[1]) + ".csv", 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(new_ref)
    f.close()