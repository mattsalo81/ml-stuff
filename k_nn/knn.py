#!/home/syrup/anaconda3/envs/tensorflow/bin/python

import nn_table as knn
import csv_reader as csv
from sys import argv

script, k, rate, distance_type = argv
k = int(k)
rate = int(rate)


data_file = "./seeds.txt"
num_inputs = 7 
num_outputs = 1
output_col = 7
# k=3
# rate = 3
# distance_type="euclidean"
vote_type="classify"


labels, all_x, all_y = csv.read_from_file(data_file, output_col)
trn_x, trn_y, tst_x, tst_y = csv.get_training_and_test_sets(all_x, all_y, rate)

knn_table = knn.KNNTable(num_inputs, num_outputs)
for i in range(len(trn_x)):
    knn_table.insert_record(trn_x[i], trn_y[i])

success = 0
for i in range(len(tst_x)):
    print(f"Testing record {tst_x[i]}")
    output = knn_table.predict_value_from_k_nearest_neighbors(
                                            tst_x[i],
                                            k=k,
                                            vote_type=vote_type,
                                            distance_type=distance_type)
    print(f"\tShould be {tst_y[i]} and predicted {output}")
    if tst_y[i] == output[0]:
        print("\tPASS")
        success += 1
    else:
        print("\tFAIL")

success_rate = 100 * success / len(tst_x)

print(f"\n\nWe got {success_rate}% of things right\n\n")

