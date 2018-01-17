#!/home/syrup/anaconda3/envs/tensorflow/bin/python

import matt_table

input_file = "./indians.txt"
output_attr = 8
min_gain = .095

table = matt_table.MattTable()
table.populate_from_file(input_file)

table.make_decision_tree(output_attr, min_gain)
