#!/home/syrup/anaconda3/envs/tensorflow/bin/python

import csv_reader as csv
import numpy as np

# main guard because I don't usually define helper functions in the same
# scope
def main():
    input_file = "./seeds.csv"
    input_dim = 7
    hidden_dim = 4
    output_dim = 3
    epochs = 20000
    one_in_x_test = 4


    labels, all_x, all_y_unf = csv.read_from_file(input_file, input_dim)
    # convert all_y into correctly formatted array
    all_y =[]
    for y in all_y_unf:
        new_row = [0] * output_dim
        new_row[y-1] = 1
        all_y.append(new_row)

    # split into training set and test set
    train_x, train_y, test_x, test_y = csv.get_training_and_test_sets(
                                            all_x, all_y, one_in_x_test)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # make the model

    w1 = 2 * np.random.rand(input_dim, hidden_dim) - 1
    b1 = np.zeros(hidden_dim)
    w2 = 2 * np.random.rand(hidden_dim, output_dim) - 1
    b2 = np.zeros(output_dim)

    for epoch in range(epochs):
        

main()

def loss()

