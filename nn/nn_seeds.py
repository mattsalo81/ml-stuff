#!/home/syrup/anaconda3/envs/tensorflow/bin/python

import csv_reader as csv
import numpy as np

# main guard because I don't usually define functions in the same scope
def main():
    input_file = "./seeds.csv"
    input_dim = 7
    hidden_dim = 4
    output_dim = 3
    epochs = 20000
    one_in_x_test = 4


    labels, all_x, all_y = csv.read_from_file(input_file, input_dim)
    # convert all_y into correctly formatted array
    #all_y =[]
    #for y in all_y_unf:
    #    new_row = [0] * output_dim
    #    new_row[y-1] = 1
    #    all_y.append(new_row)

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

def predict_scores(x, w1, b1, w2, b2):
    h1 = x.dot(w1) + b1
    a1 = np.tanh(h1)
    h2 = h1.dot(w2) + b2
    y = np.tanh(h2)
    return h1, a1, h2, y

def soft_max(array_like):
    exp_arr = np.exp(array_like)
    return exp_arr / np.sum(exp_arr, axis=1, keepdims=True)

def calc_loss(x, y, w1, b2, w2, b2):
    h1, a1, h2, y_pred = soft_max(predict_scores(x, w1, b1, w2, b2))
    # you need to get the probability of the correct class only
    # use adv slicing
    correct_prob = y_pred[range(len(x)), y]
    data_loss = np.sum(-np.log(correct_prob)) / len(x)
    return data_loss

def back_prop(x, y, w1, b1, w2, b2):
    # calculate output weights for the input
    h1, a1, h2, predicted_scores = predict_scores(x, w1, b1, w2, b2)
    predicted_probability = soft_max(predicted_scores)
    # massage y into same format of predicted output
    correct_probability = np.zeros([len(y), 2])
    correct_probability[range(len(y)), y] += 1
    # get output error as difference between probability and target
    output_error = predicted_probability - correct_probability
    # back_prop doesn't have to be calculus, can just be differences.
    # the change in second layer weights is the sum of all its inputs times
    # its respective error
    dw2 = (a2.T).dot(output_error)
    # change in activation threshold = average error for that neuron
    db2 = np.sum(output_error, axis=0, keepdims=True)
    output_error_layer_1 = ?

