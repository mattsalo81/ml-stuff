#!/home/syrup/anaconda3/envs/tensorflow/bin/python

import csv_reader as csv
import numpy as np




# main guard because I don't usually define functions in the same scope
def main():
    input_file = "./seeds.csv"
    input_dim = 7
    hidden_dim = 5
    output_dim = 3
    epochs = 20000
    one_in_x_test = 4
    alpha = .01
    lam = .01


    labels, all_x, _all_y = csv.read_from_file(input_file, input_dim)

    all_x = normalize(all_x)
    print(all_x)
    
    # split into training set and test set
    train_x, _train_y, test_x, _test_y = csv.get_training_and_test_sets(
                                            all_x, _all_y, one_in_x_test)
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    _train_y = np.array(_train_y)
    _test_y = np.array(_test_y)
    train_y = np.zeros([len(_train_y), output_dim])
    test_y = np.zeros([len(_test_y), output_dim]) 
    # aight I don't really understand indexing yet, so this was done through 
    # mostly trial and error
    #   I want to add '1' to every row, but only in the column representing
    #   the classification, so I need so index the array at all those points.
    #   To do that, I need to create a matrix of coordinates. idk really how 
    train_y[range(len(_train_y))[:np.newaxis], _train_y - 1] += 1
    test_y[range(len(_test_y))[:np.newaxis], _test_y - 1] += 1

    # make the model

    w1 = 0.1 * (np.random.rand(input_dim, hidden_dim) - 0.5)
    b1 = np.zeros([1, hidden_dim])
    w2 = 0.1 * (np.random.rand(hidden_dim, output_dim) - 0.5)
    b2 = np.zeros([1, output_dim])

    for epoch in range(epochs):
        oldw1 = np.copy(w1)
        if (epoch % 1000 == 0):
            sloss = calc_loss(test_x, test_y, w1, b1, w2, b2)
            loss = calc_loss(train_x, train_y, w1, b1, w2, b2)
            print(f"Test Loss at epoch {epoch} is {sloss}")
            print(f"Train Loss at epoch {epoch} is {loss}")
        w1, b1, w2, b2 = back_prop(train_x, train_y, w1, b1,
                                    w2, b2, alpha, lam)

    loss = calc_loss(test_x, test_y, w1, b1, w2, b2)
    h1, a1, h2, y = predict_scores(test_x, w1, b1, w2, b2)
    print("Real")
    print(test_y)
    print("Predicted")
    print(y)
    print(f"final loss is {loss}")
    print(w1)
    print(b1)
    print(w2)
    print(b2)
        


def predict_scores(x, w1, b1, w2, b2):
    h1 = x.dot(w1) + b1
    a1 = np.tanh(h1)
    h2 = h1.dot(w2) + b2
    y = np.tanh(h2) 
    return h1, a1, h2, y

def soft_max(array_like):
    array_like = np.exp(array_like)
    return array_like / np.sum(array_like, axis=1, keepdims=True)

def calc_loss(x, y, w1, b1, w2, b2):
    h1, a1, h2, y_pred = predict_scores(x, w1, b1, w2, b2)
    y_pred = soft_max(y_pred)
    # you need to get the probability of the correct class only
    # use adv slicing (ugh just index multiply)
    log_prob = -np.log(y_pred)
    log_prob *= y
    data_loss = np.sum(log_prob) / len(x)
    return data_loss

def back_prop(x, y, w1, b1, w2, b2, alpha, lam):
    # calculate output weights for the input
    h1, a1, h2, predicted_scores = predict_scores(x, w1, b1, w2, b2)
    pred_prob = soft_max(predicted_scores)
    # output_error = 1/2(y_pred - y_targ) ** 2
    # derivative of error = y_pred - y_targ
    de3 = pred_prob - y 


    # the change in second layer weights is the sum of all its inputs times
    # its respective error
    # dw2/de3
    dw2 = (a1.T).dot(de3)
    # change in activation threshold = average error for that neuron
    db2 = np.sum(de3, axis=0, keepdims=True)
    # to calculate errors in the first layer, calculate how the output of
    # the layer relates to the error

    # could probably represent each layer as an object in a ll, feeing outputs
    # as inputs to next object and then backproping the other way
    #    I think that's what tensor flow does but with generalized graphs...
    de2 = de3.dot(w2.T) * (1 - np.power(a1, 2))
    dw1 = x.T.dot(de2)
    db1 = np.sum(de2, axis=0, keepdims=True)

    # regularization (L1) - bias weights towards zero - has the effect of 
    # killing off/merging simplifying neurons
    dw1 += lam * w1
    dw2 += lam * w2

    # update each value
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2

    return w1, b1, w2, b2

def normalize(array_like):
    """maps every input to 0->1 where 0 is min and 1 is max value in set"""
    mins = np.min(array_like, axis=0)
    maxs = np.max(array_like, axis=0)
    return (array_like - mins) / (maxs - mins)

main()
