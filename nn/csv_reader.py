
def read_from_file(input_file, output_attr):
    with open(input_file) as fh:
        data = fh.read()
    lines = data.split("\n")
    x = []
    y = []
    labels = lines.pop(0).split(",")
    labels.pop()
    for line in lines:
        str_record = line.split(",")
        if len(str_record) > output_attr:
            record = [float(x) for x in str_record]
            y.append(int(record.pop(output_attr)))
            x.append(record)
    return labels, x, y

def get_training_and_test_sets(x, y, one_in_x_test):
    n = 0
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(0, len(x)):
        n += 1
        if n == one_in_x_test:
            n = 0
            test_x.append(x[i])
            test_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])
    return train_x, train_y, test_x, test_y

