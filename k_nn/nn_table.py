#!/home/syrup/anaconda3/envs/tensorflow/bin/python

class KNNTable(object):
    """A table structure that is used for K nearest neighbor algorithm
    has manhattan/euclidian distance options
    has average/vote decision options
    keeps track of min/max for each field on insert for easy normalization
    only takes float inputs
    """

    def __init__(num_input_attr, num_output_attr):
        """takes number of input/output attributes"""
        self.num_input = num_input_attr
        self.num_output = num_output_attr
        self.inputs = []
        self.outputs = []
        self.min = [None] * self.num_input 
        self.max = [None] * self.num_input 

    def insert_record(x, y):
        """inserts record into list, updates min/max for normalization
        x should be of length specified in init
        y should be of length specified in init"""
        for attr in range(self.num_input):
            self.update_min_max(attr, x[attr])
        self.inputs.append(x)
        self.outputs.append(y)

    def update_min_max(attr_num, value):
        """updates the stored min/max for the specified attribute"""
        if self.min[attr_num] is None or self.min[attr_num] > value:
            self.min[attr_num] = value
        if self.max[attr_num] is None or self.max[attr_num] < value:
            self.max[attr_num] = value

    def normalize_attr(attr, value):
        """returns the value as a normalized value between 0 and 1 
        where 0 is min value of that attr and 1 is max"""
        norm = (value - self.min[attr])/(self.max[attr] - self.min[attr])
        return norm

    def get_euclidean_distance(record1, record2):
        """Class/instance method
        calculates sum of differences squared.""" 
        dist = 0
        for attr in len(record1):
            p1 = normalize_attr(attr, record1[attr])
            p2 = normalize_attr(attr, record2[attr])
            dist += (p1 - p2) ** 2
        dist **= 0.5
        return dist

    def get_manhattan_distance(record1, record2):
        """Class/instance method
        calculates sum of absolute differences"""
        dist = 0
        for attr in len(record1):
            p1 = normalize_attr(attr, record1[attr])
            p2 = normalize_attr(attr, record2[attr])
            dist += fabs(p1 - p2)
        return dist

    def get_most_popular_value(inputs, weights=[]):
        """returns the most popular value in the input list
        weights is unused, returns None if no vals provided"""
        freq = {}
        for val in inputs:
            if val not in freq:
                freq[val] = 0
            freq[val] += 1
        most_times = 0
        mode = None
        for val in inputs:
            if freq[val] > most_times:
                most_times = freq[val]
                mode = val
        return val

    def get_average_of_list(inputs, weights=[]):
        """returns the straight average of all values in the inputs list
        weights is unused, returns None if no vals provided"""
        ave = 0
        for val in inputs:
            ave += val
        if len(inputs) == 0:
            return None
        return ave/len(inputs)

    def get_weighted_average_of_list(inputs, weights=[]):
        """returns the weighted average of all values in the inputs list
        using the corresponding values in weights.

        weights must be a value between 0 and 1.  a weight of 0 means that
        the corresponding input gets 100% weight, and a value of 1 means the
        corresponding input gets 0% weight.  Just screw you buddy"""
        ave = 0
        for i, val in enumnerate(inputs):
            if weights[i] > 1 or weights[i] < 0:
                msg = f"weight value of {weights[i]} at index {i} "
                msg += "is not between 0 and 1"
                raise ValueError(msg)
            ave += val * (1 - weights[i])
        if len(inputs) == 0:
            return None
        return ave/len(inputs)

    def predict_value_from_k_nearest_neighbors(test_record, k=1,
                                               distance_type="euclidean",
                                               vote_type="classify"):
        """finds k nearest neighbors of the test record, then predicts the 
        output values of that test record.

        k defaults to 1
        distance_type defaults to "euclidean" but can be anything specified in
        get_k_nearest_neighbors
        vote_type can be any of the following values:
            "classify"          : (default) take most popular output values 
                                  from neighbors
            "average"           : take strict average of output values from 
                                  neighbors
            "weighted average"  : average outputs by calculated distance to 
                                  each neighbor
        """
        dist, nbr, nbr_val = get_k_nearest_neighbors(test_record, k, 
                                                     distance_type)
        vote_func = None
        if vote_type == "classify":
            vote_func = get_most_popular_value
        elif vote_type == "average":
            vote_func = get_average_of_list
        elif vote_type =="weighted average":
            vote_func = get_weighted_average_of_list
        else:
            raise ValueError(f"unexpected vote_type {vote_type}")
        value = [None] * self.num_output
        for attr in range(self.num_output):
            value[attr] = vote_func(nbr_bal, dist)
        return value


    def get_k_nearest_neighbors(test_record, k=1, distance_type="euclidean"):
        """finds the k nearest neighbors to record inside of instance 
        k defaults to 1 but can be set
        distance_type can be "euclidean" or "manhattan"

        returns three lists:
            dist, x, y

            distances         -> list of distances for each returned neighbor
            neighbors         -> list of returned neighbors
            neighbors_outputs -> list of outputs of returned neighbors

        if there are multiple records at the same distance from the test such 
        that we need to include at least one of them in the k nearest, all 
        will be included

        looks through every record and puts each one into a dict where the
        keys are the distance and the value is a list of records at that 
        distance

        sorts the keys of the hash, then adds lists until we have k matches 
        """
        distanced_records = {}
        distance_func = None
        if distance_type == "euclidean":
            distance_func = get_euclidean_distance
        elif distance_type == "manhattan":
            distance_func = get_manhattan_distance
        else:
            raise ValueError(f"no distance calculation \"{distance_type}\"")
        # put each record's index into a dict by the record's distance
        for index, my_record in enumerate(self.inputs):
            dist = distance_func(my_record, test_record)
            if dist not in distanced_records:
                distanced_records[dist] = []
            distanced_records[dist].append(index)
        # sort the keys of the dict
        sorted_dist = sorted(distanced_records.keys())
        neighbors = []
        distances = []
        neighbors_outputs = []
        dist_index = 0
        # continuously add record indexes until we have at least k or
        # we run out of neighbors to add
        while len(neigbors) < k and dist_index < len(sorted_dist):
            # stick all the records 
            this_dist = sorted_dist[dist_index]
            these_neighbors_index = distanced_records[this_dist]
            for this_neighbor in these_neighbors_index:
                # add the input record, the output values, and the dist to
                # the lists
                neighbors.append(self.inputs[this_neighbor])
                neighbors_outputs.append(self.outputs[this_neighbor])
                distances.append(this_dist)
        dist_index += 1
        return distances, neighbors, neighbors_outputs








