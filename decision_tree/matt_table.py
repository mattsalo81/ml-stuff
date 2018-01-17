import math

class MattTable(object):
    """is just a table for partitioning data"""

    def __init__(self):
        self.data = []
        self.source = ""
    
    def populate_from_file(self, input_file):
        with open(input_file) as fh:
            data = fh.read()
        lines = data.split("\n")
        for line in lines:
            str_record = line.split(",")
            if len(str_record) > 1:
                self.cols = len(str_record)
                record = [float(x) for x in str_record]
                self.data.append(record)
        self.source += f"read from file <{input_file}>\n"

    def pivot_on_continuous(self, attribute_no, value, output_attr):
        """return a table with records that have attribute_no less than value
        and a table with records that have attribute_no greater than or equal
        to value, finally returns the information gain as calculated by 
        information entropy"""
        less_than = MattTable()
        less_than.cols = self.cols
        less_than.source = self.source
        greater_than = MattTable()
        greater_than.cols = self.cols
        greater_than.source = self.source

        initial_entropy = self.calculate_entropy(output_attr)
        for record in self.data:
            if record[attribute_no] < value:
                less_than.data.append(record)
            else:
                greater_than.data.append(record)
        # info gain calc
        all_data = len(self.data)
        left_data = len(less_than.data)
        right_data = len(greater_than.data)
        left_split_entropy = less_than.calculate_entropy(output_attr)
        right_split_entropy = greater_than.calculate_entropy(output_attr)
        information_gain = initial_entropy
        information_gain -= left_split_entropy * left_data/all_data
        information_gain -= right_split_entropy * right_data/all_data

        less_than.source += f"pivoted on attr {attribute_no}"
        greater_than.source += f"pivoted on attr {attribute_no}"
        less_than.source += f" less than {value}\n"
        greater_than.source += f" greater than {value}\n"
        return less_than, greater_than, information_gain
    
    def summarize(self, attribute_no):
        """gets all possible values in the numbered column, with how many
        times each occurs"""
        values = {}
        for record in self.data:
            if record[attribute_no] not in values:
                values[record[attribute_no]] = 0
            values[record[attribute_no]] += 1
        return values
    
    def calculate_entropy(self, attribute_no):
        """calculates the info entropy for the named attribute. defined as
        -SIGMA_across_all_Xi[P(Xi)log_2(P(Xi))]"""
        values = self.summarize(attribute_no)
        no_records = len(self.data)
        entropy = 0
        for possible_value in values.keys():
            probability = values[possible_value] / no_records
            entropy += probability * math.log(probability, 2)
        return -entropy

    def pivot_on_best_info_gain(self, attribute_no, output_attr):
        best_gain = -1 
        best_left = MattTable()
        best_right = MattTable()
        best_pivot = -1
        values = self.summarize(attribute_no)
        for pivot in values.keys():
            left, right, gain = self.pivot_on_continuous(attribute_no, pivot,
                                                         output_attr)

            if gain > best_gain:
                best_left = left
                best_right = right
                best_gain = gain
                best_pivot = pivot
        return best_left, best_right, best_gain, best_pivot

    def pivot_on_best_attribute(self, output_attr):
        best_gain = -1
        best_left = MattTable()
        best_right = MattTable()
        best_attr = -1
        best_pivot = 0
        for attr_no in range(0, self.cols):
            if attr_no != output_attr:
                left, right, gain, pivot = self.pivot_on_best_info_gain(
                                                        attr_no, output_attr)

                if gain > best_gain:
                    best_gain = gain
                    best_left = left
                    best_right = right
                    best_attr = attr_no
                    best_pivot = pivot
            else:
                # this is the output dir, so skip
                pass
        return best_left, best_right, best_gain

    def get_vote(self, output_attr):
        """returns most likely output from table"""
        max_votes = 0
        vote = None
        for possible_vote, num_votes in self.summarize(output_attr).items():
            if num_votes >= max_votes:
                vote = possible_vote
                max_votes = num_votes
        return vote

    def make_decision_tree(self, output_attr, min_info_gain):
        gain = -1
        if len(self.data) > 1:
            left, right, gain = self.pivot_on_best_attribute(output_attr)
            leftlen = len(left.data)
            rightlen = len(right.data)
            print(f"pivoted, gain is {gain}")

        if gain > min_info_gain and leftlen != 0 and rightlen != 0:
            # recurse
            left.make_decision_tree(output_attr, min_info_gain)
            right.make_decision_tree(output_attr, min_info_gain)
        else:
            # we hit the bottom, so... ?
            entropy = self.calculate_entropy(output_attr)
            vote = self.get_vote(output_attr)
            print("STARTSTARTSTARTSTART")
            print(f"Hit a leaf for output {vote}, entropy is {entropy}")
            print(self.source)
            print(self.summarize(output_attr))
            print("ENDENDENDENDENDENDEND")


    def __str__(self):
        return f"{self.data}"

