class Representative(object):
    """
    Representative sample of the algorithm
    """

    def __init__(self, value, label, net_output=None):
        """
        Creates a Representative object
        :param value: the value of the representative (i.e. the image)
        :param metric: the value of the metric
        :param iteration: the iteration at which the sample was selected as representative
        :param megabatch: the current megabatch
        :param net_output: the output that the neural network gives to the sample
        """
        self.value = value
        self.label = label
        self.net_output = net_output
        self.weight = 1.0

    def __eq__(self, other):
        if isinstance(other, Representative.__class__):
            return self.value.__eq__(other.value)
        return False