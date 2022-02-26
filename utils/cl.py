class Representative(object):
    """
    Representative sample of the algorithm
    """

    def __init__(self, x, y, net_output=None):
        """
        Creates a Representative object
        :param x: the value of the representative (i.e. the image)
        :param y: the label attached to the value x
        :param net_output: the output that the neural network gives to the sample
        """
        self.x = x
        self.y = y
        self.net_output = net_output
        self.weight = 1.0

    def __eq__(self, other):
        if isinstance(other, Representative.__class__):
            return self.x.__eq__(other.x)
        return False

    def get_size(self):
        return (self.x.element_size() * self.x.nelement()) / 1000000000