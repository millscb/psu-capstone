"""InputData module :: contains the InputData class to handle input data of any type."""

class InputData:
    """Class to handle input data :: of any type."""
    

    def __init__(self, data: object):
        """ Constructor """
        self.data = data

    def get_data(self) -> object:
        """ Getter for the data attribute """
        return self.data
