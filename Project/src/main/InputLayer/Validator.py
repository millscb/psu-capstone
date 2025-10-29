"""Validator module :: contains the Validator class to validate input data."""



class Validator:
    """Class to validate input data."""
   

    def __init__(self, data: object):
        """ Constructor """
        self.data = data

    def is_valid(self) -> bool:
        """ Validate the input data """
        # Implement validation logic here
        return True