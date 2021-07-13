"""Network base class"""
import os
import pickle
from abc import ABC, abstractmethod
# https://docs.python.org/3/library/abc.html

"""In Pytorch you would usually define the `forward` function which performs all the interesting computations"""


class Network(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define forward() method
    """

    def __init__(self, model_name='dummy_network'):
        """
        :param model_name: A descriptive name of the model
        """
        self.model_name = model_name
        # https://www.w3schools.com/python/gloss_python_self.asp#:~:text=The%20self%20parameter%20is%20a,that%20belongs%20to%20the%20class.

    @abstractmethod
    def forward(self, X):
        """perform the forward pass through a network"""

    def __repr__(self):
        return "This is the base class for all networks we will use"

    @abstractmethod
    def save_model(self, data=None):
        """ each model should know what are the relevant things it needs for saving itself."""


class Dummy(Network):
    """
    Dummy machine
    """

    def __init__(self, model_name="dummy_machine"):
        """
        :param model_name: A descriptive name of the model
        """
        super().__init__()
        self.model_name = model_name

    def forward(self, x):
        """
        :param x: The input to the network
        :return: set x to any integer larger than 59 to get passed
        """

        ########################################################################
        # TODO                                                                 #
        # Implement the dummy machine function.                                #
        #                                                                      #
        ########################################################################

        x = 61

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

    def __repr__(self):
        return "A dummy machine"
        # https://www.journaldev.com/22460/python-str-repr-functions

    def save_model(self, data=None):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))
