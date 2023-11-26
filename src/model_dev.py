import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):

        """
        Trains the model

        Args: 
            X_train: Training data
            y_train: Training labels
        
        Returns:
            None
        
        """

        pass

class LinearRegressionModel(Model):
    
    """
    Linear Regression Model

    """

    def train(self,X_train, y_train, **kwargs):
        
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Modle Training Complete")
            return reg

        except Exception as e:
            logging.error("The Model training is not working, error {}".format(e))
            raise e
        