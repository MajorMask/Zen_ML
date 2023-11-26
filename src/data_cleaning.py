import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining a strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):

        """
        Strategy for Preprocessing data
        """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
     
        """
        Preprocess data
        """
        try:
            data = data.drop(
                  [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delieverd_customer_date",
                    "oder_estimated_delivery_date",
                    "order_purchase_timestamp",
                  ],
                  axis = 1
                  )
