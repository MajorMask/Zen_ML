import logging
from zenml import step
import pandas as pd

@step
def evaluate_model(df: pd.DataFrame)->float:
    """
    Evaluate the model
    Args:
        df: the ingested data

    """
    return float(0)