import logging
from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin

from src.evaluation import MSE, R2, RMSE
from typing_extensions import Annotated, Tuple

@step
def evaluate_model(model: RegressorMixin,
                   X_test:pd.DataFrame, 
                   y_test:pd.DataFrame
                   
        )->Tuple[
             Annotated[float,"r2_score"],
             Annotated[float,"rmse"],
        ]:

    """
    Evaluate the model
    Args:
        df: the ingested data

    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        
        return rmse, r2
    
    except Exception as e:
        logging.error("There is a problem in evaluating the model: {}".format(e))
        raise e