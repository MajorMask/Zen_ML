import logging

import pandas as pd
from zenml import step

class IngestData:

    """ Ingesting data from data_path """
    
    def __init__(self, data_path:str):
        self.data_path = data_path
    """ Args:
        data_path: path to the data
    """
    def get_data(self):
        
        """
    Ingesting the data from the data path
        """
        
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)
    

@step
def ingest_df(data_path:str) ->pd.DataFrame:
    """   
    # Ingesting the data from the data_path.

    Args:
        data_path: path to the data
    Return:
        pd.DataFrame: the ingested data

    """
    try:
        ingest_data=IngestData(data_path)
        df = ingest_data.get_data()
        print(df.head(10))
        return df
    except Exception as e:
        logging.error("Error while ingesting the data, error code: {e}")
        raise e
 
# if __name__=="__main__":
#     df = ingest_df("D:\Machine Learning Projects\Zen_ML\data\olist_customers_dataset.csv")
