import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from logger import get_logger

logger = get_logger(__name__)

def run_real_estate_model(data_path='Datasets/final.csv', rf_n_estimators=50, rf_max_depth=None):
    """
    Executes the Real Estate Pipeline.
    """
    logger.info("Starting Real Estate Pipeline")
    results = {}
    
    try:
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        # separate input features in x
        x = df.drop('price', axis=1)
        # store the target variable in y
        y = df['price']
        
        logger.info("Splitting the dataset")
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=x.property_type_Condo, random_state=42)
        
        logger.info("Training Linear Regression Model")
        # train your model
        model = LinearRegression()
        lrmodel = model.fit(x_train, y_train)
        
        # evaluation
        train_pred = lrmodel.predict(x_train)
        train_mae = mean_absolute_error(train_pred, y_train)
        test_pred = lrmodel.predict(x_test)
        test_mae = mean_absolute_error(test_pred, y_test)
        
        results['linear_regression_train_mae'] = train_mae
        results['linear_regression_test_mae'] = test_mae
        
        logger.info("Training Random Forest Regressor Model")
        # create an instance of the model
        rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, criterion='absolute_error', random_state=42) # reduced from 200 to 100 for streamlite speed
        
        # train the model
        rfmodel = rf.fit(x_train, y_train)
        
        # make predictions
        ytest_pred = rfmodel.predict(x_test)
        rf_test_mae = mean_absolute_error(ytest_pred, y_test)
        
        results['random_forest_test_mae'] = rf_test_mae
        
        logger.info("Saving models via pickle")
        try:
            pickle.dump(lrmodel, open('RE_LR_Model.pkl','wb'))
            pickle.dump(rfmodel, open('RE_RF_Model.pkl','wb'))
        except Exception as p_e:
            logger.warning(f"Could not pickle models: {str(p_e)}")
            
        logger.info(f"Real Estate pipeline done — Linear MAE: {train_mae:.2f} (train), {test_mae:.2f} (test) | RF MAE: {rf_test_mae:.2f}")
        logger.info("Real Estate Pipeline executed successfully.")
        return results

    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}. Please check the file path.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise
