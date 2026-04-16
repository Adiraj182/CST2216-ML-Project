import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def run_loan_model(data_path='Datasets/credit.csv', lr_c=1.0, rf_n_estimators=100, rf_max_depth=None):
    """
    Executes the Loan Eligibility Model pipeline.
    Returns metrics AND trained model objects so app.py can run live predictions.
    """
    logger.info("Starting Loan Eligibility Pipeline")
    results = {}
    
    try:
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        results['initial_shape'] = df.shape
        
        # Convert columns to object type
        df['Credit_History'] = df['Credit_History'].astype('object')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')
        
        logger.info("Imputing missing values")
        df['Gender'] = df['Gender'].fillna('Male')
        df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
        df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
        df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
        df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
        
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)
            
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype=int)
        df['Loan_Approved'] = df['Loan_Approved'].replace({'Y': 1, 'N': 0}).astype(int)
        
        x = df.drop('Loan_Approved', axis=1)
        y = df.Loan_Approved
        
        # Store feature columns so app.py can reconstruct the same shape for user input
        feature_cols = list(x.columns)
        results['feature_cols'] = feature_cols
        
        logger.info("Splitting and scaling data")
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
        
        scale = MinMaxScaler()
        xtrain_scaled = scale.fit_transform(xtrain)
        xtest_scaled = scale.transform(xtest)
        results['scaler'] = scale   # <-- returned so app.py can scale user input
        
        # 1. Logistic Regression
        logger.info("Training Logistic Regression Model")
        lrmodel = LogisticRegression(C=lr_c, max_iter=200)
        lrmodel.fit(xtrain_scaled, ytrain)
        lr_ypred = lrmodel.predict(xtest_scaled)
        results['logistic_regression_acc'] = accuracy_score(ytest, lr_ypred)
        results['lr_report'] = classification_report(ytest, lr_ypred)
        results['lr_model'] = lrmodel  # <-- returned for live prediction
        
        # 2. Decision Tree
        logger.info("Training Decision Tree Model")
        dtmodel = DecisionTreeClassifier()
        dtmodel.fit(xtrain_scaled, ytrain)
        dt_ypred = dtmodel.predict(xtest_scaled)
        results['decision_tree_acc'] = accuracy_score(ytest, dt_ypred)
        results['dt_report'] = classification_report(ytest, dt_ypred)
        results['dt_model'] = dtmodel  # <-- returned for live prediction
        
        # 3. Random Forest
        logger.info("Training Random Forest Model")
        rfmodel = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
        rfmodel.fit(xtrain_scaled, ytrain)
        rf_ypred = rfmodel.predict(xtest_scaled)
        results['random_forest_acc'] = accuracy_score(ytest, rf_ypred)
        results['rf_report'] = classification_report(ytest, rf_ypred)
        results['rf_model'] = rfmodel  # <-- returned for live prediction
        
        # KFold Cross Validation
        logger.info("Performing K-Fold Cross Validation")
        kfold = KFold(n_splits=5)
        lr_scores = cross_val_score(lrmodel, xtrain_scaled, ytrain, cv=kfold)
        rf_scores = cross_val_score(rfmodel, xtrain_scaled, ytrain, cv=kfold)
        results['lr_kfold_mean'] = lr_scores.mean()
        results['rf_kfold_mean'] = rf_scores.mean()
        
        logger.info(f"Loan pipeline done — LR acc: {results.get('logistic_regression_acc', 'N/A'):.2%}, RF acc: {results.get('random_forest_acc', 'N/A'):.2%}")
        logger.info("Loan Eligibility Pipeline executed successfully.")
        return results
        
    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}. Please check the file path.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise
