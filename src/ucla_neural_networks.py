import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from logger import get_logger

logger = get_logger(__name__)

def run_admission_model(data_path='Datasets/Admission.csv', mlp_hidden_neurons=3, mlp_max_iter=200, mlp_learning_rate=0.001):
    """
    Executes the UCLA Neural Networks Pipeline
    """
    logger.info("Starting UCLA Admission Neural Network Pipeline")
    results = {}
    
    try:
        logger.info(f"Loading dataset from {data_path}")
        data = pd.read_csv(data_path)
        
        # Converting the target variable into a categorical variable
        data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
        
        # Dropping columns
        if 'Serial_No' in data.columns:
            data = data.drop(['Serial_No'], axis=1)
            
        # convert University_Rating to categorical type
        data['University_Rating'] = data['University_Rating'].astype('object')
        data['Research'] = data['Research'].astype('object')
        
        logger.info("Getting dummy variables")
        # Create dummy variables for all 'object' type variables
        clean_data = pd.get_dummies(data, columns=['University_Rating','Research'], dtype='int')
        
        x = clean_data.drop(['Admit_Chance'], axis=1)
        y = clean_data['Admit_Chance']

        logger.info("Splitting the data")
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)
        
        logger.info("Scaling features")
        scaler = MinMaxScaler()
        xtrain_scaled = scaler.fit_transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        
        logger.info("Training MLPClassifier (ReLU)")
        MLP = MLPClassifier(hidden_layer_sizes=(mlp_hidden_neurons,), batch_size=50, max_iter=mlp_max_iter, learning_rate_init=mlp_learning_rate, random_state=123)
        MLP.fit(xtrain_scaled, ytrain)
        
        ypred_train = MLP.predict(xtrain_scaled)
        train_acc = accuracy_score(ytrain, ypred_train)
        ypred = MLP.predict(xtest_scaled)
        test_acc = accuracy_score(ytest, ypred)
        
        results['relu_train_acc'] = train_acc
        results['relu_test_acc'] = test_acc
        results['relu_loss_curve'] = MLP.loss_curve_
        results['relu_report'] = classification_report(ytest, ypred)
        
        logger.info("Training MLPClassifier (Tanh)")
        MLP_tanh = MLPClassifier(hidden_layer_sizes=(mlp_hidden_neurons,), batch_size=50, max_iter=mlp_max_iter, learning_rate_init=mlp_learning_rate, random_state=123, activation='tanh')
        MLP_tanh.fit(xtrain_scaled, ytrain)
        
        ypred_train_tanh = MLP_tanh.predict(xtrain_scaled)
        train_acc_tanh = accuracy_score(ytrain, ypred_train_tanh)
        ypred_test_tanh = MLP_tanh.predict(xtest_scaled)
        test_acc_tanh = accuracy_score(ytest, ypred_test_tanh)
        
        results['tanh_train_acc'] = train_acc_tanh
        results['tanh_test_acc'] = test_acc_tanh
        results['tanh_report'] = classification_report(ytest, ypred_test_tanh)
        
        logger.info(f"UCLA pipeline done — ReLU test acc: {results.get('relu_test_acc', 'N/A'):.2%}, Tanh test acc: {results.get('tanh_test_acc', 'N/A'):.2%}")
        logger.info("UCLA Pipeline executed successfully.")
        return results

    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}. Please check the file path.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise
