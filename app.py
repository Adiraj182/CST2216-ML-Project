import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src folder to path precisely
sys.path.append(os.path.abspath('src'))

# Import custom modules
from loan_eligibility import run_loan_model
from real_estate import run_real_estate_model
from ucla_neural_networks import run_admission_model
from unsupervised_clustering import run_clustering
from logger import get_logger

logger = get_logger("app")

# Streamlit config
st.set_page_config(page_title="ML Modeler", layout="wide", page_icon="🚀")

st.title("Machine Learning Models Interface")
st.markdown("Developed for CST2216 Business Intelligence System Infrastructure")
st.markdown("---")

# Caching execution logic to speed up web reruns
@st.cache_data
def get_loan_results(lr_c, rf_n_estimators, rf_max_depth):
    return run_loan_model('Datasets/credit.csv', lr_c, rf_n_estimators, rf_max_depth)

@st.cache_data
def get_real_estate_results(rf_n_estimators, rf_max_depth):
    return run_real_estate_model('Datasets/final.csv', rf_n_estimators, rf_max_depth)

@st.cache_data
def get_ucla_results(mlp_hidden_neurons, mlp_max_iter, mlp_learning_rate):
    return run_admission_model('Datasets/Admission.csv', mlp_hidden_neurons, mlp_max_iter, mlp_learning_rate)

@st.cache_data
def get_clustering_results(k_min, k_max):
    return run_clustering('Datasets/mall_customers.csv', k_min, k_max)

# Navigation
st.sidebar.title("Navigation")
menu = ["Project Overiew", "1. Loan Eligibility Model", "2. Real Estate Pricing", "3. UCLA Admission Networks", "4. Customer Clustering"]
choice = st.sidebar.selectbox("Select a Page:", menu)
logger.info(f"User navigated to: {choice}")

st.sidebar.markdown("---")

if choice == "Project Overiew":
    st.sidebar.info("Select a model from the dropdown above to view logic execution, tune parameters, and see output metrics.")
    st.header("Welcome to the Modularized ML Framework")
    st.write("This application serves as a unified frontend for 4 distinct machine learning projects:")
    st.markdown("- **Loan Eligibility Model**: Classification using Logistic Regression and Random Forest.")
    st.markdown("- **Real Estate Pricing**: Regression to predict housing prices base on features.")
    st.markdown("- **UCLA Admission Networks**: Multilayer Perceptrons forecasting university admissions.")
    st.markdown("- **Customer Clustering**: Unsupervised KMeans clustering for customer segmentation.")
    st.info("👈 Use the navigation pane on the left to explore each model.")

elif choice == "1. Loan Eligibility Model":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    lr_c = st.sidebar.select_slider("LogReg: Reg. Strength (C)", options=[0.001, 0.01, 0.1, 1.0, 5.0], value=1.0)
    rf_n_estimators = st.sidebar.slider("Random Forest: Trees", 10, 100, 50, 10)
    rf_max_depth = st.sidebar.slider("Random Forest: Max Depth", 1, 10, 4, 1)
    
    st.header("Loan Eligibility Classification")
    
    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=True):
         st.write('''
         **Overview:** This pipeline cleans credit data and trains models to classify whether a user is eligible for a loan.
         
         **Hyperparameters Explained:**
         *   **Logistic Regression Reg. Strength (C)**: Controls the penalty on extreme weights. Low values (0.01) heavily restrict the model (underfitting) to prevent anomalies, while high values (10.0) let the model trace the data intricately (increasing risk of overfitting).
         *   **Random Forest Trees (n_estimators)**: The number of independent decision trees built. More trees reduce variance/overfitting but take longer to process.
         *   **Random Forest Max Depth**: The longest path from the root node to a leaf. Limiting this forces the tree to generalize, drastically changing precision and recall!
         ''')

    try:
        with st.spinner("Processing Loan Eligibility Data..."):
            res = get_loan_results(lr_c, rf_n_estimators, rf_max_depth)
        
        st.success("Pipeline executed successfully!")
        logger.info("Loan Eligibility page: model results rendered successfully.")
        
        st.subheader("Model Accuracies")
        c1, c2, c3 = st.columns(3)
        c1.metric("Logistic Regression", f"{res['logistic_regression_acc']:.2%}")
        c2.metric("Decision Tree", f"{res['decision_tree_acc']:.2%}")
        c3.metric("Random Forest", f"{res['random_forest_acc']:.2%}")
        
        st.subheader("Classification Reports")
        c1, c2, c3 = st.columns(3)
        with c1:
             st.write("**Logistic Regression Report**")
             st.code(res['lr_report'], language=None)
        with c2:
             st.write("**Decision Tree Report**")
             st.code(res['dt_report'], language=None)
        with c3:
             st.write("**Random Forest Report**")
             st.code(res['rf_report'], language=None)
             
        st.subheader("K-Fold Cross Validation Means")
        st.write(f"**Logistic Regression**: {res['lr_kfold_mean']:.4f}")
        st.write(f"**Random Forest**: {res['rf_kfold_mean']:.4f}")
        
    except Exception as e:
        logger.error(f"Loan Eligibility page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")

elif choice == "2. Real Estate Pricing":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    rf_n_estimators = st.sidebar.slider("Random Forest Regressor: Trees", 10, 100, 50, 10)
    rf_max_depth = st.sidebar.slider("Random Forest: Max Depth", 1, 15, 6, 1)

    st.header("Real Estate Regression Model")
    
    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=True):
         st.write('''
         **Overview:** This pipeline leverages Continuous Regression to predict exactly how much a condominium or house will be priced at based on its features (bedrooms, bathrooms, area, etc).
         
         **Hyperparameters Explained:**
         *   **Random Forest Trees (n_estimators)**: The total number of regression trees in the forest. A higher number aggregates predictions across more trees to smooth out outlier pricing guesses, stabilizing the Mean Absolute Error (MAE).
         *   **Random Forest Max Depth**: Limits the depth of the regression trees. If the depth is unlimited, the model might perfectly memorize training real estate prices, but setting a limit will immediately force the MAE error rates to jump as it generalizes.
         ''')

    try:
        with st.spinner("Running Regression pipelines..."):
            res = get_real_estate_results(rf_n_estimators, rf_max_depth)
        
        st.success("Trained successfully!")
        logger.info("Real Estate page: model results rendered successfully.")
        st.subheader("Mean Absolute Error (MAE)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Linear Reg. (Train)", f"{res['linear_regression_train_mae']:.2f}")
        c2.metric("Linear Reg. (Test)", f"{res['linear_regression_test_mae']:.2f}")
        c3.metric("Random Forest (Test)", f"{res['random_forest_test_mae']:.2f}")
        
    except Exception as e:
        logger.error(f"Real Estate page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")

elif choice == "3. UCLA Admission Networks":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    mlp_hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", 1, 10, 3, 1)
    mlp_max_iter = st.sidebar.slider("Maximum Iterations", 10, 300, 150, 10)
    mlp_learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)

    st.header("UCLA Admission Artificial Neural Networks")
    
    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=True):
         st.write('''
         **Overview:** This module creates an Artificial Neural Network (Multi-Layer Perceptron) to simulate and forecast whether a student has a high chance (>=80%) of admission based on their GRE, TOEFL, and CGPA scores.
         
         **Hyperparameters Explained:**
         *   **Learning Rate**: The most crucial parameter in Neural Networks! It determines how massive the network's mathematical steps are when trying to find the global minimum error. If it's too high (0.1), the gradients explode. If it's too low (0.0001), the model learns agonizingly slow and won't converge before reaching Max Iterations.
         *   **Hidden Layer Neurons**: The number of mathematical "nodes" in the network's processing layer. More neurons allow the model to recognize extremely complex test score patterns.
         *   **Maximum Iterations**: The maximum number of passes over the dataset (epochs) the network can make to reduce its error loss curve.
         ''')

    try:
        with st.spinner("Training Neural Networks..."):
            res = get_ucla_results(mlp_hidden_neurons, mlp_max_iter, mlp_learning_rate)
            
        st.success("Loss curves and accuracies updated!")
        logger.info("UCLA Admission page: model results rendered successfully.")
        
        # Display accuracies
        st.subheader("MLPClassifier Activation: ReLU")
        c1, c2 = st.columns(2)
        c1.metric("Train Accuracy", f"{res['relu_train_acc']:.2%}")
        c2.metric("Test Accuracy", f"{res['relu_test_acc']:.2%}")
        
        st.subheader("MLPClassifier Activation: Tanh")
        c1, c2 = st.columns(2)
        c1.metric("Train Accuracy", f"{res['tanh_train_acc']:.2%}")
        c2.metric("Test Accuracy", f"{res['tanh_test_acc']:.2%}")
        
        st.subheader("Classification Reports")
        c1, c2 = st.columns(2)
        with c1:
             st.write("**ReLU Activation Report**")
             st.code(res['relu_report'], language=None)
        with c2:
             st.write("**Tanh Activation Report**")
             st.code(res['tanh_report'], language=None)
        
        # Plotting Loss Curve
        st.subheader("Loss Curve (ReLU)")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(res['relu_loss_curve'], label='Loss', color='blue')
        ax.set_title('Loss Curve during MLP Training')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        logger.error(f"UCLA Admission page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")

elif choice == "4. Customer Clustering":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    k_min = st.sidebar.number_input("Minimum 'K' Clusters", min_value=2, max_value=5, value=3)
    k_max = st.sidebar.number_input("Maximum 'K' Clusters", min_value=6, max_value=15, value=8)

    st.header("Mall Customers KMeans Clustering")
    
    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=True):
         st.write('''
         **Overview:** Unlike the other algorithms, this is *Unsupervised Mapping*. The model does not know what a "target" is. It groups raw mall customers together based purely on geometrical similarities in their Age, Income, and Spending Score.
         
         **Hyperparameters Explained:**
         *   **K Clusters Range**: KMeans requires us to guess exactly how many clusters (groups) exist beforehand. By defining a range, we can map multiple geometries and use the *Silhouette Score* to calculate which 'K' cluster amount perfectly divided the shoppers!
         ''')

    try:
        with st.spinner("Clustering dataset..."):
             res = get_clustering_results(k_min, k_max)
             
        st.success("Unsupervised clustering completed!")
        logger.info("Customer Clustering page: results rendered successfully.")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("2D Metrics (Income, Spend)")
            st.dataframe(res['cluster_stats_2d'].style.highlight_max(subset=['Silhouette_Score'], color='lightgreen'))
        with c2:
            st.subheader("3D Metrics (Age, Income, Spend)")
            st.dataframe(res['cluster_stats_3d'].style.highlight_max(subset=['Silhouette_Score'], color='lightgreen'))
            
        st.markdown("**(Highlighted rows indicate optimal clusters based on highest Silhouette Score)**")

    except Exception as e:
        logger.error(f"Customer Clustering page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")
