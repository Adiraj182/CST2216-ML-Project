import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src folder to path
sys.path.append(os.path.abspath('src'))

from loan_eligibility import run_loan_model
from real_estate import run_real_estate_model
from ucla_neural_networks import run_admission_model
from unsupervised_clustering import run_clustering
from logger import get_logger

logger = get_logger("app")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Modeler", layout="wide", page_icon="🚀")
st.title("Machine Learning Models Interface")
st.markdown("Developed for CST2216 Business Intelligence System Infrastructure")
st.markdown("---")

# ── Cache: use cache_resource so sklearn objects (models, scalers) are cached properly
@st.cache_resource
def get_loan_results(lr_c, rf_n_estimators, rf_max_depth):
    return run_loan_model('Datasets/credit.csv', lr_c, rf_n_estimators, rf_max_depth)

@st.cache_resource
def get_real_estate_results(rf_n_estimators, rf_max_depth):
    return run_real_estate_model('Datasets/final.csv', rf_n_estimators, rf_max_depth)

@st.cache_resource
def get_ucla_results(mlp_hidden_neurons, mlp_max_iter, mlp_learning_rate):
    return run_admission_model('Datasets/Admission.csv', mlp_hidden_neurons, mlp_max_iter, mlp_learning_rate)

@st.cache_resource
def get_clustering_results(k_min, k_max):
    return run_clustering('Datasets/mall_customers.csv', k_min, k_max)

# ── Navigation ────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
menu = [
    "Project Overview",
    "1. Loan Eligibility Model",
    "2. Real Estate Pricing",
    "3. UCLA Admission Networks",
    "4. Customer Clustering"
]
choice = st.sidebar.selectbox("Select a Page:", menu)
logger.info(f"User navigated to: {choice}")
st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Project Overview
# ══════════════════════════════════════════════════════════════════════════════
if choice == "Project Overview":
    st.sidebar.info("Select a model from the dropdown above to view training metrics, tune parameters, and make live predictions.")
    st.header("Welcome to the Modularized ML Framework")
    st.write("This application serves as a unified frontend for 4 distinct machine learning projects:")
    st.markdown("- **Loan Eligibility Model**: Predict whether a loan application will be approved or denied.")
    st.markdown("- **Real Estate Pricing**: Predict housing prices based on property features.")
    st.markdown("- **UCLA Admission Networks**: Predict university admission chances using Neural Networks.")
    st.markdown("- **Customer Clustering**: Discover customer segments using unsupervised K-Means clustering.")
    st.info("👈 Use the navigation pane on the left to explore each model.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Loan Eligibility
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "1. Loan Eligibility Model":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    lr_c = st.sidebar.select_slider("LogReg: Reg. Strength (C)", options=[0.001, 0.01, 0.1, 1.0, 5.0], value=1.0)
    rf_n_estimators = st.sidebar.slider("Random Forest: Trees", 10, 100, 50, 10)
    rf_max_depth = st.sidebar.slider("Random Forest: Max Depth", 1, 10, 4, 1)

    st.header("Loan Eligibility Classification")

    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=False):
        st.write('''
        **Overview:** This pipeline cleans credit data and trains models to classify whether a user is eligible for a loan.

        **Hyperparameters Explained:**
        - **Logistic Regression Reg. Strength (C)**: Controls the penalty on extreme weights. Low values restrict the model (underfitting); high values let the model fit the data closely (risk of overfitting).
        - **Random Forest Trees (n_estimators)**: More trees reduce variance but increase compute time.
        - **Random Forest Max Depth**: Limits the tree depth. Lower depth forces generalization.
        ''')

    try:
        with st.spinner("Training models on credit data..."):
            res = get_loan_results(lr_c, rf_n_estimators, rf_max_depth)

        st.success("✅ Pipeline executed successfully!")
        logger.info("Loan Eligibility page: model results rendered successfully.")

        # ── Training Metrics ─────────────────────────────────────────────────
        st.subheader("📊 Model Accuracies")
        c1, c2, c3 = st.columns(3)
        c1.metric("Logistic Regression", f"{res['logistic_regression_acc']:.2%}")
        c2.metric("Decision Tree", f"{res['decision_tree_acc']:.2%}")
        c3.metric("Random Forest", f"{res['random_forest_acc']:.2%}")

        st.subheader("📋 Classification Reports")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Logistic Regression**")
            st.code(res['lr_report'], language=None)
        with c2:
            st.write("**Decision Tree**")
            st.code(res['dt_report'], language=None)
        with c3:
            st.write("**Random Forest**")
            st.code(res['rf_report'], language=None)

        st.subheader("🔁 K-Fold Cross Validation Means")
        st.write(f"**Logistic Regression**: {res['lr_kfold_mean']:.4f}")
        st.write(f"**Random Forest**: {res['rf_kfold_mean']:.4f}")

        st.markdown("---")

        # ── Live Prediction Form ──────────────────────────────────────────────
        st.subheader("🔮 Try It Yourself — Predict Loan Eligibility")
        st.write("Enter applicant details below and the trained models will predict whether the loan should be approved.")

        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            with col2:
                self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
                credit_history = st.selectbox("Credit History", ["1 (Good)", "0 (Bad/None)"])
                loan_term = st.selectbox("Loan Amount Term (months)", ["360", "180", "240", "120", "84", "60", "300", "480", "36"])
            with col3:
                applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
                coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0, step=500)
                loan_amount = st.number_input("Loan Amount ($k)", min_value=1, value=150, step=10)

            predict_btn = st.form_submit_button("🔍 Predict Loan Eligibility")

        if predict_btn:
            logger.info("Loan prediction requested by user.")
            credit_val = 1 if credit_history.startswith("1") else 0

            # Build a raw dict matching the pre-encoded feature set
            raw = {
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': float(coapplicant_income),
                'LoanAmount': float(loan_amount),
                'Loan_Amount_Term': loan_term,
                'Credit_History': str(credit_val),
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'Property_Area': property_area,
            }
            raw_df = pd.DataFrame([raw])
            raw_df['Credit_History'] = raw_df['Credit_History'].astype('object')
            raw_df['Loan_Amount_Term'] = raw_df['Loan_Amount_Term'].astype('object')

            # One-hot encode categoricals to match training columns
            raw_encoded = pd.get_dummies(raw_df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype=int)

            # Align to training feature columns (adds missing dummies as 0, drops any extras)
            feature_cols = res['feature_cols']
            for col in feature_cols:
                if col not in raw_encoded.columns:
                    raw_encoded[col] = 0
            raw_encoded = raw_encoded[feature_cols]

            # Scale using the fitted scaler from training
            raw_scaled = res['scaler'].transform(raw_encoded)

            # Predictions from all 3 models
            lr_pred = res['lr_model'].predict(raw_scaled)[0]
            lr_prob = res['lr_model'].predict_proba(raw_scaled)[0][1]
            dt_pred = res['dt_model'].predict(raw_scaled)[0]
            rf_pred = res['rf_model'].predict(raw_scaled)[0]
            rf_prob = res['rf_model'].predict_proba(raw_scaled)[0][1]

            logger.info(f"Loan prediction result — LR: {lr_pred}, DT: {dt_pred}, RF: {rf_pred}")

            st.markdown("### 📣 Prediction Results")
            r1, r2, r3 = st.columns(3)
            with r1:
                if lr_pred == 1:
                    st.success(f"**Logistic Regression**\n\n✅ APPROVED\n\nConfidence: {lr_prob:.1%}")
                else:
                    st.error(f"**Logistic Regression**\n\n❌ DENIED\n\nRisk: {1-lr_prob:.1%}")
            with r2:
                if dt_pred == 1:
                    st.success("**Decision Tree**\n\n✅ APPROVED")
                else:
                    st.error("**Decision Tree**\n\n❌ DENIED")
            with r3:
                if rf_pred == 1:
                    st.success(f"**Random Forest**\n\n✅ APPROVED\n\nConfidence: {rf_prob:.1%}")
                else:
                    st.error(f"**Random Forest**\n\n❌ DENIED\n\nRisk: {1-rf_prob:.1%}")

    except Exception as e:
        logger.error(f"Loan Eligibility page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Real Estate Pricing
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "2. Real Estate Pricing":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    rf_n_estimators = st.sidebar.slider("Random Forest Regressor: Trees", 10, 100, 50, 10)
    rf_max_depth = st.sidebar.slider("Random Forest: Max Depth", 1, 15, 6, 1)

    st.header("Real Estate Regression Model")

    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=False):
        st.write('''
        **Overview:** This pipeline uses Regression to predict exactly how much a property will be priced at based on its features.

        **Hyperparameters Explained:**
        - **Random Forest Trees (n_estimators)**: More trees smooth out outlier price predictions.
        - **Random Forest Max Depth**: Limits tree depth. Lower values force the model to generalize.
        ''')

    try:
        with st.spinner("Running Regression pipelines..."):
            res = get_real_estate_results(rf_n_estimators, rf_max_depth)

        st.success("✅ Trained successfully!")
        logger.info("Real Estate page: model results rendered successfully.")

        # ── Training Metrics ─────────────────────────────────────────────────
        st.subheader("📊 Mean Absolute Error (MAE)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Linear Reg. (Train)", f"${res['linear_regression_train_mae']:,.0f}")
        c2.metric("Linear Reg. (Test)", f"${res['linear_regression_test_mae']:,.0f}")
        c3.metric("Random Forest (Test)", f"${res['random_forest_test_mae']:,.0f}")
        st.caption("MAE = average dollar difference between predicted and actual price. Lower is better.")

        st.markdown("---")

        # ── Live Prediction Form ──────────────────────────────────────────────
        st.subheader("🔮 Try It Yourself — Predict Property Price")
        st.write("Enter property details below and the models will predict the sale price.")

        with st.form("realestate_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                property_type = st.selectbox("Property Type", ["House", "Condo"])
                beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
                baths = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
                sqft = st.number_input("Square Footage", min_value=400, max_value=10000, value=1800, step=100)
            with col2:
                year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=1995)
                year_sold = st.number_input("Year Sold", min_value=1993, max_value=2016, value=2010)
                lot_size = st.number_input("Lot Size (sqft)", min_value=0, max_value=50000, value=5000, step=500)
                basement = st.selectbox("Has Basement?", ["Yes", "No"])
            with col3:
                property_tax = st.number_input("Annual Property Tax ($)", min_value=0, max_value=20000, value=3000, step=100)
                insurance = st.number_input("Annual Insurance ($)", min_value=0, max_value=5000, value=500, step=50)
                popular = st.selectbox("Popular Neighbourhood?", ["Yes", "No"])
                recession = st.selectbox("Sold During Recession?", ["No", "Yes"])

            predict_btn = st.form_submit_button("💰 Predict Sale Price")

        if predict_btn:
            logger.info("Real estate prediction requested by user.")
            property_age = year_sold - year_built

            input_data = {
                'year_sold': year_sold,
                'property_tax': property_tax,
                'insurance': insurance,
                'beds': beds,
                'baths': baths,
                'sqft': sqft,
                'year_built': year_built,
                'lot_size': lot_size,
                'basement': 1 if basement == "Yes" else 0,
                'popular': 1 if popular == "Yes" else 0,
                'recession': 1 if recession == "Yes" else 0,
                'property_age': property_age,
                'property_type_Condo': 1 if property_type == "Condo" else 0,
            }

            input_df = pd.DataFrame([input_data])
            # Ensure column order matches training
            feature_cols = res['feature_cols']
            input_df = input_df[feature_cols]

            lr_price = res['lr_model'].predict(input_df)[0]
            rf_price = res['rf_model'].predict(input_df)[0]

            logger.info(f"Real estate prediction — Linear: ${lr_price:,.0f}, RF: ${rf_price:,.0f}")

            st.markdown("### 📣 Prediction Results")
            r1, r2 = st.columns(2)
            r1.success(f"**Linear Regression**\n\n💰 Predicted Price: **${lr_price:,.0f}**")
            r2.success(f"**Random Forest**\n\n💰 Predicted Price: **${rf_price:,.0f}**")

    except Exception as e:
        logger.error(f"Real Estate page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: UCLA Admission Networks
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "3. UCLA Admission Networks":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    mlp_hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", 1, 10, 3, 1)
    mlp_max_iter = st.sidebar.slider("Maximum Iterations", 10, 300, 150, 10)
    mlp_learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)

    st.header("UCLA Admission Artificial Neural Networks")

    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=False):
        st.write('''
        **Overview:** Trains a Multi-Layer Perceptron to predict whether a student has a high chance (≥80%) of admission to UCLA.

        **Hyperparameters Explained:**
        - **Learning Rate**: How large the update steps are. Too high → gradients explode. Too low → slow convergence.
        - **Hidden Layer Neurons**: More neurons = more pattern complexity the network can learn.
        - **Maximum Iterations**: Maximum number of epochs (passes over training data).
        ''')

    try:
        with st.spinner("Training Neural Networks..."):
            res = get_ucla_results(mlp_hidden_neurons, mlp_max_iter, mlp_learning_rate)

        st.success("✅ Loss curves and accuracies updated!")
        logger.info("UCLA Admission page: model results rendered successfully.")

        # ── Training Metrics ─────────────────────────────────────────────────
        st.subheader("📊 Model Accuracies")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ReLU Activation**")
            m1, m2 = st.columns(2)
            m1.metric("Train Accuracy", f"{res['relu_train_acc']:.2%}")
            m2.metric("Test Accuracy", f"{res['relu_test_acc']:.2%}")
        with col2:
            st.write("**Tanh Activation**")
            m1, m2 = st.columns(2)
            m1.metric("Train Accuracy", f"{res['tanh_train_acc']:.2%}")
            m2.metric("Test Accuracy", f"{res['tanh_test_acc']:.2%}")

        st.subheader("📋 Classification Reports")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**ReLU Activation Report**")
            st.code(res['relu_report'], language=None)
        with c2:
            st.write("**Tanh Activation Report**")
            st.code(res['tanh_report'], language=None)

        st.subheader("📉 Loss Curve (ReLU)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(res['relu_loss_curve'], label='Loss', color='royalblue')
        ax.set_title('Loss Curve during MLP Training')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.markdown("---")

        # ── Live Prediction Form ──────────────────────────────────────────────
        st.subheader("🔮 Try It Yourself — Predict Admission Chance")
        st.write("Enter your academic profile below. The networks will predict your UCLA admission chance.")

        with st.form("ucla_form"):
            col1, col2 = st.columns(2)
            with col1:
                gre_score = st.slider("GRE Score", min_value=260, max_value=340, value=315)
                toefl_score = st.slider("TOEFL Score", min_value=92, max_value=120, value=107)
                cgpa = st.slider("CGPA (out of 10)", min_value=6.0, max_value=10.0, value=8.5, step=0.1)
                university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
            with col2:
                sop = st.slider("Statement of Purpose Strength (SOP)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
                lor = st.slider("Letter of Recommendation Strength (LOR)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
                research = st.selectbox("Research Experience", ["Yes", "No"])

            predict_btn = st.form_submit_button("🎓 Predict Admission Chance")

        if predict_btn:
            logger.info("UCLA admission prediction requested by user.")
            research_val = 1 if research == "Yes" else 0

            # Build raw DataFrame matching training structure
            raw = {
                'GRE_Score': gre_score,
                'TOEFL_Score': toefl_score,
                'SOP': sop,
                'LOR': lor,
                'CGPA': cgpa,
                'University_Rating': str(university_rating),
                'Research': str(research_val),
            }
            raw_df = pd.DataFrame([raw])
            raw_df['University_Rating'] = raw_df['University_Rating'].astype('object')
            raw_df['Research'] = raw_df['Research'].astype('object')

            # One-hot encode to match training
            raw_encoded = pd.get_dummies(raw_df, columns=['University_Rating', 'Research'], dtype='int')
            feature_cols = res['feature_cols']
            for col in feature_cols:
                if col not in raw_encoded.columns:
                    raw_encoded[col] = 0
            raw_encoded = raw_encoded[feature_cols]

            # Scale using fitted scaler from training
            raw_scaled = res['scaler'].transform(raw_encoded)

            relu_pred = res['relu_model'].predict(raw_scaled)[0]
            tanh_pred = res['tanh_model'].predict(raw_scaled)[0]
            relu_prob = res['relu_model'].predict_proba(raw_scaled)[0][1]
            tanh_prob = res['tanh_model'].predict_proba(raw_scaled)[0][1]

            logger.info(f"UCLA prediction — ReLU: {relu_pred}, Tanh: {tanh_pred}")

            st.markdown("### 📣 Prediction Results")
            r1, r2 = st.columns(2)
            with r1:
                if relu_pred == 1:
                    st.success(f"**ReLU Network**\n\n🎓 HIGH Admission Chance\n\nProbability: {relu_prob:.1%}")
                else:
                    st.warning(f"**ReLU Network**\n\n📋 LOW Admission Chance\n\nProbability: {relu_prob:.1%}")
            with r2:
                if tanh_pred == 1:
                    st.success(f"**Tanh Network**\n\n🎓 HIGH Admission Chance\n\nProbability: {tanh_prob:.1%}")
                else:
                    st.warning(f"**Tanh Network**\n\n📋 LOW Admission Chance\n\nProbability: {tanh_prob:.1%}")

    except Exception as e:
        logger.error(f"UCLA Admission page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Customer Clustering
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "4. Customer Clustering":
    st.sidebar.subheader("⚙️ Model Hyperparameters")
    k_min = st.sidebar.number_input("Minimum 'K' Clusters", min_value=2, max_value=5, value=3)
    k_max = st.sidebar.number_input("Maximum 'K' Clusters", min_value=6, max_value=15, value=8)

    st.header("Mall Customers KMeans Clustering")

    with st.expander("ℹ️ About This Algorithm & Parameters", expanded=False):
        st.write('''
        **Overview:** Unsupervised learning — the model discovers natural customer segments with no labels.

        **Hyperparameters Explained:**
        - **K Clusters Range**: KMeans needs a pre-set K. By testing a range, we find the optimal K using the Silhouette Score.
        ''')

    try:
        with st.spinner("Clustering dataset..."):
            res = get_clustering_results(k_min, k_max)

        st.success("✅ Unsupervised clustering completed!")
        logger.info("Customer Clustering page: results rendered successfully.")

        # ── Training Metrics ─────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2D Metrics (Income, Spend)")
            st.dataframe(res['cluster_stats_2d'].style.highlight_max(subset=['Silhouette_Score'], color='lightgreen'))
            st.caption(f"✅ Optimal K = **{res['best_k_2d']}** clusters (highest Silhouette Score)")
        with c2:
            st.subheader("📊 3D Metrics (Age, Income, Spend)")
            st.dataframe(res['cluster_stats_3d'].style.highlight_max(subset=['Silhouette_Score'], color='lightgreen'))
            st.caption(f"✅ Optimal K = **{res['best_k_3d']}** clusters (highest Silhouette Score)")

        st.markdown("**(Highlighted rows indicate optimal clusters based on highest Silhouette Score)**")
        st.markdown("---")

        # ── Live Prediction Form ──────────────────────────────────────────────
        st.subheader("🔮 Try It Yourself — Find Your Customer Segment")
        st.write("Enter your profile below to discover which customer segment you belong to.")

        with st.form("clustering_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", min_value=18, max_value=70, value=35)
            with col2:
                annual_income = st.slider("Annual Income ($k)", min_value=15, max_value=137, value=60)
            with col3:
                spending_score = st.slider("Spending Score (1–100)", min_value=1, max_value=100, value=50)

            predict_btn = st.form_submit_button("🏷️ Find My Segment")

        if predict_btn:
            logger.info("Clustering prediction requested by user.")

            # 2D prediction (Income + Spending)
            input_2d = pd.DataFrame([[annual_income, spending_score]], columns=['Annual_Income', 'Spending_Score'])
            cluster_2d = res['model_2d'].predict(input_2d)[0]

            # 3D prediction (Age + Income + Spending)
            input_3d = pd.DataFrame([[age, annual_income, spending_score]], columns=['Age', 'Annual_Income', 'Spending_Score'])
            cluster_3d = res['model_3d'].predict(input_3d)[0]

            logger.info(f"Clustering prediction — 2D Segment: {cluster_2d}, 3D Segment: {cluster_3d}")

            st.markdown("### 📣 Prediction Results")
            r1, r2 = st.columns(2)
            r1.info(f"**2D Model (Income + Spending)**\n\n🏷️ You belong to **Customer Segment #{cluster_2d}**\n\nModel was trained with optimal K = {res['best_k_2d']} clusters")
            r2.info(f"**3D Model (Age + Income + Spending)**\n\n🏷️ You belong to **Customer Segment #{cluster_3d}**\n\nModel was trained with optimal K = {res['best_k_3d']} clusters")

    except Exception as e:
        logger.error(f"Customer Clustering page error: {str(e)}", exc_info=True)
        st.error(f"Error loading system: {str(e)}")
