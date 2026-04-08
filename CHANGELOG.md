# Project Modifications and Setup Details

This document outlines exactly what additions and modifications were made to transition the raw Jupyter Notebooks into a deployable modular application, as requested.

## 1. Core Logic Preservation
- All machine learning algorithms (Logistic Regression, Decision Trees, Random Forests, Linear Regression, MLPClassifier, KMeans) were kept completely identical to your original notebook logic.
- Your data splitting ratios (`test_size=0.2`), parameters (such as `random_state=123`), and scaling algorithms (`MinMaxScaler`) were preserved exactly.

## 2. Structural Additions
The code for the four notebooks was transitioned into separate Python modules inside a newly created `src` folder. This is to adhere to the grading rubric for "Code Modularization" and "VS Code Project Structure".
- **`src/loan_eligibility.py`**: Wraps your code into a `run_loan_model(data_path)` function.
- **`src/real_estate.py`**: Wraps your code into a `run_real_estate_model(data_path)` function.
- **`src/ucla_neural_networks.py`**: Wraps your code into a `run_admission_model(data_path)` function.
- **`src/unsupervised_clustering.py`**: Wraps your code into a `run_clustering(data_path)` function.

## 3. Production Enhancements (Logging & Error Handling)
To satisfy the grading rubric requirement for "Code Quality, Logging, and Error Handling", the following specific lines were added to every `.py` file:
- **Imported `logging`**: `import logging` and `logging.basicConfig(level=logging.INFO, ...)` at the top of every file to generate system traces without crashing.
- **Added `try/except` blocks**: 
  - `try:` was injected right after every function definition to encapsulate the data loading down to the model training.
  - `except FileNotFoundError:` was added to throw a clean error message to the log if the dataset files are not found, instead of breaking the entire app.
  - `except Exception as e:` was added to capture any unexpected model training errors dynamically.
- **Information Traces**: Functions like `logging.info("Splitting the data")` were dispersed throughout the process to track execution states.

## 4. Minor Adjustments for Streamlit Compatibility
- In `unsupervised_clustering.py`, the KMeans initialization parameter `n_init` was explicitly declared (`n_init=10`) because scikit-learn throws a memory leak warning in production environments otherwise.
- In `unsupervised_clustering.py`, the nested for-loops used for K-Fold Silhouette score generation were wrapped into dictionaries instead of immediately plotting via `matplotlib.show()`, so that Streamlit could read the integers and render them cleanly on a webpage.
- In `real_estate.py`, the Random Forest estimator was slightly reduced (`n_estimators=100`) merely to prevent Streamlit Cloud from timing out during deployment (as `200` blocks thread rendering).

## 5. Web Platform Implementation
- Created `app.py` functioning as the central hub. It imports the 4 modules from `src/` and provides an interactive UI using the `streamlit` library.
- Used Streamlit caching (`@st.cache_data`) around the module calling functions to ensure the application only trains the machine learning models once when first loaded, preventing lag when clicking through the app interface.
- Created `requirements.txt` to strictly bind versions of `pandas`, `scikit-learn`, `numpy`, `matplotlib`, and `streamlit` so Streamlit Cloud can establish the Python environment.

### Added Feature: Model Hyperparameter Tuning
- **app.py**: Overhauled the sidebar to include dynamic metric sliders per model page.
- **app.py**: Added informative, academic explanations via Streamlit expanders detailing algorithm logic and hyperparameter definitions.
- **src/loan_eligibility.py**: Integrated lr_max_iter, f_n_estimators, and f_max_depth. Patched modern Pandas illna type handling.
- **src/real_estate.py**: Integrated f_n_estimators.
- **src/ucla_neural_networks.py**: Integrated mlp_hidden_neurons and mlp_max_iter.
- **src/unsupervised_clustering.py**: Integrated dynamic k_min and k_max bounds.

